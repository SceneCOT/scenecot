import os
from datetime import timedelta
from math import ceil

import torch
import torch.nn as nn
from accelerate import DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs, ProjectConfiguration, set_seed
from omegaconf import OmegaConf
from tqdm import trange

from common.io_utils import make_dir
from common.misc import CustomAccelerator
from data.build import build_dataloader_leo
from data.datasets import LeoBase
from evaluator.build import build_eval_leo
from model.scenecot_agent import SceneCOTAgent 
from trainer.build import build_optim, latest_checkpoint, TRAINER_REGISTRY, Tracker
from safetensors.torch import load_file

logger = get_logger(__name__)

model_parallel_classes = (
    nn.parallel.DistributedDataParallel,
    nn.DataParallel,
)

gather_keys = [
    'source', 'scene_id', 'input_txt', 'output_gt', 'output_txt', 'corpus_key', 'situation',   # non-tensor
    'iou_flag', 'tgt_obj_idx', 'pos', 'ori', 'sqa_type', 'type',   # tensor
    'pred_obj_prob', 'grounding_obj_mask_gt', 'obj_masks', 'index'
]


@TRAINER_REGISTRY.register()
class LeoTrainer():
    def __init__(self, cfg):
        set_seed(cfg.rng_seed)
        self.exp_dir = cfg.exp_dir
        self.mode = cfg.mode
        self.cfg = cfg

        # initialize accelerator
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=800))
        print(f"*************************************")
        print(f"num_gpu = {cfg.num_gpu}")
        kwargs = ([ddp_kwargs] if cfg.num_gpu > 1 else []) + [init_kwargs]
        gradient_accumulation_steps = cfg.training.get('gradient_accumulation_steps', 1)

        self.accelerator = CustomAccelerator(
            project_config=ProjectConfiguration(
                project_dir=self.exp_dir,
                automatic_checkpoint_naming=True,
                total_limit=None,   # keep all checkpoints
            ),
            gradient_accumulation_steps=gradient_accumulation_steps,
            log_with=cfg.logger.name,
            kwargs_handlers=kwargs
        )

        # dataset, dataloader, evaluator
        self.eai_task_sources = []
        self.data_loaders = {'train': {}, 'val': {}, 'test': {}}
        self.evaluators = {}
        self.eval_metrics = {}
        self.task_metrics_weights = cfg.task.task_metrics_weights
        self.scene_data_source = LeoBase(cfg)   # shared scene data
        for task_name in cfg.task.keys():
            if task_name in ['training', 'task_metrics_weights']:   # hyperparameters
                continue
            for dataset_name in cfg.task[task_name].keys():
                for mode in cfg.task[task_name][dataset_name].mode:
                    if task_name not in self.data_loaders[mode]:
                        self.data_loaders[mode][task_name] = {}
                    self.data_loaders[mode][task_name][dataset_name] = build_dataloader_leo(
                        cfg=cfg, split=mode, dataset_name=dataset_name, scene_data_source=self.scene_data_source,
                        dataset_wrapper_name=cfg.task[task_name][dataset_name].dataset_wrapper,
                        dataset_wrapper_args=cfg.task[task_name][dataset_name].dataset_wrapper_args,
                        dataloader_args=cfg.task[task_name][dataset_name].train_dataloader_args if mode == 'train' \
                                        else cfg.task[task_name][dataset_name].eval_dataloader_args,
                    )
                if 'evaluator' in cfg.task[task_name][dataset_name]:
                    if task_name not in self.evaluators:
                        self.evaluators[task_name] = {}
                    self.evaluators[task_name][dataset_name] = build_eval_leo(
                        cfg, dataset_name=dataset_name, evaluator_name=cfg.task[task_name][dataset_name].evaluator
                    )
                    if task_name not in self.eval_metrics:
                        self.eval_metrics[task_name] = {}

        assert len(self.data_loaders['train']['leomix']) <= 1, "LEO requires only one training set"

        # prepare dataloaders
        all_loaders, all_loader_keys = [], []
        for mode, mode_loaders in self.data_loaders.items():
            for task, task_loaders in mode_loaders.items():
                for dataset, dataloader in task_loaders.items():
                    all_loader_keys.append((mode, task, dataset))
                    all_loaders.append(dataloader)
        assert len(all_loaders) > 1, "Accelerator should prepare more than one dataloader"
        accelerate_loaders = self.accelerator.prepare(*all_loaders)
        for k, v in zip(all_loader_keys, accelerate_loaders):
            self.data_loaders[k[0]][k[1]][k[2]] = v

        # build model
        self.model = SceneCOTAgent(cfg)

        if 'debug' in cfg and cfg.debug.flag:
            import ipdb; ipdb.set_trace()

        learnable_named_params = self.model.get_learnable_named_params()
        self.accelerator.learn_params_list = list(learnable_named_params.keys())
        optim_params = list(learnable_named_params.values())

        # prepare model, optimizer and scheduler
        train_size = len(list(self.data_loaders['train']['leomix'].values())[0])
        total_steps = ceil(train_size / gradient_accumulation_steps) * cfg.task.training.epochs
        self.optimizer, self.scheduler = build_optim(cfg, optim_params, total_steps=total_steps)
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(self.model, self.optimizer, self.scheduler)

        self.exp_tracker = Tracker(cfg)
        self.accelerator.register_for_checkpointing(self.exp_tracker)

        # load checkpoints
        pretrained_ckpt_loaded = False
        if cfg.pretrained_ckpt_path and os.path.exists(cfg.pretrained_ckpt_path):
            logger.info(f"Load pretrained model from {cfg.pretrained_ckpt_path}")
            self.load(path=cfg.pretrained_ckpt_path, model_only=True)
            pretrained_ckpt_loaded = True
        if self.mode == 'train':
            resume_ckpt = latest_checkpoint(os.path.join(self.exp_dir, 'checkpoints'))
            if resume_ckpt:
                logger.info(f"Train: resume from {resume_ckpt}")
                self.load(path=resume_ckpt, model_only=False)
            elif pretrained_ckpt_loaded:
                logger.info(f"Train: start from {cfg.pretrained_ckpt_path}")
            else:
                logger.info("Train: start from scratch")
        else:
            self_best_ckpt = os.path.join(self.exp_dir, 'best.pth')
            if os.path.exists(self_best_ckpt):
                logger.info(f"Eval: load model from {self_best_ckpt}")
                self.load(path=self_best_ckpt, model_only=True)
            elif pretrained_ckpt_loaded:
                logger.info(f"Eval: load model from {cfg.pretrained_ckpt_path}")
            else:
                logger.info(f"Eval: no checkpoint to load")

        # misc
        self.epochs = cfg.training.epochs
        self.grad_norm = cfg.training.grad_norm
        self.val_interval = cfg.eval.val_interval
        self.num_batch_val = cfg.eval.num_batch_val

        self.accelerator.init_trackers(
            project_name=cfg.name,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            init_kwargs={
                'wandb': {
                    'name': self.exp_tracker.exp_name, 'entity': cfg.logger.entity,
                    'id': self.exp_tracker.run_id, 'resume': 'allow'
                }
            }
        )

    def forward(self, data_dict, inference=False):
        if inference:
            if isinstance(self.model, model_parallel_classes):
                return self.model.module.generate(data_dict)
            else:
                return self.model.generate(data_dict)
        else:
            return self.model(data_dict)

    def backward(self, loss):
        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        if self.grad_norm is not None and self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.optimizer.step()
        self.scheduler.step()

    def train_step(self, epoch):
        logger.info(f"Start training epoch {epoch+1}")
        self.model.train()
        loader = list(self.data_loaders['train']['leomix'].values())[0]
        pbar = trange(len(loader), disable=(not self.accelerator.is_main_process))

        if self.exp_tracker.loader_step > 0:
            logger.info(f"Skip the first {self.exp_tracker.loader_step} batches")
            loader = self.accelerator.skip_first_batches(loader, self.exp_tracker.loader_step)
            pbar.update(self.exp_tracker.loader_step)

        for data_dict in loader:
            with self.accelerator.accumulate(self.model):
                # categorize tasks
                is_eai_data = [(s in self.eai_task_sources) for s in data_dict['source']]
                is_txt_data = [(s not in self.eai_task_sources) for s in data_dict['source']]

                # forward
                data_dict = self.forward(data_dict, inference=False)

                # calculate loss and optimize
                loss = data_dict['loss_gen']
                loss_all = loss.mean()
                self.backward(loss_all)

                # record
                loss_dict = {'overall': loss_all}
                loss_txt = loss[is_txt_data]
                loss_eai = loss[is_eai_data]
                if len(loss_txt) > 0:
                    loss_dict.update({'txt': loss_txt.mean()})
                if len(loss_eai) > 0:
                    loss_dict.update({'eai': loss_eai.mean()})
                if 'loss_grd' in data_dict:
                    loss_grd = data_dict['loss_grd']
                    loss_dict.update({'grd': loss_grd.mean()})
                self.log(loss_dict, mode='train', note='loss')
                self.exp_tracker.step_loader()
                pbar.update(1)

        logger.info(f"Finish training epoch {epoch+1}")

    @torch.no_grad()
    def val_step(self, epoch):
        logger.info(f"Start validation epoch {epoch+1}")
        self.model.eval()
        for task_name in self.data_loaders['val']:
            for dataset_name in self.data_loaders['val'][task_name]:
                loader = self.data_loaders['val'][task_name][dataset_name]
                pbar = trange(len(loader), disable=(not self.accelerator.is_main_process))
                for i, data_dict in enumerate(loader):
                    if i >= self.num_batch_val:
                        break
                    # inference
                    data_dict = self.forward(data_dict, inference=True)

                    # gather, ignore tensors, convert tensors required by evaluator to list
                    gather_dict = {}
                    for k in gather_keys:
                        if k in data_dict:
                            v = data_dict[k]
                            if isinstance(v, torch.Tensor):
                                gather_dict[k] = v.cpu().tolist()
                            else:
                                gather_dict[k] = v

                    gather_dict = self.accelerator.gather_for_metrics(gather_dict)

                    self.evaluators[task_name][dataset_name].update(gather_dict)
                    pbar.update(1)

                _, results = self.evaluators[task_name][dataset_name].record(
                    split='val', is_main_process=self.accelerator.is_main_process
                )

                self.eval_metrics[task_name][dataset_name] = results['target_metric']
                self.log(results, mode='val', note=f'{task_name}/{dataset_name}')
                logger.info(f"{dataset_name}: {results}")
                self.evaluators[task_name][dataset_name].reset()

        # task-wise averages and task-weighted final metric
        overall_results = {'target_metric': 0}
        for task_name, task_results in self.eval_metrics.items():
            overall_results[task_name] = sum(list(task_results.values())) / len(task_results)
            overall_results['target_metric'] += self.task_metrics_weights[task_name] * overall_results[task_name]
        self.log(overall_results, mode='val', note='overall')
        if overall_results['target_metric'] > self.exp_tracker.overall_best_result:
            is_best = True
            self.exp_tracker.overall_best_result = overall_results['target_metric']
        else:
            is_best = False
        logger.info(f"Finish validation epoch {epoch+1}, is_best = {is_best}")
        return is_best

    def _slice_batch(self, data_dict, indices):
        """Helper to slice a batch dictionary based on indices."""
        new_batch = {}
        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                new_batch[k] = v[indices]
            elif isinstance(v, list):
                new_batch[k] = [v[i] for i in indices]
            else:
                # Handle single values or metadata that don't scale with batch
                new_batch[k] = v
        return new_batch
    
    @torch.no_grad()
    def test_step(self):
        logger.info("Start final testing")
        self.model.eval()

        # [MOE CONFIG]
        moe_cfg = getattr(self.cfg, 'moe', {})
        moe_enable = moe_cfg.get('enable', False)
        
        expert1_sd, expert2_sd = None, None
        current_expert = None
        expert1_trigger_types = []

        if moe_enable:
            expert1_path = moe_cfg.get('expert1_path', None)
            expert2_path = moe_cfg.get('expert2_path', None)
            expert1_trigger_types = moe_cfg.get('expert1_trigger_types', [])

            if expert1_path and expert2_path:
                logger.info(f"Param-Swapping MOE Enabled for Batch > 1.")
                
                def get_expert_sd(path):
                    bin_path = os.path.join(path, 'pytorch_model.bin')
                    if os.path.exists(bin_path):
                        sd = torch.load(bin_path, map_location='cpu', weights_only=True)
                    else:
                        sd = load_file(os.path.join(path, 'model.safetensors'))
                    learnable_keys = [k.replace('module.', '') for k in self.model.get_learnable_named_params().keys()]
                    return {k.replace('module.', ''): v for k, v in sd.items() if k.replace('module.', '') in learnable_keys}

                expert1_sd = get_expert_sd(expert1_path)
                expert2_sd = get_expert_sd(expert2_path)
            else:
                moe_enable = False

        for task_name in self.data_loaders['test']:
            for dataset_name in self.data_loaders['test'][task_name]:
                loader = self.data_loaders['test'][task_name][dataset_name]
                pbar = trange(len(loader), disable=(not self.accelerator.is_main_process))
                
                for data_dict in loader:
                    
                    # --- Standard Logic (No MoE) ---
                    if not moe_enable or not (expert1_sd and expert2_sd):
                        self._process_batch_inference(data_dict, task_name, dataset_name)
                        pbar.update(1)
                        continue

                    # --- MoE Logic (Mixed Batch Handling) ---
                    # 1. Identify indices for each expert
                    batch_types = data_dict['type'] # Assuming this is a list of strings
                    idx_exp1 = [i for i, t in enumerate(batch_types) if t in expert1_trigger_types]
                    idx_exp2 = [i for i, t in enumerate(batch_types) if t not in expert1_trigger_types]

                    # 2. Process Expert 1 Samples
                    if len(idx_exp1) > 0:
                        if current_expert != 'expert1':
                            if isinstance(self.model, model_parallel_classes):
                                self.model.module.load_state_dict(expert1_sd, strict=False)
                            else:
                                self.model.load_state_dict(expert1_sd, strict=False)
                            current_expert = 'expert1'
                        
                        sub_batch = self._slice_batch(data_dict, idx_exp1)
                        self._process_batch_inference(sub_batch, task_name, dataset_name)

                    # 3. Process Expert 2 Samples
                    if len(idx_exp2) > 0:
                        if current_expert != 'expert2':
                            if isinstance(self.model, model_parallel_classes):
                                self.model.module.load_state_dict(expert2_sd, strict=False)
                            else:
                                self.model.load_state_dict(expert2_sd, strict=False)
                            current_expert = 'expert2'
                            
                        sub_batch = self._slice_batch(data_dict, idx_exp2)
                        self._process_batch_inference(sub_batch, task_name, dataset_name)
                    
                    pbar.update(1)

                _, results = self.evaluators[task_name][dataset_name].record(
                    split='test', is_main_process=self.accelerator.is_main_process
                )

                self.log(results, mode='test', note=f'{task_name}/{dataset_name}')
                logger.info(f"{dataset_name}: {results}")
                self.evaluators[task_name][dataset_name].reset()

        logger.info("Finish testing")

    def _process_batch_inference(self, batch_data, task_name, dataset_name):
        """Internal helper to run forward pass and update evaluator for a (sub)batch."""
        out_dict = self.forward(batch_data, inference=True)

        gather_dict = {}
        for k in gather_keys:
            if k in out_dict:
                v = out_dict[k]
                if isinstance(v, torch.Tensor):
                    gather_dict[k] = v.cpu().tolist()
                else:
                    gather_dict[k] = v

        gather_dict = self.accelerator.gather_for_metrics(gather_dict)
        self.evaluators[task_name][dataset_name].update(gather_dict)

    def log(self, results, mode='train', note='default'):
        log_dict = {}
        for key, val in results.items():
            log_dict[f'{mode}/{note}/{key}'] = val

        if mode == 'train':
            lrs = self.scheduler.get_lr()
            for i, lr in enumerate(lrs):
                log_dict[f'train/lr/group_{i}'] = lr

        self.accelerator.log(log_dict)

    def save(self, name='best.pth', model_only=False):
        if model_only:
            path = os.path.join(self.exp_dir, name)
            make_dir(path)
            model_state_dict = self.accelerator.get_state_dict(self.model)
            # automatically filter non-learnable params, and save on main_process
            self.accelerator.save(model_state_dict, os.path.join(path, 'pytorch_model.bin'))
        else:
            self.accelerator.save_state()   # automatic_checkpoint_naming = True -> self.exp_dir / checkpoints

    def load(self, path, model_only=False):
        if model_only:
            if os.path.exists(os.path.join(path, 'pytorch_model.bin')):
                model_state_dict = torch.load(os.path.join(path, 'pytorch_model.bin'), weights_only=True)
            elif os.path.exists(os.path.join(path, 'model.safetensors')):
                model_state_dict = load_file(os.path.join(path, 'model.safetensors'))
            if isinstance(self.model, model_parallel_classes):
                self.model.module.load_state_dict(model_state_dict, strict=False)
            else:
                self.model.load_state_dict(model_state_dict, strict=False)
        else:
            # resume training
            self.accelerator.load_state(path, strict=False)
            self.accelerator.project_configuration.iteration = int(str(path)[-1]) + 1
        logger.info(f"Successfully loaded from {str(path)}, load_model_only = {model_only}")

    def run(self):
        if self.mode == 'train':
            start_epoch = self.exp_tracker.epoch
            for epoch in range(start_epoch, self.epochs):
                self.train_step(epoch)
                if (epoch + 1) % self.val_interval == 0:
                    is_best = self.val_step(epoch)
                    if is_best:
                        self.save('best.pth', model_only=True)   # save the best checkpoint
                        self.accelerator.wait_for_everyone()

                self.exp_tracker.step()
                self.save(model_only=False)   # automatic checkpointing
                self.accelerator.wait_for_everyone()

            # load best checkpoint for test
            logger.info("Training finished, load best checkpoint for testing")
            self.load(os.path.join(self.exp_dir, 'best.pth'), model_only=True)

        self.test_step()
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()#
