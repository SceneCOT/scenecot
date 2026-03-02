import json
import os
import random

import numpy as np
import torch
from accelerate.logging import get_logger
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from transformers import AutoTokenizer, LlamaTokenizer, AutoProcessor
from .scan_data_loader import ScanDataLoader
from scipy import sparse
from .data_utils import LabelConverter, convert_pc_to_box, eval_ref_one_sample, construct_bbox_corners, point_cloud_iou
import math

from .build import DATASET_REGISTRY
from .data_utils import PromptType, build_rotate_mat, convert_person_view_1_to_2, \
                        convert_person_view_2_to_1, get_sqa_question_type
from .cot_utils import GRD_TOKEN_TXT, replace_cot_indicators_with_tokens
from model.llava.llava_llama import load_pretrained_llava
import re

from PIL import Image
from model.llava.vision_tower import process_images, process_images_qwen_2_5_vl

logger = get_logger(__name__)

# len(tokenized_sentence) / len(sentence)
LLAMA_TOKEN_SENT_RATIO = 0.24

LEOMIX_REQUIRED_KEYS_BASE = [
    'source',
    'scene_id',
    'scene_prompt_tokens',
    'egoview_prompt_tokens',
    'input_txt',
    # 'scene_pcds',
    # 'scene_3d_fts',
    # 'scene_2d_fts',
    # 'scene_voxs',
    # 'vox_3d_fts',
    # 'vox_2d_fts',
    # 'vox_2d_fts_mask',
    # 'obj_2d_fts',
    # 'thought_pixel_values',
    # 'obj_imgs',
    'all_obj_imgs',
    'gt_img',
    'obj_ids',
    'thought_image_prompt_tokens',
    # 'obj_pcds',
    'obj_fts_img',
    'obj_fts_vox',
    'obj_fts',
    'agent_pos_ori',   # (pos, ori)
    'index',
    # 'obj_masks',   # filled by dataset wrapper
    'obj_locs',   # xyzwdh
    # 'tgt_obj_idx',   # int, target object index
    # 'tgt_obj_mask',   # binary mask (N,), 1 denoting points of target object
    'grounding_obj_mask_gt',   # binary mask (N,), 1 denoting points of target object
    # 'anchor_loc',   # a point sampled within the area of target object
    'pos',   # agent's position, (xyz)
    'ori',   # agent's orientation, (cos, sin)
    'img_fts',
    'img_masks',
    'output_gt',
    'type',
    'gt_labels',
    'oracle_obj_prob',
    'oracle_obj_loc',
]

scan_cache_data = {}

class LeoBase(Dataset):
    r""" Unified input format:
    `scene_prompt_tokens` + <scene_tokens> + `egoview_prompt_tokens` + <img_tokens> + `input_tokens`
    `output_gt` will be tokenized during training and attached to the end of input sequence for computing loss
    """
    def __init__(self, cfg):
        super().__init__()
        self.base_dir = cfg.data.sceneverse_base
        self.inst_anno_type = 'default'
        self.mmscan_inst_mask_base = cfg.data.mmscan_inst_mask_base

        # self.scene_3d_feat_base = cfg.data.scene_3d_feat_base
        # self.scene_2d_feat_base = cfg.data.scene_2d_feat_base

        self.voxel_feat_type_3d = cfg.vision3d.feat_type_3d
        self.voxel_feat_type_2d = cfg.vision3d.feat_type_2d
        self.obj_feat_2d_base = cfg.data.obj_feat_2d_base
        self.scene_3d_feat_dim = getattr(cfg.vision3d.feat_dim_3d, self.voxel_feat_type_3d)
        self.scene_2d_feat_dim = getattr(cfg.vision3d.feat_dim_2d, self.voxel_feat_type_2d)
        self.voxel_size = cfg.vision3d.voxel_size

        self.obj_num_points = cfg.data.obj_num_points
        self.obj_img_feat_dim = cfg.data.obj_img_feat_dim
        self.obj_vox_feat_dim = cfg.data.obj_vox_feat_dim
        self.obj_feat_base = cfg.data.obj_feat_base
        self.scan_family_base = cfg.data.scan_family_base
        self.nms_iou_threshold = getattr(cfg.data, 'nms_iou_threshold', 0.25)

        self.img_size = cfg.data.img_size

        if 'vicuna' in cfg.llm.name.lower():
            self.tokenizer = LlamaTokenizer.from_pretrained(cfg.llm.cfg_path, truncation_side=cfg.llm.truncation_side)
            self.tokenizer.add_special_tokens({'pad_token': '<PAD>'})
            self.scene_prompt = "3D scene:"
            self.egoview_prompt = "Ego-view image:"
            self.input_txt = "USER: {instruction} ASSISTANT:"
        elif 'opt' in cfg.llm.name.lower() or 't5' in cfg.llm.name.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm.cfg_path, truncation_side=cfg.llm.truncation_side)
            self.scene_prompt = "3D scene:"
            self.egoview_prompt = "Ego-view image:"
            self.input_txt = "USER: {instruction} ASSISTANT:"
        elif 'llama3' in cfg.llm.name.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm.cfg_path, truncation_side=cfg.llm.truncation_side)
            self.tokenizer.add_special_tokens({'pad_token': '<PAD>'})
            self.scene_prompt = "<|start_header_id|>system<|end_header_id|>\n\n3D scene:"
            self.egoview_prompt = "Ego-view image:"
            self.input_txt = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>" + \
                             "<|start_header_id|>assistant<|end_header_id|>\n\n"
        elif 'gemma' in cfg.llm.name.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm.cfg_path, truncation_side=cfg.llm.truncation_side)
            self.scene_prompt = "3D scene:"
            self.egoview_prompt = "Ego-view image:"
            self.input_txt = "<start_of_turn>user\n{instruction}<end_of_turn>\n" + \
                             "<start_of_turn>model\n"
        elif 'llava' in cfg.llm.name.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm.cfg_path, use_fast=False, truncation_side=cfg.llm.truncation_side)
            self.scene_prompt = "3D scene:"
            self.egoview_prompt = "Ego-view image:"
            self.input_txt = "USER: {instruction} ASSISTANT:"
            self.input_txt_with_oracle_thought = "USER: {instruction} ASSISTANT:{oracle_thought}"
        elif "qwen" in cfg.llm.name.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm.cfg_path, truncation_side=cfg.llm.truncation_side)
            self.use_chat_template = True
            self.scene_prompt = "3D scene:"
            self.egoview_prompt = "Ego-view image:"
            with open(os.path.join(cfg.llm.cfg_path, 'chat_template.json')) as f:
                self.chat_template = json.load(f)['chat_template']
            self.chat_template_kwargs = {
                'tools': None, 'documents': None, 'add_generation_prompt': True,
                'continue_final_message': False, 'return_assistant_tokens_mask': False,
            }
        else:
            raise NotImplementedError()

        self.thought_image_prompt = "The images of the target objects in the question: "
        
        self.scene_prompt_tokens = self.tokenizer(
            [self.tokenizer.bos_token + self.scene_prompt if self.tokenizer.bos_token else self.scene_prompt],
            return_tensors='pt', padding='longest', add_special_tokens=False,
        ).input_ids[0]
        self.egoview_prompt_tokens = self.tokenizer(
            [self.egoview_prompt], return_tensors='pt', padding='longest', add_special_tokens=False
        ).input_ids[0]
        self.thought_image_prompt_tokens = self.tokenizer(
            [self.thought_image_prompt], return_tensors='pt', padding='longest', add_special_tokens=False
        ).input_ids[0]

        self.leomix_required_keys = LEOMIX_REQUIRED_KEYS_BASE
        # self.use_img_feat = (self.domain in cfg.data.obj_feat_base) and cfg.vision3d.use_img_feat
        # self.use_vox_feat = (self.domain in cfg.data.obj_feat_base) and cfg.vision3d.use_vox_feat
        self.use_img_feat = cfg.vision3d.use_img_feat
        self.use_vox_feat = cfg.vision3d.use_vox_feat
        if self.use_img_feat:
            self.leomix_required_keys += ['obj_fts_img']
        if self.use_vox_feat:
            self.leomix_required_keys += ['obj_fts_vox']
        if cfg.vision3d.name.lower() == 'pq3d':
            self.requires_pq3d_input = True
            self.leomix_required_keys += ['prompt', 'prompt_pad_masks', 'prompt_type']
            print('Using PQ3D input')
        else:
            self.requires_pq3d_input = False
        self.scene_data = {}


        model_path = cfg.llm.cfg_path
        if 'llava' in cfg.llm.name.lower():
            vision_model_path = cfg.llm.vision_model_path
            _, _, self.img_transform = load_pretrained_llava(model_path=model_path, vision_tower_path=vision_model_path)
            self.img_processer_type = 'openai/clip-vit-large-patch14-336' 
        elif 'qwen' in cfg.llm.name.lower():
            processor  = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
            self.img_transform = processor.image_processor
            self.img_processer_type = 'qwen_2_5_vl'
        if 'llava' in cfg.llm.name.lower():
            vision_model_path = cfg.llm.vision_model_path
            _, _, self.img_transform = load_pretrained_llava(model_path=model_path, vision_tower_path=vision_model_path)
            self.img_processer_type = 'openai/clip-vit-large-patch14-336' 
        elif 'qwen' in cfg.llm.name.lower():
            processor  = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
            self.img_transform = processor.image_processor
            self.img_processer_type = 'qwen_2_5_vl'

        # image processor
        # self.img_processer_type = cfg.img_processer_type
        # self.img_processer_type = cfg.img_processer_type

    @staticmethod
    def remove_answer_tag_content(text):
        return re.sub(r'<answer>.*?</answer>', '', text, flags=re.DOTALL)

    def get_prompts(self, instruction, situation="", dialogue=None, oracle_thought=None):

        if getattr(self, 'use_chat_template', False):
            input_txt = self.tokenizer.apply_chat_template(
                    dialogue, chat_template=self.chat_template, tokenize=False, return_dict=False, **self.chat_template_kwargs
                )
            return {'input_txt': input_txt}
        
        else:
            if dialogue is None:
                if situation:
                    instruction = f"{situation} {instruction}"
                if oracle_thought is not None:
                    input_txt = self.input_txt_with_oracle_thought.format(instruction=instruction, oracle_thought=oracle_thought)
                else:
                    input_txt = self.input_txt.format(instruction=instruction)
            else:
                input_txt = dialogue   # dialogue should be prepared in advance

            return {
                'scene_prompt_tokens': self.scene_prompt_tokens,
                'egoview_prompt_tokens': self.egoview_prompt_tokens,
                'thought_image_prompt_tokens': self.thought_image_prompt_tokens,  # used by cot
                'input_txt': input_txt,
            }

    def check_output_and_fill_dummy(self, data_dict):
        if 'tgt_obj_idx' not in data_dict:
            data_dict['tgt_obj_idx'] = -100
        # if 'tgt_obj_mask' not in data_dict or data_dict['tgt_obj_mask'] is None:
        #     data_dict['tgt_obj_mask'] = torch.zeros(data_dict['scene_voxs'].shape[0]).bool()
        if 'pos' not in data_dict:
            data_dict['pos'] = torch.zeros(3)
        if 'ori' not in data_dict:
            data_dict['ori'] = torch.FloatTensor([0, 1])
        if 'img_fts' not in data_dict:
            data_dict['img_fts'] = torch.zeros(3, *self.img_size)   # currently hardcode to 224x224
        if 'img_masks' not in data_dict:
            data_dict['img_masks'] = 0
        # if 'thought_pixel_values' not in data_dict:
        #     data_dict['thought_pixel_values'] = torch.zeros(20, 3, *self.img_size)  # currently hardcode to 224x224
        if 'obj_ids' not in data_dict:
            data_dict['obj_ids'] = torch.zeros(1).long()
        if 'type' not in data_dict:
            data_dict['type'] = ''
        # if 'obj_2d_fts' not in data_dict:
        #     data_dict['obj_2d_fts'] = torch.zeros(1, self.scene_2d_feat_dim)
        if 'obj_locs' not in data_dict:
            data_dict['obj_locs'] = torch.zeros(1, 6)
        if 'grounding_obj_mask_gt' not in data_dict:
            data_dict['grounding_obj_mask_gt'] = torch.zeros(data_dict['scene_voxs'].shape[0]).bool()
        if 'gt_labels' not in data_dict:
            data_dict['gt_labels'] = []
        if 'oracle_obj_prob' not in data_dict:
            data_dict['oracle_obj_prob'] = ''
        if 'oracle_obj_loc' not in data_dict:
            data_dict['oracle_obj_loc'] = ''
        if 'agent_pos_ori' not in data_dict:
            data_dict['agent_pos_ori'] = torch.zeros(2, 3)
        if 'index' not in data_dict:
            data_dict['index'] = -1

        for key in self.leomix_required_keys:
            if key not in data_dict:
                raise ValueError(f"Key {key} is missing in LeoMix data_dict")
        return data_dict

    def get_one_full_img(self, scan_id, inst_id, is_debug = False, transform=None):
        img_file_name = f'{scan_id}_inst{inst_id}_0.jpg'
        img_path = os.path.join(self.cfg.data.obj_img_base[self.domain], img_file_name)
        if not os.path.exists(img_path):
            # print(f"Image file {img_path} not found.")
            return None

        if self.img_processer_type in ['openai/clip-vit-base-patch32']:
            img = self.img_transform(img)
            img = img['pixel_values'][0]
        # elif self.img_processer_type in ['navigation_img_processer']:
        #     img = np.array(img)
        #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        #     ### debug ###
        #     if is_debug:
        #         cv2.imwrite(f'img_{one_bbox["label"]}_img.png', img)

        #     img = preprocess_2d(img, size=self.img_process_args.tgt_img_size)
        #     img = torch.from_numpy(img)
        elif self.img_processer_type in ['openai/clip-vit-large-patch14-336']:   # only for llava-1.5
            image = Image.open(img_path).convert('RGB')
            img = process_images([image], transform)[0]
        elif self.img_processer_type in ['qwen_2_5_vl']:   # only for qwen2.5-vl
            image = Image.open(img_path).convert('RGB')
            img = process_images_qwen_2_5_vl([image], transform)[0]
        elif self.img_processer_type in ['qwen_2_5_vl']:   # only for qwen2.5-vl
            image = Image.open(img_path).convert('RGB')
            img = process_images_qwen_2_5_vl([image], transform)[0]
        else:
            raise NotImplementedError

        return img

    def load_scene(self, domain, scene_id):
        if hasattr(self, '') and self.scene_data_source is not None:
            # get from scene_data_source
            return self.scene_data_source.load_scene(domain, scene_id)

        # use self.scene_data
        if domain not in self.scene_data:
            self.scene_data[domain] = {}
        if scene_id not in self.scene_data[domain]:
            pcd_path = os.path.join(self.base_dir, domain, 'scan_data', 'pcd_with_global_alignment', f'{scene_id}.pth')
            pcd_data = torch.load(pcd_path, weights_only=False)
            points, colors, points_inst_ids = pcd_data[0], pcd_data[1], pcd_data[-1]
            colors = colors / 127.5 - 1
            pcds = np.concatenate([points, colors], axis=1)

            # build obj_pcds @ gt
            if self.inst_anno_type == 'default':
                inst_id2label_path = os.path.join(self.base_dir, domain, 'scan_data', 'instance_id_to_label', f'{scene_id}.pth')
                inst_id2label = torch.load(inst_id2label_path, weights_only=False)
            elif self.inst_anno_type == 'mmscan':
                inst_id2label_path = os.path.join(self.mmscan_inst_mask_base, domain, f'{scene_id}.pth')
                points_inst_ids, inst_id2label = torch.load(inst_id2label_path, weights_only=False)
            else:
                raise ValueError("Unexpected instance annotation type")

            obj_pcds = {}
            # bg_indices = np.full((points.shape[0]), 0, dtype=np.bool_)
            for inst_id, inst_label in inst_id2label.items():
                mask = (points_inst_ids == inst_id)
                if not mask.any():
                    continue
                obj_pcds[inst_id] = pcds[mask]
                # if inst_label in ['wall', 'floor', 'ceiling']:
                #     bg_indices[mask] = True

            scene = {
                # 'scene_pcds': pcds, 'scene_inst_ids': points_inst_ids,
                'obj_pcds_gt': obj_pcds,
                'obj_labels': inst_id2label,
                # 'bg_indices': bg_indices,
                'scene_center': (points.max(0) + points.min(0)) / 2,   # for SQA3D transform
            }

            # if hasattr(self, 'pc_type') and self.pc_type == 'pred':
            if hasattr(self, 'pc_type'):
                mask_path = os.path.join(self.obj_feat_base[domain], 'mask', f'{str(scene_id)}.mask.npz')
                mask_score_path = os.path.join(self.obj_feat_base[domain], 'mask', f'{str(scene_id)}.score.npy')
                obj_masks = np.array(sparse.load_npz(mask_path).todense())[:100, :]   # the maximum of the raw masks is 100
                obj_masks_len = len(obj_masks)
                obj_masks_scores = np.array(np.load(mask_score_path))[:obj_masks_len]
                label_id_path = os.path.join(self.obj_feat_base[domain], 'mask', f'{str(scene_id)}.label.npy')
                label_ids = np.load(label_id_path)
                obj_pcds_pred = {}
                obj_label_names_pred = {}
                self.iou_threshold = getattr(self, 'iou_thres', 0.25)

                self.label_converter = LabelConverter(os.path.join(self.scan_family_base,
                                            "annotations/meta_data/scannetv2-labels.combined.tsv"))
                # # apply NMS
                obj_pcds_pred_raw = {}
                for i in range(obj_masks.shape[0]):
                    mask = (obj_masks[i] == 1)
                    if not mask.any():
                        continue
                    # obj_pcds_pred.update({i: {'pcds':pcds[mask], 'label_name':self.label_converter.scannet_raw_id_to_raw_name[label_ids[i]]}})
                    obj_pcds_pred_raw[i] = pcds[mask]

                nms_iou_threshold = getattr(self, 'nms_iou_threshold', 0.25)
                nms_obj_masks = []
                obj_label_names_pred = {}
                # # apply nms
                order = sorted(range(len(obj_masks_scores)), key=lambda k: -obj_masks_scores[k])
                nms_ids_cnt = 0
                # while order:
                #     i = order.pop(0)
                #     if i not in obj_pcds_pred_raw:  # aviod dummy key
                #         continue
                #     nms_obj_masks.append(obj_masks[i])
                #     obj_label_names_pred[nms_ids_cnt] = self.label_converter.scannet_raw_id_to_raw_name[label_ids[i]]
                #     obj_center_i, obj_box_size_i = convert_pc_to_box(obj_pcds_pred_raw[i])
                #     nms_ids_cnt += 1
                #     for j in order.copy():
                #         if j not in obj_pcds_pred_raw:
                #             # remove j to avoid KeyError
                #             order.remove(j)
                #             continue
                #         obj_center_j, obj_box_size_j = convert_pc_to_box(obj_pcds_pred_raw[j])
                #         iou = eval_ref_one_sample(
                #                     construct_bbox_corners(obj_center_j, obj_box_size_j),
                #                     construct_bbox_corners(obj_center_i, obj_box_size_i)
                #                 )
                #         if iou > nms_iou_threshold:
                #             order.remove(j)
                
                # mask based NMS
                while order:
                    i = order.pop(0)
                    if i not in obj_pcds_pred_raw:  
                        continue
                    
                    # Keep the current mask
                    nms_obj_masks.append(obj_masks[i])
                    try:
                        obj_label_names_pred[nms_ids_cnt] = self.label_converter.scannet_raw_id_to_raw_name[label_ids[i]]
                    except:
                        print(f"label_ids: {label_ids}")
                        print(f"label_ids[i]: {label_ids[i]}")
                        ValueError(f"label_ids[i]: {label_ids[i]} not in label_converter")
                    nms_ids_cnt += 1
                    
                    for j in order.copy():
                        if j not in obj_pcds_pred_raw:
                            order.remove(j)
                            continue

                        # Compute mask-based IoU
                        iou = point_cloud_iou(obj_masks[i], obj_masks[j])
                        
                        # Apply hierarchy-aware NMS (allow overlap if objects are different categories)
                        if iou > nms_iou_threshold and label_ids[i] == label_ids[j]:
                            order.remove(j)

                nms_obj_masks = np.array(nms_obj_masks)
                assert nms_ids_cnt > 0, f"nms_ids_cnt should be > 0, but got {nms_ids_cnt}"
                # print(f'shape of nms_obj_masks: {nms_obj_masks.shape}, nms_ids_cnt: {nms_ids_cnt}')
                
                obj_pcds_pred = {}
                for i in range(nms_obj_masks.shape[0]):
                    mask = (nms_obj_masks[i] == 1)
                    if not mask.any():
                        continue
                    obj_pcds_pred[i] = pcds[mask]

                # for i in range(obj_masks.shape[0]):
                #     mask = (obj_masks[i] == 1)
                #     if not mask.any():
                #         continue
                #     # obj_pcds_pred.update({i: {'pcds':pcds[mask], 'label_name':self.label_converter.scannet_raw_id_to_raw_name[label_ids[i]]}})
                #     obj_pcds_pred[i] = pcds[mask]
                #     if i >= len(label_ids):
                #         print(f'mask id {i} not in label_ids')
                #     if label_ids[i] not in self.label_converter.scannet_raw_id_to_raw_name:
                #         print(f'label id {label_ids[i]} not in label_converter')
                #         print(f'scan_id: {scene_id}')
                #     obj_label_names_pred[i] = self.label_converter.scannet_raw_id_to_raw_name[label_ids[i]]
                
                # get objects within the iou threshold
                gt_to_pred_id = {}
                for gt_id, gt_pcd in obj_pcds.items():
                    iou_max = 0
                    tgt_obj_id_pred = -1
                    gt_center, gt_box_size = convert_pc_to_box(gt_pcd)
                    for pred_id, pred_pcd in obj_pcds_pred.items():
                        obj_center, obj_box_size = convert_pc_to_box(pred_pcd)
                        current_iou = eval_ref_one_sample(
                    construct_bbox_corners(obj_center, obj_box_size),
                    construct_bbox_corners(gt_center, gt_box_size)
                )
                        if current_iou > iou_max:
                            iou_max = current_iou
                            tgt_obj_id_pred = pred_id
                    iou_flag = 1 if iou_max > self.iou_threshold else 0
                    if tgt_obj_id_pred != -1 and iou_max > 0:
                        gt_to_pred_id[gt_id] = (tgt_obj_id_pred, iou_flag)
                
                scene['gt_to_pred_id'] = gt_to_pred_id
                scene['obj_pcds_pred'] = obj_pcds_pred
                scene['obj_label_names_pred'] = obj_label_names_pred
                # for gt_id in gt_to_pred_id.keys():
                #     print(f'gt_id: {gt_id}, pred_id: {gt_to_pred_id[gt_id][0]}, iou_flag: {gt_to_pred_id[gt_id][1]}')
                #     print(f'gt_label: {inst_id2label[gt_id]}, pred_label: {obj_label_names_pred[gt_to_pred_id[gt_id][0]]}')
                #     print('\n')

                obj_locs_pred = []
                for pred_id, pred_pcd in obj_pcds_pred.items():
                    obj_center = pred_pcd[:, :3].mean(0)
                    obj_bounds_min = pred_pcd[:, :3].min(0)
                    obj_bounds_max = pred_pcd[:, :3].max(0)
                    obj_size = obj_bounds_max - obj_bounds_min
                    obj_locs_pred.append(np.concatenate([obj_center, obj_size], 0))
                obj_locs_pred = torch.from_numpy(np.array(obj_locs_pred)).float()
                scene['obj_locs_pred'] = obj_locs_pred

            # scene 3D features
            # if self.domain in self.scene_3d_feat_base:
            #     scene_feat_path = os.path.join(self.scene_3d_feat_base[self.domain], f'{scene_id}.pt')
            #     scene_3d_fts = torch.load(scene_feat_path, weights_only=False).float()
            #     assert scene_3d_fts.shape[0] == points_inst_ids.shape[0]
            # else:
            #     raise NotImplementedError()   # TODO

            # scene['scene_3d_fts'] = scene_3d_fts

            # # scene 2D features
            # scene_2d_fts = torch.zeros(points_inst_ids.shape[0], self.scene_2d_feat_dim)
            # if self.domain in self.scene_2d_feat_base:
            #     scene_feat_path = os.path.join(self.scene_2d_feat_base[self.domain], f'{scene_id}_0.pt')
            #     scene_feat = torch.load(scene_feat_path, weights_only=False)
            #     scene_feat_value = scene_feat['feat'].float()
            #     scene_feat_mask = scene_feat['mask_full'].cpu().numpy()

            #     if scene_feat_mask.shape[0] == points_inst_ids.shape[0]:
            #         scene_2d_fts[scene_feat_mask] = scene_feat_value

            # scene['scene_2d_fts'] = scene_2d_fts

            # if domain in self.voxel_feat_base:
            #     vox_feat_3d_path = os.path.join(self.voxel_feat_base[domain], self.voxel_feat_type_3d, f'{scene_id}.pt')
            #     scene_voxs_1, vox_3d_fts = torch.load(vox_feat_3d_path, weights_only=False)  # mpec feature
            #     vox_feat_2d_path = os.path.join(self.voxel_feat_base[domain], self.voxel_feat_type_2d, f'{scene_id}.pt')
            #     scene_voxs_2, vox_2d_fts = torch.load(vox_feat_2d_path, weights_only=False)
            #     # assert (scene_voxs_1 == scene_voxs_2).all()
            # else:
            #     raise NotImplementedError()   # TODO
            
            if domain in self.obj_feat_2d_base:
                obj_feat_2d_path = os.path.join(self.obj_feat_2d_base[domain], f'{scene_id}.pt')
                obj_pcd_feat = torch.load(obj_feat_2d_path, weights_only=False, map_location="cpu")  # NOTE: some of the data points are stored on GPU, adding cpu map_location to avoid error
                obj_2d_fts = []
                obj_locs = []
                for inst_id, item in obj_pcd_feat.items():
                    if inst_id not in obj_pcds:
                        continue
                    obj_2d_fts.append(item['features'])
                    obj_pcd = obj_pcds[inst_id]
                    obj_center = obj_pcd[:, :3].mean(0)
                    obj_bounds_min = obj_pcd[:, :3].min(0)
                    obj_bounds_max = obj_pcd[:, :3].max(0)
                    obj_size = obj_bounds_max - obj_bounds_min
                    obj_locs.append(np.concatenate([obj_center, obj_size], 0))
                obj_2d_fts = torch.from_numpy(np.stack(obj_2d_fts, 0)).float()
                obj_locs = torch.from_numpy(np.array(obj_locs)).float()
                assert obj_2d_fts.shape[0] == obj_locs.shape[0]
                scene['obj_locs_gt'] = obj_locs
                scene['obj_2d_fts'] = obj_2d_fts

            # scene.update({'scene_voxs': scene_voxs_1, 'vox_3d_fts': vox_3d_fts, 'vox_2d_fts': vox_2d_fts})
            # scene.update({'scene_voxs': scene_voxs_2, 'vox_3d_fts': vox_3d_fts, 'vox_2d_fts': vox_2d_fts})

            # object features pq3d @ gt
            obj_fts_img_gt = {i: torch.zeros(self.obj_img_feat_dim) for i in obj_pcds.keys()}
            obj_fts_vox_gt = {i: torch.zeros(self.obj_vox_feat_dim) for i in obj_pcds.keys()}
            if self.domain in self.obj_feat_base:
                feat_pth = os.path.join(self.obj_feat_base[self.domain], f'image_obj_feat_gt', f'{scene_id}.pth')
                if os.path.exists(feat_pth):
                    feat_dict = torch.load(feat_pth, weights_only=False)
                    for i in obj_fts_img_gt.keys():
                        if i in feat_dict:
                            obj_fts_img_gt[i] = feat_dict[i]

                feat_pth = os.path.join(self.obj_feat_base[self.domain], f'voxel_obj_feat_gt', f'{scene_id}.pth')
                if os.path.exists(feat_pth):
                    feat_dict = torch.load(feat_pth, weights_only=False)
                    for i in obj_fts_vox_gt.keys():
                        if i in feat_dict:
                            obj_fts_vox_gt[i] = feat_dict[i]

            scene['obj_fts_img_gt'] = obj_fts_img_gt
            scene['obj_fts_vox_gt'] = obj_fts_vox_gt

            # object features pq3d @ pred
            if hasattr(self, 'pc_type') and self.pc_type == 'pred':
                obj_fts_img_pred = {i: torch.zeros(self.obj_img_feat_dim) for i in obj_pcds_pred.keys()}
                obj_fts_vox_pred = {i: torch.zeros(self.obj_vox_feat_dim) for i in obj_pcds_pred.keys()}
                if self.domain in self.obj_feat_base:
                    feat_pth = os.path.join(self.obj_feat_base[self.domain], f'image_obj_feat_pred', f'{scene_id}.pth')
                    if os.path.exists(feat_pth):
                        feat_dict = torch.load(feat_pth, weights_only=False)
                        for i in obj_fts_img_pred.keys():
                            if i in feat_dict:
                                obj_fts_img_pred[i] = feat_dict[i]

                    feat_pth = os.path.join(self.obj_feat_base[self.domain], f'voxel_obj_feat_pred', f'{scene_id}.pth')
                    if os.path.exists(feat_pth):
                        feat_dict = torch.load(feat_pth, weights_only=False)
                        for i in obj_fts_vox_pred.keys():
                            if i in feat_dict:
                                obj_fts_vox_pred[i] = feat_dict[i]
                scene['obj_fts_img_pred'] = obj_fts_img_pred
                scene['obj_fts_vox_pred'] = obj_fts_vox_pred

            self.scene_data[domain][scene_id] = scene

        return self.scene_data[domain][scene_id].copy()

    def check_3rscan_obj_feat_path(self, scene_id, img=True, vox=True):
        def check_feat(key):
            return os.path.exists(os.path.join(self.obj_feat_base['3RScan'], key, f'{scene_id}.pth'))
        if ( img and not ( check_feat('image_obj_feat_gt') and check_feat('image_obj_feat_pred') ) ):
            img_flag = False
        else:
            img_flag = True
        if ( vox and not ( check_feat('voxel_obj_feat_gt') and check_feat('voxel_obj_feat_pred') ) ):
            vox_flag = False
        else:
            vox_flag = True
        return img_flag and vox_flag

    def preprocess_pcd(self, scene_pcds, obj_pcds, tgt_obj_idx=-100, situation=None, rot_aug=True):
        extra_output = None   # anchor_loc for object refer, pos and ori for SQA
        # rotate scene
        rot_matrix = build_rotate_mat(self.split, rot_aug=rot_aug)
        if rot_matrix is not None:
            scene_pcds[:, :3] = np.matmul(scene_pcds[:, :3], rot_matrix.transpose())

        # normalize pc and calculate location
        obj_locs = []
        for i, obj_pcd in enumerate(obj_pcds):
            if rot_matrix is not None:
                obj_pcd[:, :3] = np.matmul(obj_pcd[:, :3], rot_matrix.transpose())

            obj_center = obj_pcd[:, :3].mean(0)
            obj_bounds_min = obj_pcd[:, :3].min(0)
            obj_bounds_max = obj_pcd[:, :3].max(0)
            obj_size = obj_bounds_max - obj_bounds_min
            obj_locs.append(np.concatenate([obj_center, obj_size], 0))
            if i == tgt_obj_idx:
                # Select a loc within the obj bbox as the anchor.
                extra_output = obj_pcd[:, :3].min(0) + np.random.rand(3) * obj_size

            # subsample
            pcd_idxs = np.random.choice(len(obj_pcd), size=self.obj_num_points,
                                        replace=len(obj_pcd) < self.obj_num_points)
            obj_pcd = obj_pcd[pcd_idxs]

            # normalize
            obj_pcd[:, :3] = obj_pcd[:, :3] - obj_center
            max_dist = np.sqrt((obj_pcd[:, :3]**2).sum(1)).max()
            if max_dist < 1e-6:   # take care of tiny point-clouds, i.e., padding
                max_dist = 1
            obj_pcd[:, :3] = obj_pcd[:, :3] / max_dist
            obj_pcds[i] = obj_pcd

        # convert to torch
        scene_pcds = torch.from_numpy(scene_pcds).float()
        obj_pcds = torch.from_numpy(np.stack(obj_pcds, 0)).float()
        obj_locs = torch.from_numpy(np.array(obj_locs)).float()
        if situation is not None:
            # include location and orientation
            pos = np.array(situation[0])
            ori = np.array(situation[1])
            if rot_matrix is None:
                pos_new = pos
                ori_new = ori
            else:
                pos_new = pos.reshape(1, 3) @ rot_matrix.transpose()
                pos_new = pos_new.reshape(-1)
                ori_new = R.from_quat(ori).as_matrix()
                ori_new = rot_matrix @ ori_new
                ori_new = R.from_matrix(ori_new).as_quat()
                ori_new = ori_new.reshape(-1)
            ori_new = self.quat_to_xy_orient(ori_new)
            extra_output = (
                torch.from_numpy(pos_new).float(), torch.from_numpy(ori_new).float()
            )
        elif extra_output is not None:
            extra_output = torch.from_numpy(extra_output).float()

        return scene_pcds, obj_pcds, obj_locs, extra_output
    
    def preprocess_obj_pcd(self, obj_pcds, tgt_obj_idx=-100, situation=None, rot_aug=True):
        extra_output = None   # anchor_loc for object refer, pos and ori for SQA
        # rotate scene
        rot_matrix = build_rotate_mat(self.split, rot_aug=rot_aug)

        # normalize pc and calculate location
        obj_locs = []
        for i, obj_pcd in enumerate(obj_pcds):
            if rot_matrix is not None:
                obj_pcd[:, :3] = np.matmul(obj_pcd[:, :3], rot_matrix.transpose())

            obj_center = obj_pcd[:, :3].mean(0)
            obj_bounds_min = obj_pcd[:, :3].min(0)
            obj_bounds_max = obj_pcd[:, :3].max(0)
            obj_size = obj_bounds_max - obj_bounds_min
            obj_locs.append(np.concatenate([obj_center, obj_size], 0))
            if i == tgt_obj_idx:
                # Select a loc within the obj bbox as the anchor.
                extra_output = obj_pcd[:, :3].min(0) + np.random.rand(3) * obj_size

            # subsample
            pcd_idxs = np.random.choice(len(obj_pcd), size=self.obj_num_points,
                                        replace=len(obj_pcd) < self.obj_num_points)
            obj_pcd = obj_pcd[pcd_idxs]

            # normalize
            obj_pcd[:, :3] = obj_pcd[:, :3] - obj_center
            max_dist = np.sqrt((obj_pcd[:, :3]**2).sum(1)).max()
            if max_dist < 1e-6:   # take care of tiny point-clouds, i.e., padding
                max_dist = 1
            obj_pcd[:, :3] = obj_pcd[:, :3] / max_dist
            obj_pcds[i] = obj_pcd

        # convert to torch
        obj_pcds = torch.from_numpy(np.stack(obj_pcds, 0)).float()
        obj_locs = torch.from_numpy(np.array(obj_locs)).float()
        if situation is not None:
            # include location and orientation
            pos = np.array(situation[0])
            ori = np.array(situation[1])
            if rot_matrix is None:
                pos_new = pos
                ori_new = ori
            else:
                pos_new = pos.reshape(1, 3) @ rot_matrix.transpose()
                pos_new = pos_new.reshape(-1)
                ori_new = R.from_quat(ori).as_matrix()
                ori_new = rot_matrix @ ori_new
                ori_new = R.from_matrix(ori_new).as_quat()
                ori_new = ori_new.reshape(-1)
            ori_new = self.quat_to_xy_orient(ori_new)
            extra_output = (
                torch.from_numpy(pos_new).float(), torch.from_numpy(ori_new).float()
            )
        elif extra_output is not None:
            extra_output = torch.from_numpy(extra_output).float()

        return obj_pcds, obj_locs, extra_output

    def preprocess_vox(self, scene_voxs, tgt_obj_pcds=[], situation=None, rot_aug=True):
        tgt_obj_mask = np.zeros(scene_voxs.shape[0]).astype(bool)

        for obj_pcd in tgt_obj_pcds:
            obj_bounds_min = obj_pcd[:, :3].min(0)
            obj_bounds_max = obj_pcd[:, :3].max(0)
            # highlight area
            this_obj_mask = (obj_bounds_min - self.voxel_size <= scene_voxs[:, :3]) \
                            & (scene_voxs[:, :3] <= obj_bounds_max + self.voxel_size)
            this_obj_mask = np.all(this_obj_mask, axis=1)
            tgt_obj_mask = tgt_obj_mask | this_obj_mask

        # rotate scene
        rot_matrix = build_rotate_mat(self.split, rot_aug=rot_aug)
        if rot_matrix is not None:
            scene_voxs[:, :3] = np.matmul(scene_voxs[:, :3], rot_matrix.transpose())

        # convert to torch
        scene_voxs = torch.from_numpy(scene_voxs).float()
        tgt_obj_mask = torch.from_numpy(tgt_obj_mask).bool()
        if situation is not None:
            # include location and orientation
            pos = np.array(situation[0])
            ori = np.array(situation[1])
            if rot_matrix is None:
                pos_new = pos
                ori_new = ori
            else:
                pos_new = pos.reshape(1, 3) @ rot_matrix.transpose()
                pos_new = pos_new.reshape(-1)
                ori_new = R.from_quat(ori).as_matrix()
                ori_new = rot_matrix @ ori_new
                ori_new = R.from_matrix(ori_new).as_quat()
                ori_new = ori_new.reshape(-1)
            ori_new = self.quat_to_xy_orient(ori_new)
            situation = (
                torch.from_numpy(pos_new).float(), torch.from_numpy(ori_new).float()
            )

        return scene_voxs, tgt_obj_mask, situation

    def _split_sentence(self, sentence, max_length, prefix=''):
        # only split during training
        if self.split == 'train' and len(prefix + sentence) > max_length:
            all_caps = []
            sents = sentence.split('. ')
            tmp = prefix
            for i in range(len(sents)):
                if len(tmp + sents[i] + '. ') > max_length:
                    all_caps.append(tmp)
                    tmp = prefix
                tmp += sents[i] + '. '

            all_caps.append(tmp)   # last chunk

            # final check
            ret = []
            for cap in all_caps:
                if len(cap) <= max_length:
                    ret.append(cap)
            return ret
        else:
            return [prefix + sentence]

    @staticmethod
    def quat_to_xy_orient(quat):
        # assume the orientation parallel with xy plane
        angle = R.from_quat(quat).as_euler('xyz')[-1]
        x = -np.cos(angle)
        y = -np.sin(angle)
        # SQA3D is annotated with the opposite direction
        return np.array([x, y])


@DATASET_REGISTRY.register()
class QACOTBase(LeoBase):
    def __init__(self, cfg, split, dataset):
        super().__init__(cfg)

        self.scan_data_loader = ScanDataLoader(cfg, dataset = dataset)
        logger.info(f"Loading {self.__class__.__name__} {split}-set annotations")
        self.meta_data = self.load_anno(cfg.data.cotqa.anno_dir)
        logger.info(f"Finish loading {self.__class__.__name__} {split}-set qa annotations, collected {len(self)} data\n")
        # self.meta_data_thought = self.load_thought_anno(cfg.data.qa.anno_dir)
        logger.info(f"Finish loading {self.__class__.__name__} {split}-set thought annotations, collected {len(self)} data\n")
        self.cfg = cfg

        if self.requires_pq3d_input:
            self.pq3d_tokenizer = AutoTokenizer.from_pretrained(cfg.data.pq3d_tokenizer_path)

    def load_anno(self, anno_dir):
        raise NotImplementedError()    
    
    ### load with global cache dict ###
    def prepare_data_loading_with_cache(self, dataset_name, scan_id, data_type_list = []):
        global scan_cache_data
        if dataset_name not in scan_cache_data:
            scan_cache_data[dataset_name] = {}

        if scan_id not in scan_cache_data[dataset_name]:
            scan_cache_data[dataset_name][scan_id] = {}

        data_type_to_process = []
        for data_type in data_type_list:
            if data_type not in scan_cache_data[dataset_name][scan_id]:
                data_type_to_process.append(data_type)
        if len(data_type_to_process) > 0:
            one_scan = self.scan_data_loader.get_data(dataset_name, scan_id, data_type = data_type_to_process)
            scan_cache_data[dataset_name][scan_id].update(one_scan)

        return scan_cache_data[dataset_name][scan_id]

    def __len__(self):
        return len(self.meta_data)

    def get_thought(self):
        thought = []
        for item in self.meta_data:
            thought.append(item['thought'])
        return thought
    
    def __getitem__(self, index):
        meta_data = self.meta_data[index]
        scene_id = meta_data['scene_id']
        situation = meta_data['situation']
        question_index = meta_data['index'] if 'index' in meta_data else -1
        if 'pos_ori' in meta_data:
            pos_ori = meta_data['pos_ori']
        else:
            pos_ori = None
        agent_pos_ori = meta_data['agent_pos_ori']
        agent_pos_ori = torch.tensor(agent_pos_ori) 
        obj_ids = [int(i) for i in meta_data['obj_ids']]
        question = meta_data['question']
        answers = meta_data['answers']
        if 'answer_with_cot' in meta_data:
            answer_with_cot_raw = meta_data['answer_with_cot']
            meta_data['answer_with_cot'] = replace_cot_indicators_with_tokens(meta_data['answer_with_cot'])
            answers = [meta_data['answer_with_cot']]
        if hasattr(self.cfg, 'grounding') and self.cfg.grounding.use_region_mask:
            if 'answer_with_cot_region' in meta_data:
                meta_data['answer_with_cot_region'] = replace_cot_indicators_with_tokens(meta_data['answer_with_cot_region'])
                answers = [meta_data['answer_with_cot_region']]
        if 'obj_prob' in meta_data:
            oracle_obj_prob = meta_data['obj_prob']
        else:
            oracle_obj_prob = ''
        if 'obj_loc' in meta_data:
            oracle_obj_loc = meta_data['obj_loc']
        else:
            oracle_obj_loc = ''
        question_type = meta_data['type'] if 'type' in meta_data else ''
        # oracle_thought = meta_data['obj_COT'] if 'obj_COT' in meta_data else ''
        oracle_thought = ''
        # prob_types = ['counting', 'existence', 'refer', 'affordance', 'room type']
        # location_types = ['spatial relationship', 'navigation']
        image_types = ['description', 'attribute']

        # prepare mv_info
        # scan_data = self.prepare_data_loading_with_cache(dataset_name = 'ScanNet', scan_id = scene_id, data_type_list = ['obj_pcds', 'mv_info'])
        img_list = []
        # HACK
        # print(obj_ids)
        # print(scan_data['mv_info'].keys())
        # print(self.cfg.dataset_wrapper_args.max_thought_img_num)
        # if question_type in image_types:
        #     for obj_id in obj_ids[:self.cfg.dataset_wrapper_args.max_thought_img_num]:
        #         if obj_id in scan_data['mv_info'].keys():
        #             one_bbox = random.choice(scan_data['mv_info'][int(obj_id)])
        #             img = self.scan_data_loader.get_one_img(one_bbox)    # HACK: 224*224
        #             img_list.append(img.unsqueeze(0))
            # HACK: insert dummy images if len(img_list) == 0

        if len(img_list) == 0:
            img_list.append(torch.zeros(1, 3, 224, 224))
        thought_pixel_values = torch.cat(img_list, dim = 0)
        # print(f'in datasets.py, shape of thought_pixel_values: {thought_pixel_values.shape}')

        # load pcds
        scene_data = self.load_scene(self.domain, scene_id)
        # scene_pcds = scene_data['scene_pcds']   # np.ndarray (N, 6)
        # scene_inst_ids = scene_data['scene_inst_ids']   # np.ndarray (N,)
        # scene_3d_fts = scene_data['scene_3d_fts']   # torch.Tensor (N, scene_3d_feat_dim)
        # scene_2d_fts = scene_data['scene_2d_fts']   # torch.Tensor (N, scene_2d_feat_dim)
        obj_pcds = scene_data['obj_pcds_gt']   # dict {int: np.ndarray (n, 6)}
        if self.pc_type == 'gt':
            obj_fts_img_gt = scene_data['obj_fts_img_gt']   # dict {int: np.ndarray (obj_img_feat_dim)}
            obj_fts_vox_gt = scene_data['obj_fts_vox_gt']   # dict {int: np.ndarray (obj_vox_feat_dim)}
            obj_pcds_gt = scene_data['obj_pcds_gt']   # dict {int: np.ndarray (n, 6)}
            obj_locs_gt = scene_data['obj_locs_gt']   # dict {int: np.ndarray (3,)}
            gt_to_pred_id = scene_data['gt_to_pred_id']
            obj_label_names_pred = scene_data['obj_label_names_pred']
        elif self.pc_type == 'pred':
            obj_fts_img_pred = scene_data['obj_fts_img_pred']
            obj_fts_vox_pred = scene_data['obj_fts_vox_pred']
            obj_pcds_pred = scene_data['obj_pcds_pred']
            obj_label_names_pred = scene_data['obj_label_names_pred']
            obj_locs_pred = scene_data['obj_locs_pred']
            gt_to_pred_id = scene_data['gt_to_pred_id']
        # scene_voxs = scene_data['scene_voxs']
        # vox_3d_fts = scene_data['vox_3d_fts'].float()
        # vox_2d_fts = scene_data['vox_2d_fts']
        # if vox_2d_fts is not None:
        #     vox_2d_fts = vox_2d_fts.float()
        #     vox_2d_fts_mask = 1
        # else:
        #     vox_2d_fts = torch.zeros(vox_3d_fts.shape[0], self.scene_2d_feat_dim)
        #     vox_2d_fts_mask = 0

        # grounding only
        if hasattr(self.cfg, 'grounding') and self.cfg.grounding.grd_only:
            answers = [f'Thought: {GRD_TOKEN_TXT}']
        
        if hasattr(self.cfg.cot, 'disable_cot') and self.cfg.cot.disable_cot:
            answers = meta_data['answers']

        if self.split == 'train':
            # random.shuffle(obj_idx_list)
            answers = random.choice(answers)
            if not isinstance(situation, str):
                situation = random.choice(situation)
        else:
            if not isinstance(situation, str):
                situation = situation[0]

        if self.__class__.__name__ == 'QACOTScanNetSQA3D':
            question_type = get_sqa_question_type(question)
        else:
            question_type = "default"

        # scene_voxs, tgt_obj_mask, pos_ori = self.preprocess_vox(
        #     scene_voxs=scene_voxs, tgt_obj_pcds=[], situation=pos_ori
        # )

        if question_type not in image_types:
            question_with_oracle_thought = f"{question} Thought: {oracle_thought}" if oracle_thought != '' else question
        # print(f'question_with_oracle_thought: {question_with_oracle_thought}')
        # data_dict = self.get_prompts(instruction=question_with_oracle_thought, situation=situation)
        if hasattr(self, 'use_oracle_thought') and self.use_oracle_thought:
            oracle_thought = replace_cot_indicators_with_tokens(self.remove_answer_tag_content(answer_with_cot_raw))
            data_dict = self.get_prompts(instruction=question, situation=situation, oracle_thought=oracle_thought)
            print(f" !!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f" oracle_thought: {oracle_thought}")
        else:
            data_dict = self.get_prompts(instruction=question, situation=situation)


        # object features and locations
        if 'obj_2d_fts' in scene_data:
            obj_2d_fts = scene_data['obj_2d_fts']
            obj_2d_fts = torch.tensor(obj_2d_fts).float()
            obj_locs = torch.tensor(scene_data['obj_locs_gt']).float()
            inst2_labels = scene_data['obj_labels']

        # build grounding_obj_mask_gt
        obj_pcds = scene_data['obj_pcds_gt'].copy()
        remained_obj_ids = [i for i in obj_pcds.keys() if i not in obj_ids]
        if hasattr(self.cfg, 'grounding') and hasattr(self.cfg.grounding, 'use_region_mask') and self.cfg.grounding.use_region_mask:
            if 'question_region_obj_ids' in meta_data:
                question_region_obj_ids = meta_data['question_region_obj_ids']
                remained_obj_ids = [i for i in remained_obj_ids if i in question_region_obj_ids]
        selected_obj_2d_fts = []
        selected_obj_ids = []
        selected_locs = []
        selected_gt_labels = []
        selected_pred_labels = []
        selected_obj_pcds = []
        selected_images = []
        
        # hardcode: gt img
        img_obj_id = obj_ids[0] if len(obj_ids) > 0 else -1   # only choose one image, fill in a dummy value if no obj_ids
        # if int(img_obj_id) in scan_data['mv_info']:
            # one_bbox = scan_data['mv_info'][int(img_obj_id)][0]  # default: choose the first image
            # gt_img = self.scan_data_loader.get_one_full_img(one_bbox=one_bbox, transform=self.img_transform)   
        # else:
        #     gt_img = torch.zeros(3, 336, 336)
        #     print(f'Warning: obj_id {img_obj_id} not in mv_info!')
        gt_img = self.get_one_full_img(scan_id=scene_id, inst_id=img_obj_id, transform=self.img_transform) 
        if gt_img is None:
            gt_img = torch.zeros(3, 336, 336)
            # print(f'Warning: obj_id {img_obj_id} not in mv_info!')

        # pq3d feature
        selected_obj_fts_img = []
        selected_obj_fts_vox = []
        if self.split == 'train':
            random.shuffle(remained_obj_ids)

        # print(f'obj ids in obj_pcds: {obj_pcds.keys()}')
        # print(f'obj ids in mv info: {scan_data["mv_info"].keys()}')
        
        if self.pc_type == 'gt':
            for i in remained_obj_ids:
                selected_obj_2d_fts.append(obj_2d_fts[i])
                selected_obj_ids.append(i)
                selected_locs.append(obj_locs_gt[i])
                if hasattr(self, 'label_type') and self.label_type == 'pred':
                    obj_id = i
                    if obj_id in gt_to_pred_id:
                        pred_id = gt_to_pred_id[obj_id][0]
                        selected_gt_labels.append(obj_label_names_pred[pred_id])
                    else:
                        selected_gt_labels.append(inst2_labels[obj_id])
                else:
                    selected_gt_labels.append(inst2_labels[i])
                selected_obj_fts_img.append(obj_fts_img_gt[i])
                selected_obj_fts_vox.append(obj_fts_vox_gt[i])
                selected_obj_pcds.append(obj_pcds_gt[i])
                # HACK: if there is no image for the obj id, build a dummy image
                # if int(i) not in scan_data['mv_info']:
                #     # print(f'Warning: obj_id {i} not in mv_info!')
                #     img = torch.zeros(3, 336, 336)
                #     selected_images.append(img.unsqueeze(0))
                #     continue
                # one_bbox = scan_data['mv_info'][int(i)][0]  # default: choose the first image
                # img = self.scan_data_loader.get_one_full_img(one_bbox, transform=self.img_transform)    # hardcode: 224*224
                img = self.get_one_full_img(scan_id=scene_id, inst_id=i, transform=self.img_transform)
                if img is None:
                    # print(f'Warning: obj_id {i} not in mv_info!')
                    img = torch.zeros(3, 336, 336)
                selected_images.append(img.unsqueeze(0))

            valid_obj_ids = [i for i in obj_ids if i in obj_pcds_gt]

            for j, obj_id in enumerate(valid_obj_ids):
                random_index = random.randint(0, len(selected_obj_ids))
                selected_obj_2d_fts.insert(random_index, obj_2d_fts[obj_id])
                selected_obj_ids.insert(random_index, obj_id)
                selected_locs.insert(random_index, obj_locs_gt[obj_id])
                if hasattr(self, 'label_type') and self.label_type == 'pred':
                    if obj_id in gt_to_pred_id:
                        pred_id = gt_to_pred_id[obj_id][0]
                        selected_gt_labels.append(obj_label_names_pred[pred_id])
                    else:
                        selected_gt_labels.append(inst2_labels[obj_id])
                else:
                    selected_gt_labels.insert(random_index, inst2_labels[obj_id])
                selected_obj_fts_img.insert(random_index, obj_fts_img_gt[obj_id])
                selected_obj_fts_vox.insert(random_index, obj_fts_vox_gt[obj_id])
                selected_obj_pcds.insert(random_index, obj_pcds_gt[obj_id])
                # if int(obj_id) not in scan_data['mv_info']:
                #     # print(f'Warning: obj_id {i} not in mv_info!')
                #     img = torch.zeros(3, 336, 336)
                #     selected_images.insert(random_index, img.unsqueeze(0))
                #     continue
                # one_bbox = scan_data['mv_info'][int(obj_id)][0]
                # img = self.scan_data_loader.get_one_full_img(one_bbox, transform=self.img_transform)
                img = self.get_one_full_img(scan_id=scene_id, inst_id=obj_id, transform=self.img_transform)
                if img is None:
                    # print(f'Warning: obj_id {obj_id} not in mv_info!')
                    img = torch.zeros(3, 336, 336)
                selected_images.insert(random_index, img.unsqueeze(0))

            if len(selected_obj_pcds) == 0:
                dummy_ref_key = next(iter(obj_pcds))

                # dummy point cloud (NumPy) – required by preprocess_obj_pcd
                dummy_pcd = np.zeros_like(obj_pcds[dummy_ref_key])
                selected_obj_pcds.append(dummy_pcd)

                # refer to actual tensors for dtype and device safety
                selected_obj_2d_fts.append(torch.zeros_like(obj_2d_fts[dummy_ref_key]))
                selected_obj_ids.append(-1)
                selected_locs.append(torch.zeros_like(obj_locs[dummy_ref_key]))
                selected_gt_labels.append("Empty")
                selected_obj_fts_img.append(torch.zeros_like(obj_fts_img_gt[dummy_ref_key]))
                selected_obj_fts_vox.append(torch.zeros_like(obj_fts_vox_gt[dummy_ref_key]))
                selected_images.append(torch.zeros(1, 3, 336, 336, device=selected_obj_fts_img[-1].device))

            assert len(selected_obj_2d_fts) == len(selected_obj_ids) == len(selected_locs) == len(selected_gt_labels)
            obj_labels = selected_gt_labels

            obj_fts, obj_locs, _ = self.preprocess_obj_pcd(selected_obj_pcds)

            obj_2d_fts = torch.stack(selected_obj_2d_fts, dim=0)
            obj_locs = torch.stack(selected_locs, dim=0)

            obj_fts_raw, obj_locs_raw, _ = self.preprocess_obj_pcd(selected_obj_pcds, rot_aug=False)

            # print(f'!!!!!!!!!!check obj_locs_raw!!!!!!!!!!!!')
            # print(obj_locs_raw)
            # print(f'!!!!!!!!!!check obj_locs!!!!!!!!!!!!')
            # print(obj_locs)

            grounding_obj_mask = torch.zeros(len(selected_obj_ids))
            grounding_obj_mask[[selected_obj_ids.index(i) for i in valid_obj_ids]] = 1
            grounding_obj_mask = grounding_obj_mask.float()
        else:
            valid_obj_ids_pred = []
            remaind_obj_ids_pred = []
            # print(f'all items of gt_to_pred_id: {gt_to_pred_id.items()}')
            for obj_id in obj_ids:
                if obj_id in gt_to_pred_id:
                    valid_obj_ids_pred.append(gt_to_pred_id[obj_id][0])   # to mimic in-the-wild scenario, we choose all the IOU > 0 predicted masks in the sub region.
                # else:
                #     print(f'Inference: obj_id: {obj_id} not in gt_to_pred_id!')

            pred_to_gt_id = {}
            for obj_id in gt_to_pred_id:
                pred_to_gt_id[gt_to_pred_id[obj_id]] = obj_id  
        
            for obj_id in remained_obj_ids:   # include: 
                if obj_id in gt_to_pred_id:
                    remaind_obj_ids_pred.append(gt_to_pred_id[obj_id][0])
            
            for obj_id_pred in remaind_obj_ids_pred:
                selected_obj_ids.append(obj_id_pred)
                selected_locs.append(obj_locs_pred[obj_id_pred])
                selected_pred_labels.append(obj_label_names_pred[obj_id_pred])
                selected_obj_fts_img.append(obj_fts_img_pred[obj_id_pred])
                selected_obj_fts_vox.append(obj_fts_vox_pred[obj_id_pred])
                selected_obj_pcds.append(obj_pcds_pred[obj_id_pred])
                # if obj_id_pred not in pred_to_gt_id or int(pred_to_gt_id[obj_id_pred]) not in scan_data['mv_info']:
                #     # print(f'Warning: obj_id {i} not in mv_info!')
                #     img = torch.zeros(3, 336, 336)
                #     selected_images.append(img.unsqueeze(0))
                #     continue
                # one_bbox = scan_data['mv_info'][int(pred_to_gt_id[obj_id_pred])][0]  # for image retrieval: choose the first image
                # img = self.scan_data_loader.get_one_full_img(one_bbox, transform=self.img_transform)    # hardcode: 224*224
                if obj_id_pred not in pred_to_gt_id:
                    # print(f'Inference: obj_id {obj_id_pred} not in pred_to_gt_id!')
                    img = torch.zeros(3, 336, 336)
                    selected_images.append(img.unsqueeze(0))
                else:
                    img = self.get_one_full_img(scan_id=scene_id, inst_id=pred_to_gt_id[obj_id_pred], transform=self.img_transform)
                    if img is None:
                        # print(f'Warning: obj_id {pred_to_gt_id[obj_id_pred]} not in mv_info!')
                        img = torch.zeros(3, 336, 336)
                    selected_images.append(img.unsqueeze(0))
            
            for j, obj_id_pred in enumerate(valid_obj_ids_pred):  
                random_index = random.randint(0, len(selected_obj_ids))
                selected_obj_ids.insert(random_index, obj_id_pred)
                selected_locs.insert(random_index, obj_locs_pred[obj_id_pred])
                selected_pred_labels.insert(random_index, obj_label_names_pred[obj_id_pred])
                selected_obj_fts_img.insert(random_index, obj_fts_img_pred[obj_id_pred])
                selected_obj_fts_vox.insert(random_index, obj_fts_vox_pred[obj_id_pred])
                selected_obj_pcds.insert(random_index, obj_pcds_pred[obj_id_pred])
                # if obj_id_pred not in pred_to_gt_id or int(pred_to_gt_id[obj_id_pred]) not in scan_data['mv_info']:
                #     # print(f'Warning: obj_id {i} not in mv_info!')
                #     img = torch.zeros(3, 336, 336)
                #     selected_images.insert(random_index, img.unsqueeze(0))
                #     continue
                # one_bbox = scan_data['mv_info'][int(pred_to_gt_id[obj_id_pred])][0]
                # img = self.scan_data_loader.get_one_full_img(one_bbox, transform=self.img_transform)
                if obj_id_pred not in pred_to_gt_id:
                    # print(f'Inference: obj_id {obj_id_pred} not in pred_to_gt_id!')
                    img = torch.zeros(3, 336, 336)
                    selected_images.insert(random_index, img.unsqueeze(0))
                else:
                    img = self.get_one_full_img(scan_id=scene_id, inst_id=pred_to_gt_id[obj_id_pred], transform=self.img_transform)
                    if img is None:
                        # print(f'Warning: obj_id {pred_to_gt_id[obj_id_pred]} not in mv_info!')
                        img = torch.zeros(3, 336, 336)
                    selected_images.insert(random_index, img.unsqueeze(0))

            if len(selected_obj_pcds) == 0:
                dummy_ref_key = next(iter(obj_pcds))

                # dummy point cloud (NumPy) – required by preprocess_obj_pcd
                dummy_pcd = np.zeros_like(obj_pcds[dummy_ref_key])
                selected_obj_pcds.append(dummy_pcd)

                # refer to actual tensors for dtype and device safety
                selected_obj_2d_fts.append(torch.zeros_like(obj_2d_fts[dummy_ref_key]))
                selected_obj_ids.append(-1)
                selected_locs.append(torch.zeros_like(obj_locs[dummy_ref_key]))
                selected_pred_labels.append("Empty")
                selected_obj_fts_img.append(torch.zeros_like(obj_fts_img_pred[dummy_ref_key]))
                selected_obj_fts_vox.append(torch.zeros_like(obj_fts_vox_pred[dummy_ref_key]))
                selected_images.append(torch.zeros(1, 3, 336, 336, device=selected_obj_fts_img[-1].device))

            assert len(selected_obj_ids) == len(selected_locs) == len(selected_pred_labels)
            obj_labels = selected_pred_labels

            obj_fts, obj_locs, _ = self.preprocess_obj_pcd(selected_obj_pcds)

            obj_locs = torch.stack(selected_locs, dim=0)
            grounding_obj_mask = torch.zeros(len(selected_obj_ids))
            grounding_obj_mask[[selected_obj_ids.index(i) for i in valid_obj_ids_pred]] = 1
            grounding_obj_mask = grounding_obj_mask.float()

        obj_ids = torch.tensor(obj_ids).long()
        data_dict.update({
            'source': self.__class__.__name__,
            'scene_id': scene_id,
            # 'scene_pcds': scene_pcds,
            # 'scene_3d_fts': scene_3d_fts,
            # 'scene_2d_fts': scene_2d_fts,
            # 'scene_voxs': scene_voxs,
            # 'vox_3d_fts': vox_3d_fts,
            # 'vox_2d_fts': vox_2d_fts,
            # 'vox_2d_fts_mask': vox_2d_fts_mask,
            # 'obj_2d_fts': obj_2d_fts,
            'grounding_obj_mask_gt': grounding_obj_mask,
            'agent_pos_ori': agent_pos_ori,
            # 'obj_pcds': obj_pcds,
            # 'obj_locs': obj_locs,
            'obj_fts_img': torch.stack(selected_obj_fts_img),
            'obj_fts_vox': torch.stack(selected_obj_fts_vox),
            'obj_fts': obj_fts,
            # 'tgt_obj_mask': tgt_obj_mask,
            'obj_ids': obj_ids,
            'gt_img': gt_img,
            'output_gt': answers,
            'obj_locs': obj_locs,
            'gt_labels': obj_labels,
            'oracle_obj_prob': oracle_obj_prob,
            'oracle_obj_loc': oracle_obj_loc,
            'thought_pixel_values': thought_pixel_values,
            'all_obj_imgs': torch.cat(selected_images, dim=0),
            'index': question_index,
        })
        if pos_ori is not None:
            data_dict['pos'] = pos_ori[0]
            data_dict['ori'] = pos_ori[1]
        if 'sqa_type' in meta_data:
            data_dict['sqa_type'] = meta_data['sqa_type']
        else:
            data_dict['sqa_type'] = question_type
        if 'type' in meta_data:
            data_dict['type'] = meta_data['type']
        # print(f'type of data_dict: {data_dict["type"]}')
        if self.requires_pq3d_input:
            data_dict['prompt'] = self.pq3d_tokenizer(
                [f"{situation} {question}"], return_tensors='pt',
                add_special_tokens=True, truncation=True,
            ).input_ids[0].float()
            data_dict['prompt_pad_masks'] = torch.ones((len(data_dict['prompt']))).bool()
            data_dict['prompt_type'] = PromptType.TXT

        return self.check_output_and_fill_dummy(data_dict)

@DATASET_REGISTRY.register()
class QACOTScanNetMSR3D(QACOTBase):
    def __init__(self, cfg, split, scene_data_source=None):
        self.domain = 'ScanNet'
        self.scene_data_source = scene_data_source
        self.scan_data_loader = ScanDataLoader(cfg, dataset = 'ScanNet')
        super(QACOTBase, self).__init__(cfg)
        logger.info(f"Loading {self.__class__.__name__} {split}-set annotations")
        self.cfg = cfg
        self.split = split
        if split == 'train':
            if hasattr(cfg.data.cotqa, "use_pred_for_train"):
                self.pc_type = 'pred' if cfg.data.cotqa.use_pred_for_train else 'gt'
        else:
            self.pc_type = getattr(cfg.data.cotqa.msr3d, 'pc_type', 'gt')
            self.label_type = getattr(cfg.data.cotqa.msr3d, 'label_type', 'gt')
            self.use_oracle_thought = getattr(cfg.data.cotqa.msr3d, 'use_oracle_thought', False)

        logger.info(f"<<<<<<<<<In QACOTScanNetMSR3D>>>>>>>>")
        logger.info(f"Loading {self.__class__.__name__} {split}-set annotations")
        self.meta_data = self.load_anno(cfg.data.cotqa.msr3d.anno_dir)

        logger.info(f"pc type: {self.pc_type}")

        if split == 'val':
            self.meta_data = self.meta_data[:cfg.cot.msr3d_val_set_size]

        if split == 'val':
            self.meta_data = self.meta_data[:cfg.cot.msr3d_val_set_size]

        if 'debug_size' in cfg.task.leomix.LeoMix:
            self.meta_data = self.meta_data[:cfg.task.leomix.LeoMix.debug_size]
        logger.info(f"Finish loading {self.__class__.__name__} {split}-set annotations, collected {len(self)} data\n")

        if self.requires_pq3d_input:
            self.pq3d_tokenizer = AutoTokenizer.from_pretrained(cfg.data.pq3d_tokenizer_path)

    def load_anno(self, anno_dir):
        meta_data = []
        with open(os.path.join(anno_dir, f'situated_qa_{self.split}_pure_txt.json')) as f:
            json_data = json.load(f)
        for item in json_data:
            answers = []
            for a in item['answers']:
                a = a.strip()
                # a = a[0].upper() + a[1:]
                a = convert_person_view_2_to_1(a)
                answers.append(a)
            info_dict = {
                'scene_id': item['scene_id'],
                'situation': convert_person_view_1_to_2(item['situation']),
                'pos_ori': [item['position'], item['rotation']],
                'agent_pos_ori': [item['position'], [math.cos(item['orientation_angle']), math.sin(item['orientation_angle']), 0]],
                'obj_ids': item['obj_ids'],
                'labels': item['labels'],
                'question': convert_person_view_1_to_2(item['question']),
                'answers': answers,
                'type': item['type'],   
            }
            if 'obj_COT' in item:
                info_dict['obj_COT'] = item['obj_COT']
            if 'answer_with_cot' in item:
                info_dict['answer_with_cot'] = item['answer_with_cot']
            if 'obj_prob' in item:
                info_dict['obj_prob'] = item['obj_prob']
            if 'question_region_obj_ids' in item:
                info_dict['question_region_obj_ids'] = item['question_region_obj_ids']
            if 'answer_with_cot_region' in item:
                info_dict['answer_with_cot_region'] = item['answer_with_cot_region']
            if 'index' in item:
                info_dict['index'] = item['index']
            
            meta_data.append(info_dict)

        return meta_data

@DATASET_REGISTRY.register()
class QACOTScanNetSQA3D(QACOTBase):
    def __init__(self, cfg, split, scene_data_source=None):
        self.domain = 'ScanNet'
        self.scene_data_source = scene_data_source
        self.scan_data_loader = ScanDataLoader(cfg, dataset = 'ScanNet')
        super(QACOTBase, self).__init__(cfg)
        logger.info(f"Loading {self.__class__.__name__} {split}-set annotations")
        self.cfg = cfg
        self.split = split
        if split == 'train':
            if hasattr(cfg.data.cotqa, "use_pred_for_train"):
                self.pc_type = 'pred' if cfg.data.cotqa.use_pred_for_train else 'gt'
        else:
            self.pc_type = getattr(cfg.data.cotqa.sqa3d, 'pc_type', 'gt')
            self.label_type = getattr(cfg.data.cotqa.sqa3d, 'label_type', 'gt')
            self.use_oracle_thought = getattr(cfg.data.cotqa.sqa3d, 'use_oracle_thought', False)

        logger.info(f"<<<<<<<<<In QACOTScanNetSQA3D>>>>>>>>")
        logger.info(f"Loading {self.__class__.__name__} {split}-set annotations")
        self.meta_data = self.load_anno(cfg.data.cotqa.sqa3d.anno_dir)

        logger.info(f"Keys of meta_data: {self.meta_data[0].keys()}")

        logger.info(f"pc type: {self.pc_type}")
        if cfg.debug and 'debug_size' in cfg.task.leomix.LeoMix:
            self.meta_data = self.meta_data[:cfg.task.leomix.LeoMix.debug_size]

        if split == 'val':
            self.meta_data = self.meta_data[:cfg.cot.sqa3d_val_set_size]

        logger.info(f"Finish loading {self.__class__.__name__} {split}-set annotations, collected {len(self)} data\n")

        if self.requires_pq3d_input:
            self.pq3d_tokenizer = AutoTokenizer.from_pretrained(cfg.data.pq3d_tokenizer_path)

    def load_anno(self, anno_dir):
        meta_data = []  
        with open(os.path.join(anno_dir, f'sqa_cot_{self.split}.json')) as f:
            json_data = json.load(f)


        for item in json_data:
            answers = []
            for a in item['answers']:
                a = a.strip()
                # a = a[0].upper() + a[1:]
                answers.append(a)
            sqa_type = get_sqa_question_type(item['question'])
            info_dict = {
                'scene_id': item['scene_id'],
                'situation': item['situation'],
                # 'pos_ori': [item['position'], item['rotation']],
                'agent_pos_ori': [item['position'], [math.cos(item['orientation_angle']), math.sin(item['orientation_angle']), 0]],
                'obj_ids': item['obj_ids'],
                'labels': item['labels'],
                'question': item['question'],
                'answers': answers,
                'type': item['type'],   
                'sqa_type': sqa_type,
                'index': item['question_id']
            }
            if 'obj_COT' in item:
                info_dict['obj_COT'] = item['obj_COT']
            if 'answer_with_cot' in item:
                info_dict['answer_with_cot'] = item['answer_with_cot']
            if 'obj_prob' in item:
                info_dict['obj_prob'] = item['obj_prob']
            if 'question_region_obj_ids' in item:
                info_dict['question_region_obj_ids'] = item['question_region_obj_ids']
            if 'answer_with_cot_region' in item:
                info_dict['answer_with_cot_region'] = item['answer_with_cot_region']

            if self.split == 'train':
                if 'answer_with_cot' not in item:
                    print(f"?????????????????????????????????????????")
                    print(f"No answer_with_cot is detected in the scene: {item['scene_id']}")

                if "[OBJ]" not in item['answer_with_cot']:
                    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print(f"No [OBJ] is detected in the scene: {item['scene_id']}")

            meta_data.append(info_dict)
        return meta_data

@DATASET_REGISTRY.register()
class QACOTScanNetGQA3D(QACOTBase):
    def __init__(self, cfg, split, scene_data_source=None):
        self.domain = 'ScanNet'
        self.scene_data_source = scene_data_source
        self.scan_data_loader = ScanDataLoader(cfg, dataset = 'ScanNet')
        super(QACOTBase, self).__init__(cfg)
        logger.info(f"Loading {self.__class__.__name__} {split}-set annotations")
        self.cfg = cfg
        self.split = split
        if split == 'train':
            if hasattr(cfg.data.cotqa, "use_pred_for_train"):
                self.pc_type = 'pred' if cfg.data.cotqa.use_pred_for_train else 'gt'
        else:
            self.pc_type = getattr(cfg.data.cotqa.gqa3d, 'pc_type', 'gt')
        
        logger.info(f"<<<<<<<<<In QACOTScanNetGQA3D>>>>>>>>")
        logger.info(f"Loading {self.__class__.__name__} {split}-set annotations")
        self.meta_data = self.load_anno(cfg.data.cotqa.gqa3d.anno_dir)

        logger.info(f"Keys of meta_data: {self.meta_data[0].keys()}")
        logger.info(f"pc type: {self.pc_type}")
        if cfg.debug and 'debug_size' in cfg.task.leomix.LeoMix:
            self.meta_data = self.meta_data[:cfg.task.leomix.LeoMix.debug_size]

        if split == 'val':
            self.meta_data = self.meta_data[:cfg.cot.gqa3d_val_set_size]
        
        logger.info(f"Finish loading {self.__class__.__name__} {split}-set annotations, collected {len(self)} data\n")

        if self.requires_pq3d_input:
            self.pq3d_tokenizer = AutoTokenizer.from_pretrained(cfg.data.pq3d_tokenizer_path)
        
    def load_anno(self, anno_dir):
        meta_data = []  
        with open(os.path.join(anno_dir, f'gqa3d_{self.split}.json')) as f:
            json_data = json.load(f)

        for item in json_data:
            answers = []
            for a in item['answers']:
                a = a.strip()
                # a = a[0].upper() + a[1:]
                answers.append(a)
            info_dict = {
                'scene_id': item['scene_id'],
                'situation': '',
                # 'pos_ori': [item['position'], item['rotation']],
                'agent_pos_ori': [item['position'], [math.cos(item['orientation_angle']), math.sin(item['orientation_angle']), 0]],
                'obj_ids': item['obj_ids'],
                'labels': item['labels'],
                'question': item['question'],
                'answers': answers,
                'type': item['type'],   
            }
            if 'obj_COT' in item:
                info_dict['obj_COT'] = item['obj_COT']
            if 'answer_with_cot' in item:
                info_dict['answer_with_cot'] = item['answer_with_cot']
            if 'obj_prob' in item:
                info_dict['obj_prob'] = item['obj_prob']
            if 'question_region_obj_ids' in item:
                info_dict['question_region_obj_ids'] = item['question_region_obj_ids']
            if 'answer_with_cot_region' in item:
                info_dict['answer_with_cot_region'] = item['answer_with_cot_region']

            meta_data.append(info_dict)
        return meta_data

@DATASET_REGISTRY.register()
class QACOTScanNetScanQA(QACOTBase):
    def __init__(self, cfg, split, scene_data_source=None):
        self.domain = 'ScanNet'
        self.scene_data_source = scene_data_source
        self.scan_data_loader = ScanDataLoader(cfg, dataset = 'ScanNet')
        super(QACOTBase, self).__init__(cfg)
        logger.info(f"Loading {self.__class__.__name__} {split}-set annotations")
        self.cfg = cfg
        if split == 'train':
            self.split = 'train'
            self.pc_type = 'gt'
        else:
            self.split = 'val'
            self.pc_type = getattr(cfg.data.qa.scanqa, 'pc_type', 'gt')

        logger.info(f"Loading {self.__class__.__name__} {split}-set annotations")
        self.meta_data = self.load_anno(cfg.data.qa.scanqa.anno_dir)

        if 'debug_size' in cfg.task.leomix.LeoMix:
            self.meta_data = self.meta_data[:cfg.task.leomix.LeoMix.debug_size]
            
        logger.info(f"Finish loading {self.__class__.__name__} {split}-set annotations, collected {len(self)} data\n")

        if self.requires_pq3d_input:
            self.pq3d_tokenizer = AutoTokenizer.from_pretrained(cfg.data.pq3d_tokenizer_path)

    def load_anno(self, anno_dir):
        meta_data = []
        anno_file = os.path.join(anno_dir, f'ScanQA_v1.0_{self.split}.json')
        with open(anno_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        for item in json_data:
            answers = []
            for a in item['answers']:
                a = a.strip()
                # a = a[0].upper() + a[1:]
                answers.append(a)
            meta_data.append({
                'scene_id': item['scene_id'],
                'situation': "",
                'pos_ori': None,
                'obj_ids': item['object_ids'],
                'question': item['question'],
                'answers': answers,   # list
            })

        return meta_data
    
@DATASET_REGISTRY.register()
class QACOTARKitScenesMSR3D(QACOTBase):
    def __init__(self, cfg, split, scene_data_source=None):
        self.domain = 'ARkit'
        self.scene_data_source = scene_data_source
        self.scan_data_loader = ScanDataLoader(cfg, dataset = 'ARKitScenes')
        super(QACOTBase, self).__init__(cfg)
        logger.info(f"Loading {self.__class__.__name__} {split}-set annotations")
        self.cfg = cfg
        if split == 'train':
            self.split = 'train'
            self.pc_type = 'gt'
        else:
            self.split = 'val'
            self.pc_type = getattr(cfg.data.qa.msr3d, 'pc_type', 'gt')

        logger.info(f"Loading {self.__class__.__name__} {split}-set annotations")
        self.meta_data = self.load_anno(cfg.data.cotqa.msr3d.anno_dir)

        if 'debug_size' in cfg.task.leomix.LeoMix:
            self.meta_data = self.meta_data[:cfg.task.leomix.LeoMix.debug_size]

@DATASET_REGISTRY.register()
class LeoMix(Dataset):
    mapping = {
        'qacot_scannet_msr3d': QACOTScanNetMSR3D,
        'qacot_scannet_sqa3d': QACOTScanNetSQA3D,
        'qacot_scannet_gqa3d': QACOTScanNetGQA3D,
        'qacot_scannet_scanqa': QACOTScanNetScanQA,
        'qacot_arkit_msr3d': QACOTARKitScenesMSR3D,
    }

    def __init__(self, cfg, split, scene_data_source=None):
        self.datasets = []
        mix_dict = cfg.task.leomix.LeoMix.mix
        ratio = cfg.task.leomix.LeoMix.ratio
        logger.info(f"LeoMix about to load:\n{OmegaConf.to_yaml(mix_dict, resolve=True)}")
        for task, v0 in mix_dict.items():
            for domain, v1 in v0.items():
                for source in v1:
                    dataset_key = f'{task}_{domain}_{source}'
                    self.datasets.append(self.mapping[dataset_key](cfg, split, scene_data_source))
        self.leomix_required_keys = self.datasets[0].leomix_required_keys

        if isinstance(ratio, (int, float)):
            self.index_range = list(np.cumsum([int(len(d)*ratio) for d in self.datasets]))
        else:
            self.index_range = list(np.cumsum([int(len(d)*ratio[i]) for i, d in enumerate(self.datasets)]))
        self.index_range = [0] + self.index_range
        logger.info(f"Indices of LeoMix datasets: {self.index_range}")

    def __len__(self):
        return self.index_range[-1]

    def streamline_output(self, data_dict):
        new_data_dict = {}
        for key in self.leomix_required_keys:
            if key not in data_dict:
                raise ValueError(f"Key {key} is missing in LeoMix data_dict")
            else:
                new_data_dict[key] = data_dict[key]
        return new_data_dict

    def __getitem__(self, index):
        for i in range(len(self.index_range)-1):
            if self.index_range[i] <= index < self.index_range[i+1]:
                data_dict = self.datasets[i][index-self.index_range[i]]
                break

        return self.streamline_output(data_dict)
