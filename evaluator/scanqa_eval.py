import json
from pathlib import Path

import numpy as np

from data.data_utils import clean_answer
from evaluator.build import EVALUATOR_REGISTRY
from evaluator.ngram_metrics.bleu.bleu import Bleu
from evaluator.ngram_metrics.cider.cider import Cider
from evaluator.ngram_metrics.meteor.meteor import Meteor
from evaluator.ngram_metrics.rouge.rouge import Rouge
import torch

@EVALUATOR_REGISTRY.register()
class ScanQAEvaluator():
    def __init__(self, cfg, dataset_name):
        self.dataset_name = dataset_name

        self.cider_scorer = Cider()
        self.bleu_scorer = Bleu()
        self.meteor_scorer = Meteor()
        self.rouge_scorer = Rouge()

        self.best_result = -np.inf

        self.save_dir = Path(cfg.exp_dir) / 'eval_results' / dataset_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.reset()

    def reset(self):
        self.eval_dict = {
            'target_metric': [], 'em': [], 'em_refined': [],
            'cider': 0, 'bleu': 0, 'meteor': 0, 'rouge': 0,
        }
        self.total_count = 0
        self.save_results = []
        self.gt_sentence_mp = []
        self.pred_sentence_mp = []

    def answer_match(self, pred, gts):
        # return EM and refined EM
        for gt in gts:
            if pred == gt:
                return 1, 1
            elif ''.join(pred.split()) in ''.join(gt.split()):
                return 0, 1
            elif ''.join(gt.split()) in ''.join(pred.split()):
                return 0, 1
        return 0, 0

    def batch_metrics(self, data_dict):
        metrics = {}
        em = 0
        em_refined = 0
        for answer_pred, answer_gts in zip(data_dict['output_txt'], data_dict['output_gt']):
            answer_pred = clean_answer(answer_pred)
            answer_gts = [clean_answer(gt) for gt in answer_gts]
            print(f'in scanqa_eval.py line 59, output_gt: {data_dict["output_gt"]}')
            print(f'in scanqa_eval.py line 60, output_txt: {data_dict["output_txt"]}')
            em_flag, em_refined_flag = self.answer_match(pred=answer_pred, gts=answer_gts)
            em += em_flag
            em_refined += em_refined_flag

            self.pred_sentence_mp.append([answer_pred])
            self.gt_sentence_mp.append(answer_gts)

        batch_size = len(data_dict['output_gt'])
        metrics['total_count'] = batch_size
        metrics['em'] = em / batch_size
        metrics['em_refined'] = em_refined / batch_size
        metrics['target_metric'] = metrics['em_refined']

        # grounding metrics
        if 'pred_obj_prob' in data_dict:
            threshold = data_dict['pred_obj_prob'].sum(1) / (data_dict['pred_obj_prob'] > 0.01).sum(1)
            threshold = threshold.unsqueeze(1)
            pred_obj_mask = data_dict['pred_obj_prob'] > threshold
            data_dict['pred_obj_mask'] = pred_obj_mask
            gt_obj_mask = data_dict['grounding_obj_mask_gt'] > 0.5
            obj_masks = data_dict['obj_masks']

            if pred_obj_mask.shape != gt_obj_mask.shape:
                metrics['recall'] = 0.0
                metrics['fpr'] = 0.0
                metrics['precision'] = 0.0
                metrics['f1_score'] = 0.0
                metrics['top1_accuracy'] = 0.0
            else:
                TP_mask = torch.logical_and(pred_obj_mask, gt_obj_mask)
                TP = torch.logical_and(TP_mask, obj_masks).sum()
                FN_mask = torch.logical_and(gt_obj_mask, torch.logical_not(pred_obj_mask))
                FN = torch.logical_and(FN_mask, obj_masks).sum()
                FP_mask = torch.logical_and(torch.logical_not(gt_obj_mask), pred_obj_mask)
                FP = torch.logical_and(FP_mask, obj_masks).sum()
                TN_mask = torch.logical_and(torch.logical_not(gt_obj_mask), torch.logical_not(pred_obj_mask))
                TN = torch.logical_and(TN_mask, obj_masks).sum()

                recall = TP / (TP + FN + 1e-7)
                precision = TP / (TP + FP + 1e-7)
                fpr = FP / (FP + TN + 1e-7)
                f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)

                batch_size = gt_obj_mask.shape[0]

                metrics['total_count'] = batch_size 
                metrics['recall'] = recall
                metrics['fpr'] = fpr
                metrics['precision'] = precision
                metrics['f1_score'] = f1_score
        else:
            metrics['recall'] = 0.0
            metrics['fpr'] = 0.0
            metrics['precision'] = 0.0
            metrics['f1_score'] = 0.0
            metrics['top1_accuracy'] = 0.0

        return metrics

    def update(self, data_dict):
        metrics = self.batch_metrics(data_dict)
        batch_size = metrics['total_count']
        self.total_count += batch_size

        for i in range(batch_size):
            self.save_results.append({
                # vision
                'source': data_dict['source'][i],
                'scene_id': data_dict['scene_id'][i],
                # language
                'instruction': data_dict['input_txt'][i],
                'response_gt': data_dict['output_gt'][i],
                'response_pred': data_dict['output_txt'][i],
            })

        for key in self.eval_dict.keys():
            if key not in ['cider', 'bleu', 'meteor', 'rouge']:
                self.eval_dict[key].append(metrics[key] * batch_size)

    def record(self, split, is_main_process):
        # ngram metrics
        self.gt_sentence_mp = {k: v for k, v in enumerate(self.gt_sentence_mp)}
        self.pred_sentence_mp = {k: v for k, v in enumerate(self.pred_sentence_mp)}

        try:
            self.eval_dict['cider'] = self.cider_scorer.compute_score(self.gt_sentence_mp, self.pred_sentence_mp)[0]
        except:
            print('Cider error')
            print(f'gt_sentence_mp: {self.gt_sentence_mp}')
            print(f'pred_sentence_mp: {self.pred_sentence_mp}')
        self.eval_dict['bleu'] = self.bleu_scorer.compute_score(self.gt_sentence_mp, self.pred_sentence_mp)[0][-1]
        self.eval_dict['meteor'] = self.meteor_scorer.compute_score(self.gt_sentence_mp, self.pred_sentence_mp)[0]
        self.eval_dict['rouge'] = self.rouge_scorer.compute_score(self.gt_sentence_mp, self.pred_sentence_mp)[0]

        # others
        for k, v in self.eval_dict.items():
            if k not in ['cider', 'bleu', 'meteor', 'rouge']:
                self.eval_dict[k] = sum(v) / self.total_count

        if self.eval_dict['target_metric'] > self.best_result:
            is_best = True
            self.best_result = self.eval_dict['target_metric']
        else:
            is_best = False

        if (is_best or split == 'test') and is_main_process:
            with open(str(self.save_dir / 'results.json'), 'w') as f:
                json.dump(self.save_results, f, indent=2)

        return is_best, self.eval_dict
