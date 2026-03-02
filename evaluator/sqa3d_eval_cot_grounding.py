import json

from data.data_utils import clean_answer
from evaluator.build import EVALUATOR_REGISTRY
from evaluator.scanqa_eval import ScanQAEvaluator
from data.cot_utils import parse_cot_answer, replace_cot_tokens_with_indicators
import torch

@EVALUATOR_REGISTRY.register()
class SQA3DCOTGroundingEvaluator(ScanQAEvaluator):
    def reset(self):
        self.eval_dict = {
            'target_metric': [], 'em_overall': [], 'em_refined_overall': [],
            'em_type0': [], 'em_refined_type0': [], 'em_type1': [], 'em_refined_type1': [],
            'em_type2': [], 'em_refined_type2': [], 'em_type3': [], 'em_refined_type3': [],
            'em_type4': [], 'em_refined_type4': [], 'em_type5': [], 'em_refined_type5': [],
            'cider_overall': 0, 'bleu_overall': 0, 'meteor_overall': 0, 'rouge_overall': 0,
            'recall': [], 'fpr': [], 'precision': [], 'f1_score': [],
        }
        self.total_count = 0
        self.type_count = {0: 1e-10, 1: 1e-10, 2: 1e-10, 3: 1e-10, 4: 1e-10, 5: 1e-10}
        self.save_results = []

    def batch_metrics(self, data_dict):
        metrics = {
            'type0_count': 1e-10, 'type1_count': 1e-10, 'type2_count': 1e-10,
            'type3_count': 1e-10, 'type4_count': 1e-10, 'type5_count': 1e-10,
        }

        em_overall = 0
        em_refined_overall = 0
        em_type = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        em_refined_type = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

        for answer_pred, answer_gts, sqa_type in zip(
            data_dict['output_txt'], data_dict['output_gt'], data_dict['sqa_type']
        ):
            if parse_cot_answer(replace_cot_tokens_with_indicators(answer_gts[0]))['answer'] != "Empty content for this key!":
                answer_pred = parse_cot_answer(answer_pred)['answer']
                answer_gts = [parse_cot_answer(replace_cot_tokens_with_indicators(gt))['answer'] for gt in answer_gts]
            answer_pred = clean_answer(answer_pred)
            answer_gts = [clean_answer(gt) for gt in answer_gts]
            em_flag, em_refined_flag = self.answer_match(pred=answer_pred, gts=answer_gts)
            em_overall += em_flag
            em_refined_overall += em_refined_flag

            em_type[sqa_type] += em_flag
            em_refined_type[sqa_type] += em_refined_flag
            metrics[f'type{sqa_type}_count'] += 1

            print(f'************* Evaluation *************')
            print(f'answer_gts: {answer_gts}\n')
            print(f'answer_pred: {answer_pred}\n')
            print(f'em_flag: {em_flag}\n')


        batch_size = len(data_dict['output_gt'])
        metrics['total_count'] = batch_size
        metrics['em_overall'] = em_overall / batch_size
        metrics['em_refined_overall'] = em_refined_overall / batch_size
        for key in em_type.keys():
            metrics[f'em_type{key}'] = em_type[key] / metrics[f'type{key}_count']
            metrics[f'em_refined_type{key}'] = em_refined_type[key] / metrics[f'type{key}_count']

        metrics['target_metric'] = metrics['em_refined_overall']

        if 'pred_obj_prob' in data_dict:
            # print(f'pred_obj_prob: {data_dict["pred_obj_prob"]}\n')
            data_dict['pred_obj_prob'] = torch.tensor(data_dict['pred_obj_prob'])
            # threshold = data_dict['pred_obj_prob'].sum(1) / (data_dict['pred_obj_prob'] > 0.01).sum(1)
            # threshold = threshold.unsqueeze(1)
            threshold = 0.5
            pred_obj_mask = data_dict['pred_obj_prob'] > threshold
            # print(f'pred_obj_mask: {pred_obj_mask}\n')
            data_dict['pred_obj_mask'] = pred_obj_mask
            data_dict['grounding_obj_mask_gt'] = torch.tensor(data_dict['grounding_obj_mask_gt'])
            data_dict['obj_masks'] = torch.tensor(data_dict['obj_masks'])
            gt_obj_mask = data_dict['grounding_obj_mask_gt'] > 0.5
            obj_masks = data_dict['obj_masks']

            if len(pred_obj_mask.shape) > len(gt_obj_mask.shape):
                pred_obj_mask = pred_obj_mask.squeeze(1)

            # import ipdb
            # ipdb.set_trace()
            if pred_obj_mask.shape != gt_obj_mask.shape:
                metrics['recall'] = 0.0
                metrics['fpr'] = 0.0
                metrics['precision'] = 0.0
                metrics['f1_score'] = 0.0
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

        return metrics

    def update(self, data_dict):
        metrics = self.batch_metrics(data_dict)
        batch_size = metrics['total_count']
        self.total_count += batch_size
        for key in metrics.keys():
            if 'type' in key and 'count' in key:
                # type{x}_count
                self.type_count[int(key[4])] += metrics[key]

        for i in range(batch_size):
            self.save_results.append({
                # vision
                'source': data_dict['source'][i],
                'scene_id': data_dict['scene_id'][i],
                'position': data_dict['pos'][i],
                'orientation': data_dict['ori'][i],
                # language
                'instruction': data_dict['input_txt'][i],
                'response_gt': data_dict['output_gt'][i],
                'response_pred': data_dict['output_txt'][i],
            })

        # save eval dict
        for key in self.eval_dict.keys():
            if key in ['cider_overall', 'bleu_overall', 'meteor_overall', 'rouge_overall']:
                continue
            if 'type' in key:
                self.eval_dict[key].append(metrics[key] * metrics[f'type{key[-1]}_count'])
            else:
                self.eval_dict[key].append(metrics[key] * batch_size) 

        print(f'************* Evaluation Metric *************')
        print(f'em_overall: {metrics["em_overall"]}\n')
        print(f'em_refined_overall: {metrics["em_refined_overall"]}\n')
        print(f"current target_metric: {self.eval_dict['target_metric']}\n")

    def record(self, split, is_main_process):

        # others
        for k, v in self.eval_dict.items():
            if k in ['cider_overall', 'bleu_overall', 'meteor_overall', 'rouge_overall']:
                continue
            if 'type' in k:
                self.eval_dict[k] = sum(v) / self.type_count[int(k[-1])]
            else:
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
