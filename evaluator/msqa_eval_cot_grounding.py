import json

from data.data_utils import clean_answer
from evaluator.build import EVALUATOR_REGISTRY
from evaluator.scanqa_eval import ScanQAEvaluator
import torch
from data.cot_utils import parse_cot_answer, replace_cot_tokens_with_indicators

qa_type_metric_mapping_dict = {
    'counting': ['em', 'em_refined'],
    'existence': ['em', 'em_refined'],
    'refer': ['em', 'em_refined'],
    'affordance': ['em', 'em_refined'],
    'room type': ['em', 'em_refined'],
    'attribute': ['em', 'em_refined', 'meteor', 'rouge'],
    'description': ['em', 'em_refined', 'meteor', 'rouge'],
    'spatial relationship': ['em', 'em_refined'],
    'navigation': ['em', 'em_refined'],
}

all_metric_types = ['em', 'em_refined', 'meteor', 'rouge', 'cider', 'bleu']

@EVALUATOR_REGISTRY.register()
class MSQACOTGroundingEvaluator(ScanQAEvaluator):
    def reset(self):
        self.eval_dict = {
            'target_metric': [], 'em_overall': [], 'em_refined_overall': [],
            'recall': [], 'fpr': [], 'precision': [], 'f1_score': []
        }
        for qa_type in qa_type_metric_mapping_dict.keys():
            for metric in qa_type_metric_mapping_dict[qa_type]:
                self.eval_dict[f'{qa_type}_{metric}'] = []
        self.total_count = 0
        self.qa_type_count = {}
        for qa_type in qa_type_metric_mapping_dict.keys():
            self.qa_type_count[f'{qa_type}_count'] = 1e-10
        self.save_results = []
        self.gt_sentence_mp = []
        self.pred_sentence_mp = [] 
        self.cap_metric_gt_sentence_mp_dict = {}
        self.cap_metric_pred_sentence_mp_dict = {}
        with_caption_metric_qa_types = ['attribute', 'description','spatial relationship']
        for qa_type in with_caption_metric_qa_types:
            self.cap_metric_gt_sentence_mp_dict[qa_type] = []
            self.cap_metric_pred_sentence_mp_dict[qa_type] = []

    def batch_metrics(self, data_dict):
        metrics = {}
        for qa_type in qa_type_metric_mapping_dict.keys():
            metrics[f'{qa_type}_count'] = 1e-10
            metrics[f'{qa_type}_em'] = 0
            metrics[f'{qa_type}_em_refined'] = 0

        em_overall = 0
        em_refined_overall = 0

        metric_qa_type_record_dict = {}
        
        for qa_type in qa_type_metric_mapping_dict.keys():
            for metric_type in all_metric_types:
                if metric_type in qa_type_metric_mapping_dict[qa_type]:
                    if metric_type not in metric_qa_type_record_dict:
                            metric_qa_type_record_dict[metric_type] = {}
                    if metric_type == 'em' or metric_type == 'em_refined':
                        metric_qa_type_record_dict[metric_type][qa_type] = 0
                    else:
                        metric_qa_type_record_dict[metric_type][qa_type] = []
            

        for answer_pred, answer_gts, qa_type in zip(
            data_dict['output_txt'], data_dict['output_gt'], data_dict['type']
        ):
            if parse_cot_answer(replace_cot_tokens_with_indicators(answer_gts[0]))['answer'] != "Empty content for this key!":
                answer_pred = parse_cot_answer(answer_pred)['answer']
                answer_gts = [parse_cot_answer(replace_cot_tokens_with_indicators(gt))['answer'] for gt in answer_gts]
            print(f'************* Evaluation *************')
            print(f'answer_gts: {answer_gts}\n')
            print(f'answer_pred: {answer_pred}\n')
            answer_pred = clean_answer(answer_pred)
            answer_gts = [clean_answer(gt) for gt in answer_gts]
            em_flag, em_refined_flag = self.answer_match(pred=answer_pred, gts=answer_gts)
            em_overall += em_flag
            em_refined_overall += em_refined_flag
            print(f'em_flag: {em_flag}\n')


            if qa_type not in qa_type_metric_mapping_dict.keys():
                continue

            metrics[f'{qa_type}_count'] += 1

            if qa_type not in ['attribute', 'description','spatial relationship']:
                metric_qa_type_record_dict['em'][qa_type] += em_flag
            
            if qa_type not in ['description']:
                metric_qa_type_record_dict['em_refined'][qa_type] += em_refined_flag
            
            if qa_type in ['attribute', 'description','spatial relationship']:
                self.cap_metric_gt_sentence_mp_dict[qa_type].append(answer_gts)
                self.cap_metric_pred_sentence_mp_dict[qa_type].append([answer_pred])


        batch_size = len(data_dict['output_gt'])
        metrics['total_count'] = batch_size
        metrics['em_overall'] = em_overall / batch_size
        metrics['em_refined_overall'] = em_refined_overall / batch_size
        for qa_type in metric_qa_type_record_dict['em'].keys():
            metrics[f'{qa_type}_em'] = metric_qa_type_record_dict['em'][qa_type]

        for qa_type in metric_qa_type_record_dict['em_refined'].keys():    
            metrics[f'{qa_type}_em_refined'] = metric_qa_type_record_dict['em_refined'][qa_type]

        # print(f'line 100: metrics: {metrics}\n')

        metrics['target_metric'] = metrics['em_refined_overall']

        # grounding metrics
        # import ipdb07
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
        for qa_type in qa_type_metric_mapping_dict.keys():
            self.qa_type_count[f'{qa_type}_count'] += metrics[f'{qa_type}_count']

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
                'type': data_dict['type'][i]
            })

        # save eval dict
        for key in self.eval_dict.keys():
            if key in ['em_overall', 'em_refined_overall', 'target_metric']:
                continue
            if 'em' in key or 'em_refined' in key:
                self.eval_dict[key].append(metrics[key])
        
        for key in ['em_overall', 'em_refined_overall', 'target_metric']:
            self.eval_dict[key].append(metrics[key] * batch_size)
        
        for key in ['recall', 'fpr', 'precision', 'f1_score']:
            self.eval_dict[key].append(metrics[key] * batch_size)

    def record(self, split, is_main_process):
        for qa_type in self.cap_metric_gt_sentence_mp_dict.keys():
            if len(self.cap_metric_gt_sentence_mp_dict[qa_type]) == 0:
                continue
            print(f'record: line 159: self.cap_metric_gt_sentence_mp_dict[qa_type]: {self.cap_metric_gt_sentence_mp_dict[qa_type]}\n')
            self.cap_metric_gt_sentence_mp_dict[qa_type] = {k: v for k, v in enumerate(self.cap_metric_gt_sentence_mp_dict[qa_type])}
            self.cap_metric_pred_sentence_mp_dict[qa_type] = {k: v for k, v in enumerate(self.cap_metric_pred_sentence_mp_dict[qa_type])}
            if qa_type == 'description':
                self.eval_dict['description_cider'] = self.cider_scorer.compute_score(self.cap_metric_gt_sentence_mp_dict[qa_type], self.cap_metric_pred_sentence_mp_dict[qa_type])[0]
                self.eval_dict['description_meteor'] = self.meteor_scorer.compute_score(self.cap_metric_gt_sentence_mp_dict[qa_type], self.cap_metric_pred_sentence_mp_dict[qa_type])[0]
                self.eval_dict['description_rouge'] = self.rouge_scorer.compute_score(self.cap_metric_gt_sentence_mp_dict[qa_type], self.cap_metric_pred_sentence_mp_dict[qa_type])[0]
            elif qa_type == 'attribute':
                self.eval_dict['attribute_meteor'] = self.meteor_scorer.compute_score(self.cap_metric_gt_sentence_mp_dict[qa_type], self.cap_metric_pred_sentence_mp_dict[qa_type])[0]
        # em, em_refined
        print(f'record: line 159: self.eval_dict: {self.eval_dict}\n')
        for k, v in self.eval_dict.items():
            if k in ['description_cider', 'description_meteor', 'description_rouge', 'attribute_meteor', 'spatial relationship_meteor']:
                continue
            if k.split('_')[0] in qa_type_metric_mapping_dict.keys():
                self.eval_dict[k] = sum(v) / self.qa_type_count[f'{k.split("_")[0]}_count']
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
