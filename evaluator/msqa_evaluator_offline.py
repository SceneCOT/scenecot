from eval_utils import load_json, execute_chat, save_to_json, path_verify
import re
import os
import torch
import json
from tqdm import tqdm
from lang_eval.evaluator import Evaluator
import hydra
from omegaconf import OmegaConf
import logging
from eval_utils import parse_cot_answer, replace_cot_tokens_with_indicators
logger = logging.getLogger(__name__)

def extract_question(text):
    # Using regular expression to find text between 'USER:' and 'ASSISTANT:'
    match = re.search(r"USER: (.*?) ASSISTANT:", text)
    return match.group(1) if match else None

def extract_number(text):
    # Using regular expression to find number in text
    match = re.search(r"\d+", text)
    return int(match.group(0)) if match else None

def extract_question_type(text):
    # Pattern to match "This is a/an <type> question"
    match = re.search(r'This is (?:an?|a) (.+?) question', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return 'unknown'  # or raise an error / return 'unknown'

def extract_question_sentences(text):
    # Split text into sentences using punctuation as delimiters
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())

    # Look for first sentence that contains a '?'
    for i, sentence in enumerate(sentences):
        if '?' in sentence:
            return ' '.join(sentences[i:])

    # If no question mark is found, return the last sentence
    return sentences[-1] if sentences else ''

class LLMEvaluator():
    def __init__(self, config):
        self.cfg = config
        self.eval_dict = {"total_cnt": 0}
        self.metric_type_list = ['gpt_score', 'em1', 'em1_strict', 'cider', 'bleu', 'meteor', 'rouge']
        for metric_type in self.metric_type_list:
            self.eval_dict[metric_type] = 0
    
    def update(self, score_dict):
        '''
            update the evaluation results
        '''
        for metric_type in score_dict:
            if metric_type in self.metric_type_list:
                self.eval_dict[metric_type] += score_dict[metric_type]
        self.eval_dict["total_cnt"] += 1
    
    def summary(self):
        '''
        '''
        # self.eval_dict["gpt_score"] = self.eval_dict["gpt_score"]/self.eval_dict["total_cnt"]
        for metric_type in self.eval_dict:
            if metric_type in self.metric_type_list:
                self.eval_dict[metric_type] = self.eval_dict[metric_type]/self.eval_dict["total_cnt"]
        return self.eval_dict

    def get_gpt_score(self, question, answer, gt):
        '''
            evaluate the results
        '''
        model = self.cfg.gpt_model
        api_key = self.cfg.api_key
        api_version = self.cfg.api_version
        region = self.cfg.region
        messages = load_json(self.cfg.gpt_score_prompt_path)
        user_prompt = "\n".join([f"Question: {question}", f"Answer: {answer}", f"Ground Truth: {gt}"])
        messages.append({"role": "user", "content": user_prompt})
        response = execute_chat(messages, api_key, model)
        score = extract_number(response)
        return score

class MSQAEvaluator():
    '''
        process the data from files
    '''
    def __init__(self, config):
        self.cfg = config
        self.eval_dict = {"gpt_score": 0, "total_cnt": 0}
        self.evaluator = LLMEvaluator(self.cfg)

    def eval_metrics(self):

        result_dir = self.cfg.result_dir
        dataset_names_list = self.cfg.evaluate_dataset
        file_tag = 'with_gpt_score' if self.cfg.gpt_score_flag else 'without_gpt_score'
        for model_name in self.cfg.evaluated_models:
            for dataset_name in dataset_names_list:
                result_score_file = os.path.join(result_dir, model_name, "eval_results", dataset_name, f"eval_results_{file_tag}.json")
                if os.path.exists(result_score_file):
                    continue
                result_file = os.path.join(result_dir, model_name, "eval_results", dataset_name, "results.pt") if os.path.exists(os.path.join(result_dir, model_name, "eval_results", dataset_name, "results.pt")) else os.path.join(result_dir, model_name, "eval_results", dataset_name, "results.json")
                if not os.path.exists(result_file):
                    result_file = os.path.join(result_dir, model_name, "eval_results", dataset_name,"results.json")
                if '.pt' in result_file:
                    result_dict_list = torch.load(result_file, map_location="cpu")
                elif 'json' in result_file:
                    result_dict_list = load_json(result_file)
                
                score_list = []
                for i in tqdm(range(len(result_dict_list))):
                    result_dict = result_dict_list[i]
                    if 'question' in result_dict:
                        question = result_dict["question"]
                    else:
                        if "instruction" in result_dict:
                            question = extract_question(result_dict["instruction"])  
                        elif "prompt" in result_dict:
                            question = result_dict["prompt"]
                    if "response_gt" in result_dict:
                        gt = result_dict["response_gt"][0]
                    elif "gt_answer" in result_dict:
                        gt = result_dict["gt_answer"]
                    if "response_pred" in result_dict:
                        answer = result_dict["response_pred"]
                    elif 'text' in result_dict:
                        answer = result_dict["text"]

                    question = extract_question_sentences(question)
                    if 'answer the question' in answer:
                        if "<answer>" in answer and "</answer>" in answer:
                            answer = parse_cot_answer(replace_cot_tokens_with_indicators(answer))['answer']
                            gt = parse_cot_answer(replace_cot_tokens_with_indicators(gt[0]))['answer']
                    scored_dict = {"question": question, "answer": answer, "gt": gt}
                    if 'index' in result_dict:
                        if type(result_dict['index']) == str:
                            index = int(result_dict['index'])
                        elif type(result_dict['index']) == int:
                            index = result_dict['index']
                        else:
                            index = int(result_dict['index'].item())
                    else:
                        index = None
                    if 'type' in result_dict:
                        scored_dict['type'] = result_dict['type']
                    elif 'question_type' in result_dict:
                        scored_dict['type'] = result_dict['question_type']
                    elif 'type' not in scored_dict:
                        scored_dict['type'] = extract_question_type(answer)
                    print("question: ", question)
                    print("answer: ", answer)
                    print("gt: ", gt)
                    if 'with_gpt_score' in result_score_file:
                        score = self.evaluator.get_gpt_score(question, answer, gt)
                        scored_dict['gpt_score'] = (score-1)*25
                        print("gpt score: ", scored_dict['gpt_score'])
                        print(f"\n")
                    lang_evaluator = Evaluator()
                    scored_dict.update(lang_evaluator.eval_instance(answer, [gt]))
                    self.evaluator.update(scored_dict)
                    if index:
                        scored_dict['index'] = index
                    score_list.append(scored_dict)
                summary_dict = self.evaluator.summary()
                save_to_json(result_score_file, score_list)
        
        dataset_names_list = self.cfg.evaluate_dataset
        testset_mapping_dict = load_json(self.cfg.testset_mapping_path)
        QA_type_list = [
            "counting",
            "existence",
            "attribute",
            "spatial relationship",
            "navigation",
            "refer",
            "affordance",
            "description",
            "room type",
        ]
        # statistic_dict = {'scannet': {}, 
        #                  'RScan': {},
        #                  'ARKitScenes': {}}
        statistic_dict = {}
        for dataset in dataset_names_list:
            statistic_dict[dataset] = {}
            
        testset_mapping_dict = load_json(self.cfg.testset_mapping_path)
        # metric_type_list = ['em1', 'em1_strict']
        metric_type_list = []
        result_dict = {}
        result_dir = self.cfg.result_dir
        # result_dir = "/mnt/fillipo/linghu/Situated_Scene_Understanding/eval_results/refined_data_baseline_NIPS/NIPS_pretrain_refined_data_balanced_3_dataset/eval_results/scannet"
        file_tag = 'with_gpt_score' if self.cfg.gpt_score_flag else 'without_gpt_score'
        if file_tag == 'with_gpt_score':
            metric_type_list.append('gpt_score')
        for model_name in self.cfg.evaluated_models:
            for dataset_name in dataset_names_list:
                result_score_file = os.path.join(result_dir, model_name, "eval_results", dataset_name, f"eval_results_{file_tag}.json")
                if not os.path.exists(result_score_file):
                    result_score_file = result_score_file.replace("/scannet", "/QACOTScanNetMSR3D")
                scores_data = load_json(result_score_file)
                data_instance_cnt = 0
                for data_instance in scores_data:
                    if 'type' in data_instance:
                        data_QA_type = data_instance['type']
                    else:
                        data_QA_type = testset_mapping_dict[dataset_name][str(data_instance['index'])]['image_text']['type']
                    for metric_type in metric_type_list:
                        if metric_type not in statistic_dict[dataset_name]:
                            statistic_dict[dataset_name][metric_type] = {}
                        for QA_type in QA_type_list:
                            if QA_type in data_QA_type.lower():
                                if QA_type not in statistic_dict[dataset_name][metric_type]:
                                    statistic_dict[dataset_name][metric_type][QA_type] = {'score': [], 'cnt': 0, "avg": 0}
                                if metric_type not in data_instance:
                                    statistic_dict[dataset_name][metric_type][QA_type]['score'].append((data_instance['score']-1)*25)
                                else:
                                    statistic_dict[dataset_name][metric_type][QA_type]['score'].append(data_instance[metric_type])
                                statistic_dict[dataset_name][metric_type][QA_type]['cnt'] += 1

            for dataset_name in dataset_names_list:
                for metric_type in metric_type_list:
                    for QA_type in QA_type_list:
                        if QA_type in statistic_dict[dataset_name][metric_type]:
                            statistic_dict[dataset_name][metric_type][QA_type]['avg'] = sum(statistic_dict[dataset_name][metric_type][QA_type]['score'])/statistic_dict[dataset_name][metric_type][QA_type]['cnt']
            
            for metric_type in metric_type_list:
                statistic_dict[metric_type] = {'overall': {}}
                for QA_type in QA_type_list:
                    score_list = []
                    cnt = 0
                    for dataset_name in dataset_names_list:
                        if QA_type in statistic_dict[dataset_name][metric_type]:
                            score_list.append(statistic_dict[dataset_name][metric_type][QA_type]['avg'] * statistic_dict[dataset_name][metric_type][QA_type]['cnt'])
                            cnt += statistic_dict[dataset_name][metric_type][QA_type]['cnt']
                    if cnt > 0:
                        statistic_dict[metric_type]['overall'][QA_type] = sum(score_list)/cnt

            merged_QA_type_list = ['counting', 'existence', 'attribute_description', 'spatial_refer', 'navigation', 'others']
            for metric_type in metric_type_list:
                statistic_dict[metric_type]['merged'] = {}

            for metric_type in metric_type_list:
                for QA_type in merged_QA_type_list:
                    score_list = []
                    cnt = 0
                    for dataset_name in dataset_names_list:
                        if QA_type in ['counting', 'existence', 'navigation']:
                            if QA_type in statistic_dict[dataset_name][metric_type]:
                                score_list.append(statistic_dict[dataset_name][metric_type][QA_type]['avg'] * statistic_dict[dataset_name][metric_type][QA_type]['cnt'])
                                cnt += statistic_dict[dataset_name][metric_type][QA_type]['cnt']
                        elif QA_type == 'attribute_description':
                            if 'attribute' in statistic_dict[dataset_name][metric_type]:
                                score_list.append(statistic_dict[dataset_name][metric_type]['attribute']['avg'] * statistic_dict[dataset_name][metric_type]['attribute']['cnt'])
                                cnt += statistic_dict[dataset_name][metric_type]['attribute']['cnt']
                            if 'description' in statistic_dict[dataset_name][metric_type]:
                                score_list.append(statistic_dict[dataset_name][metric_type]['description']['avg'] * statistic_dict[dataset_name][metric_type]['description']['cnt'])
                                cnt += statistic_dict[dataset_name][metric_type]['description']['cnt']
                        elif QA_type == 'spatial_refer':
                            if 'spatial relationship' in statistic_dict[dataset_name][metric_type]:
                                score_list.append(statistic_dict[dataset_name][metric_type]['spatial relationship']['avg'] * statistic_dict[dataset_name][metric_type]['spatial relationship']['cnt'])
                                cnt += statistic_dict[dataset_name][metric_type]['spatial relationship']['cnt']
                            if 'refer' in statistic_dict[dataset_name][metric_type]:
                                score_list.append(statistic_dict[dataset_name][metric_type]['refer']['avg'] * statistic_dict[dataset_name][metric_type]['refer']['cnt'])
                                cnt += statistic_dict[dataset_name][metric_type]['refer']['cnt']
                        elif QA_type == 'others':
                            if 'affordance' in statistic_dict[dataset_name][metric_type]:
                                score_list.append(statistic_dict[dataset_name][metric_type]['affordance']['avg'] * statistic_dict[dataset_name][metric_type]['affordance']['cnt'])
                                cnt += statistic_dict[dataset_name][metric_type]['affordance']['cnt']
                            if 'room type' in statistic_dict[dataset_name][metric_type]:
                                score_list.append(statistic_dict[dataset_name][metric_type]['room type']['avg'] * statistic_dict[dataset_name][metric_type]['room type']['cnt'])
                                cnt += statistic_dict[dataset_name][metric_type]['room type']['cnt']
                        else:
                            ValueError("Invalid QA type")
                    if cnt > 0:
                        statistic_dict[metric_type]['merged'][QA_type] = sum(score_list)/cnt
                        statistic_dict[metric_type]['merged'][QA_type + '_cnt'] = cnt
                statistic_dict[metric_type]['merged']['weighted_avg_score'] = sum([statistic_dict[metric_type]['merged'][QA_type] * statistic_dict[metric_type]['merged'][QA_type + '_cnt'] for QA_type in merged_QA_type_list])/sum([statistic_dict[metric_type]['merged'][QA_type + '_cnt'] for QA_type in merged_QA_type_list])
                result_dict = {}
                # for key in statistic_dict['em1']['merged']:
                #     if 'cnt' in key:
                #         continue
                #     if 'weighted' in key:
                #         result_dict['EM-R_overall'] = statistic_dict['em1']['merged'][key]
                #     else:
                #         result_dict[f'EM-R_{key}'] = statistic_dict['em1']['merged'][key]
                if 'gpt_score' in statistic_dict:
                    for key in statistic_dict['gpt_score']['merged']:
                        if 'cnt' in key:
                            continue
                        if 'weighted' in key:
                            result_dict['GPT-Score_overall'] = statistic_dict['gpt_score']['merged'][key]
                        else:
                            result_dict[f'GPT-Score_{key}'] = statistic_dict['gpt_score']['merged'][key]
            save_path = os.path.join(result_dir, model_name, 'eval_results')
            path_verify(save_path)
            
            save_to_json(os.path.join(save_path, f"MSQA_eval_results_{file_tag}.json"), statistic_dict)

def evaluate_execute(cfg):
    '''
        execute the evaluation
    '''
    msqaevaluator = MSQAEvaluator(cfg)
    msqaevaluator.eval_metrics()

@hydra.main(version_base=None, config_path="msqa", config_name="configs")
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    evaluate_execute(cfg) 

if __name__ == "__main__":
    main()