import json
import os
from .evaluator import Evaluator

def calculate_instance_score(pred, gt):
    answer_all = [[pred, gt]]
    sgqa_evaltor = Evaluator()
    results_dict = sgqa_evaltor.eval_all(answer_all)
    return results_dict

# test_set = 'cache_gpt-3.5-turbo_gpt_short'

# json_dir = os.path.join('./cache', test_set)
# answer_list = os.listdir(json_dir)
# answer_all = []
# for one_json in answer_list:
#     with open(os.path.join(json_dir, one_json), 'r') as f:
#         one_answer = json.load(f)
#     # answer_all.append({
#     #     'pred' : one_answer['pred']['content'], 
#     #     'gt_list': one_answer['qa']['A']
#     # })
#     answer_all.append([
#         one_answer['pred']['content'], 
#         one_answer['qa']['A'][0]
#     ])

# sgqa_evaltor = Evaluator()
# results_dict = sgqa_evaltor.eval_all(answer_all)
# print(results_dict)

# output_file = os.path.join('./results', test_set + '_eval.json')
# with open(output_file, 'w') as f:
#     json.dump(results_dict, f, indent=4)