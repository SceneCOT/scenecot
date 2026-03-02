from fvcore.common.registry import Registry

EVALUATOR_REGISTRY = Registry('Evaluator')


def build_eval_leo(cfg, dataset_name, evaluator_name):
    return EVALUATOR_REGISTRY.get(evaluator_name)(cfg, dataset_name)
