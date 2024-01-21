from .eva1 import Evaluator1

EVALUATORS = {
    Evaluator1.code(): Evaluator1,
}

def evaluator_factory(cfg, activation_func, model):
    evaluator = EVALUATORS[cfg['evaluator']]
    return evaluator(cfg, activation_func, model)