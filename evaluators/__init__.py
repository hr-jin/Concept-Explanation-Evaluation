from .itc import InputTopicCoherenceEvaluator
from .otc import OutputTopicCoherenceEvaluator

EVALUATORS = {
    InputTopicCoherenceEvaluator.code(): InputTopicCoherenceEvaluator,
    OutputTopicCoherenceEvaluator.code(): OutputTopicCoherenceEvaluator,
}

def evaluator_factory(cfg, activation_func, model):
    evaluator = EVALUATORS[cfg['evaluator']]
    return evaluator(cfg, activation_func, model)