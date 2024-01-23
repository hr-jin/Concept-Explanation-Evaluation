from .itc import InputTopicCoherenceEvaluator
from .otc import OutputTopicCoherenceEvaluator
from .faithfulness import FaithfulnessEvaluator

EVALUATORS = {
    InputTopicCoherenceEvaluator.code(): InputTopicCoherenceEvaluator,
    OutputTopicCoherenceEvaluator.code(): OutputTopicCoherenceEvaluator,
    FaithfulnessEvaluator.code(): FaithfulnessEvaluator,
}

def evaluator_factory(cfg, activation_func, model):
    evaluator = EVALUATORS[cfg['evaluator']]
    return evaluator(cfg, activation_func, model)