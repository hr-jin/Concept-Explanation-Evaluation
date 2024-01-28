from .rc import ReliabilityConsistencyEvaluator
from .rs import ReliabilityStabilityEvaluator
from .vr import ValidityRelevanceEvaluator

METRICEVALUATORS = {
    ReliabilityConsistencyEvaluator.code(): ReliabilityConsistencyEvaluator,
    ReliabilityStabilityEvaluator.code(): ReliabilityStabilityEvaluator,
    ValidityRelevanceEvaluator.code(): ValidityRelevanceEvaluator,
}

def metric_evaluator_factory(cfg):
    evaluator = METRICEVALUATORS[cfg['metric_evaluator']]
    return evaluator(cfg)