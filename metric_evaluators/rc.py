from .base import BaseMetricEvaluator

import torch
import torch.nn as nn
from logger import logger
from audtorch.metrics.functional import pearsonr
import torch.nn.functional as F

class ReliabilityConsistencyEvaluator(nn.Module, BaseMetricEvaluator):
    def __init__(self, cfg):
        nn.Module.__init__(self)
        BaseMetricEvaluator.__init__(self, cfg)
        
    @classmethod
    def code(cls):
        return 'rc'
    
    def get_metric(
        self, 
        eval_tokens, 
        evaluator_dict=dict(), 
        concepts=[], 
        concept_idxs=[], 
        **kwargs
    ):            
        evaluator_names = list(evaluator_dict.keys())        
        minibatch = self.cfg['metric_eval_batchsize']
        origin_tokens = eval_tokens
        eval_tokens = eval_tokens.split(minibatch, dim=0) 
          
        metric_list = []
        for i, tokens in enumerate(eval_tokens):
            logger.info('Metric evaluation on subdataset {}...\n'.format(i+1))
            tmp_metric_list = []
            for name, evaluator in evaluator_dict.items():  
                logger.info('Evaluating {} ...'.format(name))
                concept_metric_list = []
                for j, concept_idx in enumerate(concept_idxs):
                    concept = concepts[j]
                    evaluator.update_concept(concept, concept_idx) 
                    concept_metric = evaluator.get_metric(tokens)
                    concept_metric_list.append(concept_metric)
                tmp_metric_list.append(concept_metric_list)
            metric_list.append(tmp_metric_list)
        separate_metrics = torch.tensor(metric_list) # n_minibatch, n_metrics, n_concepts
        separate_metrics = separate_metrics.permute(1,0,2) # n_metrics, n_minibatch, n_concepts
        separate_vars_agg = torch.var(separate_metrics, dim=-1)
        print('separate_vars:\n',separate_vars_agg)
        separate_vars_agg = separate_vars_agg.sum(-1) # n_metrics
        integral_vars = torch.var(separate_metrics.sum(1), dim=-1) # n_metrics
        
        J = separate_metrics.shape[1] 
        final_metrics = J / (J - 1) * (integral_vars - separate_vars_agg) / integral_vars # n_metrics
        print('J:', J)
        print('separate_vars_agg:', separate_vars_agg)
        print('integral_vars:', integral_vars)
        logger.info('Metric Consistency: \n{}'.format(
            ' '.join(
                ['{}:{:4f}'.format(evaluator_names[i],final_metrics[i]) for i in range(final_metrics.shape[0])]
                )
            ))    
        return final_metrics
        
            