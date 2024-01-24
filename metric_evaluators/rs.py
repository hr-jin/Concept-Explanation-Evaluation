from .base import BaseMetricEvaluator
import torch
import torch.nn as nn
from logger import logger
from audtorch.metrics.functional import pearsonr


class ReliabilityStabilityEvaluator(nn.Module, BaseMetricEvaluator):
    def __init__(self, cfg):
        nn.Module.__init__(self)
        BaseMetricEvaluator.__init__(self, cfg)
        
    @classmethod
    def code(cls):
        return 'rs'
    
    def get_metric(
        self, 
        eval_tokens, 
        evaluator_dict=dict(), 
        **kwargs
    ):            
        evaluator_names = list(evaluator_dict.keys())  
        minibatch = self.cfg['metric_eval_batchsize']
        eval_tokens = eval_tokens.split(minibatch, dim=0)  
        
        metric_list = []
        for i, tokens in enumerate(eval_tokens):
            logger.info('Metric evaluation, iter {} ...\n'.format(i+1))
            tmp_metric_list = []
            for name, evaluator in evaluator_dict.items():   
                logger.info('Evaluating {} ...'.format(name))
                metric = evaluator.get_metric(tokens)
                tmp_metric_list.append(metric)
            metric_list.append(tmp_metric_list)
        metrics_1 = torch.tensor(metric_list).transpose(0,1) # n_metrics, n_iterations
        
        metric_list = []
        for i, tokens in enumerate(eval_tokens):
            logger.info('Metric evaluation, iter {} ...\n'.format(i+1))
            tmp_metric_list = []
            for name, evaluator in evaluator_dict.items():   
                logger.info('Evaluating {} ...'.format(name))
                metric = evaluator.get_metric(tokens)
                tmp_metric_list.append(metric)
            metric_list.append(tmp_metric_list)
        metrics_2 = torch.tensor(metric_list).transpose(0,1) # n_metrics, n_iterations
        
        metrics = pearsonr(metrics_1, metrics_2).squeeze() # n_metrics              
        logger.info('Metric Stability: \n{}'.format(
            ' '.join(
                ['{}:{:4f}'.format(evaluator_names[i],metrics[i]) for i in range(metrics.shape[0])]
                )
            ))    
        return metrics
        
            