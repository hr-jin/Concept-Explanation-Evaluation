from .base import BaseMetricEvaluator
import torch
import torch.nn as nn
from logger import logger
from audtorch.metrics.functional import pearsonr
import numpy as np

class ValidityRelevanceEvaluator(nn.Module, BaseMetricEvaluator):
    def __init__(self, cfg):
        nn.Module.__init__(self)
        BaseMetricEvaluator.__init__(self, cfg)
        
    @classmethod
    def code(cls):
        return 'vr'
    
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
        metrics = torch.tensor(metric_list).transpose(0,1) # n_metrics, n_iterations
        
        pearsonr_list = []
        for i in metrics:
            tmp_list = []
            for j in metrics:
                # tmp_list.append(pearsonr(i, j))
                tmp_list.append(torch.cosine_similarity(i, j, dim=-1))
            pearsonr_list.append(tmp_list)
        metrics = np.array(pearsonr_list) # n_metrics * n_metrics
        logger.info('Metric Validity Relevance: \n{}'.format(str(metrics)))    
        return metrics
        
            