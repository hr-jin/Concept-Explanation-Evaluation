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
        concepts=[], 
        concept_idxs=[], 
        **kwargs
    ):           
        logger.info('Metric evaluation ...\n')
        metric_list = []
        for name, evaluator in evaluator_dict.items():   
            logger.info('Evaluating {} ...'.format(name))   
            concept_metric_list = []             
            for j, concept_idx in enumerate(concept_idxs):
                concept = concepts[j]
                evaluator.update_concept(concept, concept_idx) 
                concept_metric = evaluator.get_metric(eval_tokens)
                concept_metric_list.append(concept_metric)
            metric_list.append(concept_metric_list)
        metrics = torch.tensor(metric_list) # n_metrics, n_concepts 
        
        pearsonr_list = []
        for i in metrics:
            tmp_list = []
            for j in metrics:
                tmp_list.append(pearsonr(i, j))
            pearsonr_list.append(tmp_list)
            
        cosine_sim_list = []
        for i in metrics:
            tmp_list = []
            for j in metrics:
                tmp_list.append(torch.cosine_similarity(i, j, dim=-1))
            cosine_sim_list.append(tmp_list)
            
        pearsonr_metrics = torch.tensor(pearsonr_list).cpu().numpy() # n_metrics * n_metrics
        cosine_sim_metrics = torch.tensor(cosine_sim_list).cpu().numpy() # n_metrics * n_metrics
        
        np.save('pearsonr_metrics.npy',pearsonr_metrics)
        np.save('cosine_sim_metrics.npy',cosine_sim_metrics)
        
        logger.info('Metrics: '.format(str(list(evaluator_dict.keys()))))
        logger.info('Metric Validity Relevance (cosine similarity): \n{}'.format(str(cosine_sim_metrics))) 
        logger.info('Metric Validity Relevance (pearsonr similarity): \n{}'.format(str(pearsonr_metrics)))   
        
        
        return cosine_sim_metrics, pearsonr_metrics
        
            