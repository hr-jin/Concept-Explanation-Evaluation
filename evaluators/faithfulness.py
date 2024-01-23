from .base import BaseEvaluator
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import pprint
from logger import logger
import os
import time
import numpy as np
from tqdm import tqdm
from transformer_lens import HookedTransformer

class FaithfulnessEvaluator(nn.Module, BaseEvaluator):
    def __init__(self, cfg, activation_func, model):
        nn.Module.__init__(self)
        BaseEvaluator.__init__(self, cfg, activation_func, model)
        
    @classmethod
    def code(cls):
        return 'faithfulness'
    
    def get_metric(self, 
                   eval_tokens, 
                   concept=None, 
                   concept_idx=-1, 
                   disturb='ablation', 
                   measure_obj='loss', 
                   corr_func='cosine', 
                   class_idx=0, 
                   logits_corr_topk=None
        ):
        
        if measure_obj not in ['loss', 'class_logit', 'logits']:
            assert False, "measure_obj must be one of ['loss', 'class_logit', 'logits']."
            
        if disturb not in ['ablation', 'gradient', 'replace']:
            assert False, "disturb must be one of ['ablation', 'gradient', 'replace']."
        
        if corr_func not in ['cosine', 'KL_div', 'openai_var']:
            assert False, "corr_func must be one of ['cosine', 'KL_div', 'openai_var']."
        
        _, maxlen = eval_tokens.shape[0], eval_tokens.shape[1]
        minibatch = self.cfg['concept_eval_batchsize']
        eval_tokens = eval_tokens.split(minibatch, dim=0)
            
        metrics = []
        concept_acts = []
        params = self.model.parameters()
        optimizer = torch.optim.Adam(params, lr=0.001)
        for tokens in tqdm(eval_tokens, desc='Traverse the evaluation corpus to calculate metrics'):            
            
            concept_act = self.activation_func(tokens, self.model, concept, concept_idx) # minibatch * maxlen
            concept_acts.append(concept_act.cpu().reshape(minibatch, maxlen).numpy())
            
            if disturb == 'gradient':
                if measure_obj == 'logits':
                    assert False, "When the disturbance type is 'gradient', the measurement object must be one of ['loss', 'class_logit']."
                elif measure_obj == 'loss':
                    grads, hidden_state = self.get_loss_gradient(tokens)
                    metric = (grads @ concept.cpu().numpy().T).squeeze() # minibatch * maxlen
                elif measure_obj == 'class_logit':
                    grads, hidden_state = self.get_class_logit_gradient(tokens, class_idx)
                    metric = (grads @ concept.cpu().numpy().T).squeeze() # minibatch * maxlen
                metric = metric[:,:-1]
                optimizer.zero_grad()
                
            elif disturb == 'ablation':
                with torch.no_grad():
                    if measure_obj == 'logits':
                        metric = self.get_logit_distribution_corr(tokens, 
                                                                concept=concept, 
                                                                hook=self.ablation_hook, 
                                                                topk=logits_corr_topk, 
                                                                corr_func=corr_func) # minibatch * maxlen
                    elif measure_obj == 'loss':
                        metric = self.get_loss_diff(tokens, 
                                                    concept=concept,
                                                    hook=self.ablation_hook
                                                    ) # minibatch * (maxlen-1)
                    elif measure_obj == 'class_logit':
                        metric = self.get_class_logit_diff(tokens,
                                                        concept=concept,
                                                        class_idx=class_idx,
                                                        hook=self.ablation_hook
                                                        ) # minibatch * maxlen
                
            elif disturb == 'replace':
                with torch.no_grad():
                    if measure_obj == 'logits':
                        metric = self.get_logit_distribution_corr(tokens, 
                                                                concept=concept, 
                                                                hook=self.replacement_hook, 
                                                                topk=logits_corr_topk, 
                                                                corr_func=corr_func)
                    elif measure_obj == 'loss':
                        metric = self.get_loss_diff(tokens, 
                                                    concept=concept,
                                                    hook=self.replacement_hook
                                                    )
                    elif measure_obj == 'class_logit':
                        metric = self.get_class_logit_diff(tokens,
                                                        concept=concept,
                                                        class_idx=class_idx,
                                                        hook=self.replacement_hook
                                                        )
            metrics.append(metric)
        
        
        concept_acts = np.concatenate(concept_acts, axis=0)
        metrics = np.concatenate(metrics, axis=0)
        
        if (disturb in ['ablation', 'replace']) and (measure_obj in ['loss']):
            concept_acts = concept_acts[:,:-1]
        
        pos_act_metric = metrics[concept_acts>0].mean()
        
        pos_act_metric_08max = metrics[concept_acts>0.8*concept_acts.max()].mean()
        
        pos_act_metric_09max = metrics[concept_acts>0.9*concept_acts.max()].mean()
        
        weighted_metric = (metrics * concept_acts).mean()
    
        normed_concept_acts = concept_acts / torch.tensor(concept_acts).norm(p=1, keepdim=True).numpy()
        weighted_normed_metric = (metrics * normed_concept_acts).sum()
            
        softmax_concept_acts = torch.softmax(torch.tensor(concept_acts[concept_acts>0]), dim=-1).numpy()
        weighted_softmax_metric = (metrics[concept_acts>0] * softmax_concept_acts).sum()
            
        logger.info('Faithfulness Metrics ({} {} {} logits_corr_topk={})'.format(disturb, measure_obj, corr_func, logits_corr_topk))    
        logger.info('avg where concept activation > 0: {:4E}'.format(pos_act_metric))    
        logger.info('avg where concept activation > 0.8 max: {:4E}'.format(pos_act_metric_08max))  
        logger.info('avg where concept activation > 0.9 max: {:4E}'.format(pos_act_metric_09max))  
        logger.info('weighted avg by concept activation: {:4E}'.format(weighted_metric))    
        logger.info('weighted sum by 1-normed concept activation: {:4E}'.format(weighted_normed_metric))  
        logger.info('weighted sum by softmaxed concept activation: {:4E}'.format(weighted_softmax_metric))      
        return metric
        
            