from .base import BaseEvaluator

import torch
import torch.nn as nn
from logger import logger
import numpy as np
from tqdm import tqdm

class FaithfulnessEvaluator(nn.Module, BaseEvaluator):
    def __init__(
        self, 
        cfg, 
        activation_func, 
        hidden_state_func,
        model, 
        concept=None, 
        concept_idx=-1, 
        disturb='ablation', 
        measure_obj='loss', 
        corr_func='pearson', 
        class_idx=0, 
        logits_corr_topk=None,
    ):
        nn.Module.__init__(self)
        BaseEvaluator.__init__(self, cfg, activation_func, hidden_state_func, model)
        self.concept = concept
        self.concept_idx = concept_idx
        self.disturb = disturb
        self.measure_obj = measure_obj
        self.corr_func = corr_func
        self.class_idx = class_idx
        self.logits_corr_topk = logits_corr_topk
        self.return_type = cfg['return_type']
        
    @classmethod
    def code(cls):
        return 'faithfulness'
    
    def update_concept(self, concept=None, concept_idx=-1):
        self.concept = concept
        self.concept_idx = concept_idx
    
    def get_metric(self, eval_tokens, pre_metrics=None, pre_concept_acts=None, return_metric_and_acts=False,**kwargs):
        
        _, maxlen = eval_tokens.shape[0], eval_tokens.shape[1]
        minibatch = self.cfg['concept_eval_batchsize']
        eval_tokens = eval_tokens.split(minibatch, dim=0)
            
        metrics = []
        concept_acts = []
        if pre_metrics is None:
            params = self.model.parameters()
            optimizer = torch.optim.Adam(params, lr=0.001)
            for tokens in tqdm(eval_tokens, desc='Traverse the evaluation corpus to calculate metrics'):            
                
                concept_act = self.activation_func(tokens, self.model, self.concept, self.concept_idx) # minibatch * maxlen
                concept_acts.append(concept_act.cpu().reshape(tokens.shape[0], maxlen).numpy())
                
                if self.disturb == 'gradient':
                    if self.measure_obj == 'logits':
                        assert False, "When the disturbance type is 'gradient', the measurement object must be one of ['loss', 'class_logit']."
                    elif self.measure_obj == 'loss':
                        grads, hidden_state = self.get_loss_gradient(tokens)
                        metric = -(grads @ self.concept.cpu().numpy().T).squeeze() # minibatch * maxlen
                    elif self.measure_obj == 'pred_logit':
                        grads, hidden_state = self.get_class_logit_gradient(tokens, -1, concept_act)
                        metric = (grads @ self.concept.cpu().numpy().T).squeeze() # minibatch * maxlen
                    elif self.measure_obj == 'next_logit':
                        grads, hidden_state = self.get_class_logit_gradient(tokens, -2, concept_act)
                        metric = (grads @ self.concept.cpu().numpy().T).squeeze() # minibatch * maxlen
                        
                    metric = metric[:,:-1]
                    optimizer.zero_grad()
                    
                elif self.disturb == 'ablation':
                    with torch.no_grad():
                        if self.measure_obj == 'logits':
                            metric = -self.get_logit_distribution_corr(
                                tokens, 
                                concept=self.concept, 
                                hook=self.ablation_hook, 
                                topk=self.logits_corr_topk, 
                                corr_func=self.corr_func,
                                concept_act=concept_act,
                            ) # minibatch * maxlen
                        elif self.measure_obj == 'loss':
                            metric = self.get_loss_diff(
                                tokens, 
                                concept=self.concept,
                                hook=self.ablation_hook,
                                concept_act=concept_act,
                            ) # minibatch * (maxlen-1)
                        elif self.measure_obj == 'next_logit':
                            metric = -self.get_class_logit_diff(
                                tokens,
                                concept=self.concept,
                                class_idx=-2,
                                hook=self.ablation_hook,
                                concept_act=concept_act,
                            ) # minibatch * maxlen
                        elif self.measure_obj == 'pred_logit':
                            metric = -self.get_class_logit_diff(
                                tokens,
                                concept=self.concept,
                                class_idx=-1,
                                hook=self.ablation_hook,
                                concept_act=concept_act,
                            ) # minibatch * maxlen
                metrics.append(metric)
            concept_acts = np.concatenate(concept_acts, axis=0)
            metrics = np.concatenate(metrics, axis=0)
        else:
            metrics = pre_metrics
            concept_acts = pre_concept_acts
        
        
        concept_acts = concept_acts[:,:metrics.shape[1]]
        
        origin_acts = concept_acts
        concept_acts = concept_acts * (concept_acts > 0.)
        
        pos_act_metric = metrics[concept_acts>0].mean()
        
        pos_act_metric_0min = metrics[concept_acts<=0*concept_acts.max()].mean()
        
        weighted_metric = (metrics * concept_acts).mean()
        
        weighted_origin_metric = (metrics * origin_acts).mean()
    
        normed_concept_acts = concept_acts / torch.tensor(concept_acts).norm(p=1, keepdim=True).numpy()
        weighted_normed_metric = (metrics * normed_concept_acts).sum()
            
        softmax_concept_acts = torch.softmax(torch.tensor(concept_acts[concept_acts>0]), dim=-1).numpy()
        weighted_softmax_metric = (metrics[concept_acts>0] * softmax_concept_acts).sum()
            
        logger.info('Faithfulness Metrics ({} {} {} logits_corr_topk={})'.format(
            self.disturb, 
            self.measure_obj, 
            self.corr_func, 
            self.logits_corr_topk
        ))    
        logger.info('avg where concept activation < 0: {:4E}'.format(pos_act_metric_0min))    
        logger.info('max activation: {:4E}'.format(concept_acts.max()))    
        logger.info('weighted avg by concept activation: {:4E}'.format(weighted_metric))    
        logger.info('weighted avg by origin concept activation: {:4E}'.format(weighted_origin_metric))  
        logger.info('weighted sum by 1-normed concept activation: {:4E}'.format(weighted_normed_metric))  
        logger.info('weighted sum by softmaxed concept activation: {:4E}'.format(weighted_softmax_metric))
        if self.return_type == 'avg_0max':
            final_metric = pos_act_metric
        elif self.return_type == 'weighted':
            final_metric = weighted_metric
        elif self.return_type == 'weighted_normed':
            final_metric = weighted_normed_metric
        elif self.return_type == 'weighted_softmax':
            final_metric = weighted_softmax_metric
        logger.info('final metric: {:4E}'.format(final_metric))     
        
        if return_metric_and_acts:
            return final_metric, metrics, concept_acts
        else:
            return final_metric 
            