from .base import BaseEvaluator

import torch
import torch.nn as nn
from logger import logger
import numpy as np

class InputTopicCoherenceEvaluator(nn.Module, BaseEvaluator):
    def __init__(
        self, 
        cfg, 
        activation_func, 
        model, 
        concept=None, 
        concept_idx=-1, 
        pmi_type='uci',
        
    ):
        nn.Module.__init__(self)
        BaseEvaluator.__init__(self, cfg, activation_func, model)
        self.concept = concept
        self.concept_idx = concept_idx
        self.pmi_type = pmi_type
        
    @classmethod
    def code(cls):
        return 'itc'
    
    def update_concept(self, concept=None, concept_idx=-1):
        self.concept = concept
        self.concept_idx = concept_idx
        
    def update_pmi_type(self, pmi_type):
        if pmi_type not in ['uci', 'umass', 'silhouette']:
            assert False, "PMI type not supported yet. please choose from: ['uci', 'umass', 'silhouette']."
        self.pmi_type = pmi_type
        
    
    def get_metric(self, eval_tokens, topic_tokens=None, topic_idxs=None, origin_topic_idxs=None, return_tokens=False, **kwargs):
        if topic_tokens is None:
            most_critical_tokens, most_critical_token_idxs, origin_df, origin_critical_token_idxs = self.get_most_critical_tokens(eval_tokens, self.concept, self.concept_idx)
        else:
            most_critical_tokens = topic_tokens
            most_critical_token_idxs = topic_idxs
            origin_critical_token_idxs = origin_topic_idxs
        most_critical_token_idxs = most_critical_token_idxs[most_critical_tokens != '\ufffd']
        most_critical_tokens = most_critical_tokens[most_critical_tokens != '\ufffd']
        
        if most_critical_tokens.shape[0] == 1:
            most_critical_tokens = np.repeat(most_critical_tokens, 2)
            most_critical_token_idxs = np.repeat(most_critical_token_idxs, 2)
        
        if self.pmi_type == 'uci':  
            itc = self.get_topic_coherence(eval_tokens, most_critical_tokens)
        elif self.pmi_type == 'umass':
            itc = self.get_topic_coherence(eval_tokens, most_critical_tokens)
        elif self.pmi_type == 'silhouette':
            best_num, best_score = self.get_silhouette_score(np.array(origin_critical_token_idxs))
            # best_num, best_score = self.get_silhouette_score(np.array(most_critical_token_idxs))            
            itc = best_score / best_num
        elif self.pmi_type == 'emb_dist':
            itc = -self.get_emb_topic_coherence(np.array(origin_critical_token_idxs))
            # itc = -self.get_emb_topic_coherence(np.array(most_critical_token_idxs))
        elif self.pmi_type == 'emb_cos':
            itc = self.get_emb_topic_coherence(np.array(origin_critical_token_idxs))
            # itc = self.get_emb_topic_coherence(np.array(most_critical_token_idxs))
        else:
            assert False, "PMI type not supported yet. please choose from: ['uci', 'umass', 'silhouette']."
        
        logger.info('Input Topic Coherence Metric ({}): {:.4f}'.format(self.pmi_type, itc))    
        if return_tokens:
            return itc, most_critical_tokens, origin_df
        else:
            return itc
        
            