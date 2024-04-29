from .base import BaseEvaluator

import torch
import torch.nn as nn
from logger import logger
import numpy as np

class OutputTopicCoherenceEvaluator(nn.Module, BaseEvaluator):
    def __init__(
        self, 
        cfg, 
        activation_func,
        hidden_state_func,
        model, 
        concept=None, 
        concept_idx=None, 
        pmi_type='uci',
    ):
        nn.Module.__init__(self)
        BaseEvaluator.__init__(self, cfg, activation_func, hidden_state_func, model)
        self.concept = concept
        self.concept_idx = concept_idx
        self.pmi_type = pmi_type
        
    @classmethod
    def code(cls):
        return 'otc'
    
    def update_pmi_type(self, pmi_type):
        if pmi_type not in ['uci', 'umass', 'silhouette']:
            assert False, "PMI type not supported yet. please choose from: ['uci', 'umass', 'silhouette']."
        self.pmi_type = pmi_type
    
    def update_concept(self, concept=None, concept_idx=-1):
        self.concept = concept
        self.concept_idx = concept_idx
    
    def get_metric(self, eval_tokens, return_tokens=False, **kwargs):
        _, most_preferred_tokens, topk_indices = self.get_preferred_predictions_of_concept(eval_tokens, self.concept)
        topk_indices = topk_indices[most_preferred_tokens != '\ufffd']
        most_preferred_tokens = most_preferred_tokens[most_preferred_tokens != '\ufffd']
        sentences = np.array(self.model.to_string(eval_tokens[:,1:]))
        inclusion = torch.tensor([[token in sentence.lower() for sentence in sentences] for token in most_preferred_tokens]).to(int)
        logger.info('most_preferred_tokens:' + str(most_preferred_tokens))

        if self.pmi_type == 'silhouette':
            best_num, best_score = self.get_silhouette_score(np.array(topk_indices))
            otc = best_score
        elif self.pmi_type == 'emb_dist':
            otc = -self.get_emb_topic_coherence(np.array(topk_indices))
        elif self.pmi_type == 'emb_cos':
            otc = self.get_emb_topic_coherence(np.array(topk_indices))
        else:
            assert False, "PMI type not supported yet. please choose from: ['silhouette']."
        logger.info('Output Topic Coherence Metric ({}): {:.4f}'.format(self.pmi_type, otc))    
        if return_tokens:
            return otc, most_preferred_tokens
        else:
            return otc
        
            