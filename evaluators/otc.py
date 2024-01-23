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

class OutputTopicCoherenceEvaluator(nn.Module, BaseEvaluator):
    def __init__(self, cfg, activation_func, model):
        nn.Module.__init__(self)
        BaseEvaluator.__init__(self, cfg, activation_func, model)
        
    @classmethod
    def code(cls):
        return 'otc'
    
    def get_metric(self, eval_tokens, concept=None, concept_idx=None, pmi_type='uci'):
        _, most_preferred_tokens, topk_indices = self.get_preferred_predictions_of_concept(eval_tokens, concept)
        sentences = np.array(self.model.to_string(eval_tokens[:,1:]))
        inclusion = torch.tensor([[token in sentence.lower() for sentence in sentences] for token in most_preferred_tokens]).to(int)
        epsilon=1e-10
        corpus_len = sentences.shape[0]
        binary_inclusion = inclusion @ inclusion.T / corpus_len
        token_inclusion = inclusion.sum(-1) / corpus_len
        token_inclusion_mult = token_inclusion.unsqueeze(0).T @ token_inclusion.unsqueeze(0)
        if pmi_type == 'uci':  
            pmis = torch.log((binary_inclusion + epsilon) / token_inclusion_mult)
            mask = torch.triu(torch.ones_like(pmis),diagonal=1)
            final_pmi = (pmis * mask).sum() / mask.sum()
        elif pmi_type == 'umass':
            pmis = torch.log((binary_inclusion + epsilon) / token_inclusion)
            mask = torch.triu(torch.ones_like(pmis),diagonal=1)
            final_pmi = (pmis * mask).sum() / mask.sum()
        elif pmi_type == 'silhouette':
            best_num, best_score = self.get_silhouette_score(topk_indices)
            final_pmi = best_score / best_num
        else:
            assert False, "PMI type not supported yet. please choose from: ['uci', 'umass', 'silhouette']."
        logger.info('Output Topic Coherence Metric ({}): {:.4f}'.format(pmi_type, final_pmi))    
        return final_pmi
        
            