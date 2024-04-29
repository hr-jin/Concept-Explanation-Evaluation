from .base import BaseExtractor
import torch
import torch.nn as nn
import torch.nn.functional as F

class Neuron(nn.Module, BaseExtractor):
    """
    An unsupervised learning method that hopes to learn sparsely activated concepts through AutoEncoder
    """
    def __init__(self, cfg, dataloader):
        super().__init__()
        d_hidden = cfg["dict_size"]
        if cfg['site'] == 'mlp_post':
            self.d_out = cfg["d_mlp"]
        else:
            self.d_out = cfg["d_model"]
        self.cfg = cfg
        self.d_hidden = d_hidden
        self.dataloader = dataloader
    
    @classmethod
    def code(cls):
        return 'neuron'
        
    def forward(self, x):
        x_cent = x - self.b_dec
        if self.cfg['tied_enc_dec'] == 1:
            acts = F.relu(x_cent @ self.W_dec.T + self.b_enc)
        else:
            acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        return x_reconstruct, acts
    
    def extract_concepts(self, model):
        self.concepts = torch.eye(self.d_out).to(self.cfg['device'])
    
    def get_concepts(self):
        return self.concepts
        
    @torch.no_grad()
    def activation_func(self, tokens, model, concept=None, concept_idx=None):    
        _, cache = model.run_with_cache(tokens, stop_at_layer=self.cfg["layer"]+1, names_filter=self.cfg["act_name"])
        hidden_states = cache[self.cfg["act_name"]]
    
        assert tokens.shape[1] == hidden_states.shape[1]
        
        if self.cfg['site'] == 'mlp_post':
            hidden_states = hidden_states.reshape(-1, self.cfg['d_mlp'])
        else: 
            hidden_states = hidden_states.reshape(-1, self.cfg['d_model'])
        
        if concept_idx == None:
            results = (hidden_states * concept).sum(-1) / (concept * concept).sum()
        else:
            results = (hidden_states * self.concepts[concept_idx, :]).sum(-1) / (concept * concept).sum()
        return results