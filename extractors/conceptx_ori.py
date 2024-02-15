from .base import BaseExtractor
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import torch.nn as nn
import torch
import pprint
import json

class ConceptXOri(nn.Module, BaseExtractor):
    def __init__(self, cfg, dataloader):
        super().__init__()
        self.cfg = cfg
        self.dataloader = dataloader
        self.k = cfg["clustering_k"]

    def forward(self):
        ...

    @classmethod
    def code(cls):
        return "conceptx_ori"

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

    def extract_concepts(self, model):
        points = self.dataloader.get_points()
        clustering = AgglomerativeClustering(n_clusters=self.k,compute_distances=True).fit(points)
        centroids = np.array([points[clustering.labels_ == j].mean(axis=0) for j in range(self.k)])
        self.concepts = torch.tensor(centroids, device=self.cfg["device"])
        torch.save(centroids, "./data/conceptx_concepts.pt")
        
    @classmethod
    def load_from_file(cls, dataloader, path=None, cfg=None):
        """
        """
        if cfg == None:
            cfg = json.load(open(path + ".json", "r"))
        if path == None:
            path = cfg['load_path']
        pprint.pprint(cfg)
        self = cls(cfg=cfg, dataloader=dataloader)
        self.concepts = torch.tensor(torch.load("./data/conceptx_concepts.pt")).to(cfg['device'])
        return self

    def get_concepts(self):
        return self.concepts
