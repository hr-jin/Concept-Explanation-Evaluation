from .base import BaseExtractor
from functools import partial
from collections import OrderedDict

import torch
from sklearn.metrics import accuracy_score

import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import json
import pprint
from logger import logger
import os
import time
import random
from sklearn.linear_model import LogisticRegression


class ConceptX(nn.Module, BaseExtractor):
    def __init__(self, cfg, dataloader):
        super().__init__()
        self.cfg = cfg
        self.dataloader = dataloader
            

    def forward(self):
        ...

    @classmethod
    def code(cls):
        return "conceptx"
    
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


    # @torch.no_grad()
    # def activation_func(self, tokens, model, concept=None, concept_idx=None):
    #     _, cache = model.run_with_cache(tokens, stop_at_layer=self.cfg["layer"]+1, names_filter=self.cfg["act_name"])
    #     hidden_states = cache[self.cfg["act_name"]]
    
    #     assert tokens.shape[1] == hidden_states.shape[1]
        
    #     if self.cfg['site'] == 'mlp_post':
    #         hidden_states = hidden_states.reshape(-1, self.cfg['d_mlp'])
    #     else: 
    #         hidden_states = hidden_states.reshape(-1, self.cfg['d_model'])
            
    #     if concept_idx == None:
    #         results = 1 / ((hidden_states - concept).square().sum(-1).sqrt() + 1e-8)
    #         # results = results * (results > 0.)
    #     else:
    #         results = 1 / ((hidden_states - self.concepts[concept_idx, :]).square().sum(-1).sqrt() + 1e-8)
    #         # results = results * (results > 0.)
    #     return results
    
    # @torch.no_grad()
    # def activation_func(self, tokens, model, concept=None, concept_idx=None):    
    #     _, cache = model.run_with_cache(tokens, stop_at_layer=self.cfg["layer"]+1, names_filter=self.cfg["act_name"])
    #     hidden_states = cache[self.cfg["act_name"]]
    
    #     assert tokens.shape[1] == hidden_states.shape[1]
        
    #     if self.cfg['site'] == 'mlp_post':
    #         hidden_states = hidden_states.reshape(-1, self.cfg['d_mlp'])
    #     else: 
    #         hidden_states = hidden_states.reshape(-1, self.cfg['d_model'])
            
    #     if concept_idx == None:
    #         results = torch.cosine_similarity(hidden_states, concept, dim=-1)
    #         # results = results * (results > 0.)
    #     else:
    #         results = torch.cosine_similarity(hidden_states, self.concepts[concept_idx, :], dim=-1)
    #         # results = results * (results > 0.)
    #     return results
    
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
        # print('results.shape:',results.shape)
        # print('hidden_states.shape:',hidden_states.shape)
        # print('results:',results)
        # print('hidden_states:',hidden_states)
        return results

    def extract_concepts(self, model):
        hidden_states = []
        token_num = 0
        for i in range(len(self.dataloader)):
            data = self.dataloader.get_pointer_batch().to(self.cfg["device"])
            self.dataloader.next()

            llm_outputs = model.run_with_cache(data)
            logits = llm_outputs[1][self.cfg['act_name']]  # (Batch, Token, Hidden)
            logits = logits.reshape(-1, 512)  # (B*T,H)

            if token_num + len(logits) > self.cfg["ConceptX_max_token"]:
                logits = logits[: self.cfg["ConceptX_max_token"] - token_num - len(logits)]
                hidden_states.append(logits)
                break
            token_num += len(logits)
            hidden_states.append(logits)

        hidden_states = torch.cat(hidden_states, dim=0).cpu().numpy()   #(all_token_num, 512)
        print('hidden_states.shape:',hidden_states.shape)
        clustering = AgglomerativeClustering(n_clusters=self.cfg["ConceptX_clusters"], compute_distances=True).fit(hidden_states)

        concepts = []
        
        print(clustering)
        
        labels = clustering.labels_
        print(labels)
        lengths = []
        clusters_data = []
        for i in range(self.cfg["ConceptX_clusters"]):
            indices = np.where(labels == i)[0]
            no_indices = np.where(labels != i)[0]
            concepts.append(np.mean(hidden_states[indices], axis=0))
        #     X = np.concatenate(
        #             [
        #                 hidden_states[indices],
        #                 hidden_states[random.sample(list(no_indices), len(indices))]
        #             ],
        #             axis=0,
        #         )
        #     Y = np.array(
        #             [1 for _ in range(len(indices))] +
        #             [0 for _ in range(len(indices))]
        #         )
        #     random_idx = random.sample([i for i in range(2*len(indices))], 2*len(indices))
        #     X = X[random_idx]
        #     Y = Y[random_idx]
        #     clusters_data.append([X, Y])
            
        # concepts = []
        # for i, (X, Y) in enumerate(clusters_data):
        #     print('cluster ' + str(i+1))
        #     classifier = LogisticRegression(solver='saga', max_iter=10000)
        #     classifier.fit(X, Y)
        #     predictions_train = classifier.predict(X)
        #     accuracy_train = accuracy_score(Y, predictions_train)
        #     print(Y, predictions_train)
        #     logger.info('Acc in training set: {:.2f}' .format(np.mean(accuracy_train)))
        #     cav = classifier.coef_[0]
        #     # print(cav)
        #     concepts.append(cav)
            
        concepts = np.array(concepts)   #(cluster_num, 512)
        # concept_torch = torch.tensor(concepts)
        # corr = [[i for i in range(concepts.shape[0])] for _ in range(concepts.shape[0])]
        # print(corr)
        # for i in range(concepts.shape[0]):
        #     print(i)
        #     for j in range(concepts.shape[0]):
        #         corr[i][j] = torch.cosine_similarity(concept_torch[i], concept_torch[j], dim=-1).item()
        # print(np.array(corr))
        self.concepts = torch.tensor(concepts, device=self.cfg["device"])
        torch.save(concepts, "./data/conceptx_concepts.pt")

    def get_concepts(self):
        return self.concepts
