from .base import BaseExtractor
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import torch.nn as nn



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

    def activation_func(self):
        ...

    def extract_concepts(self, model):
        points = self.dataloader.get_points()
        clustering = AgglomerativeClustering(n_clusters=self.k,compute_distances=True).fit(points)
        centroids = np.array([points[clustering.labels_ == j].mean(axis=0) for j in range(self.k)])
        self.concepts = centroids

    def get_concepts(self):
        return self.concepts
