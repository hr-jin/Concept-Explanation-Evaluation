from .base import AbstractDataloader

import torch

class ConceptXNaiveDataloader(AbstractDataloader):
    def __init__(self, cfg, data, model):
        super().__init__()
        self.cfg = cfg
        self.data = data

    @classmethod
    def code(cls):
        return "conceptx_ori"

    def get_points(self):
        return self.data

    def __len__(self):
        return len(self.data)
    
    def next(self):
        pass
