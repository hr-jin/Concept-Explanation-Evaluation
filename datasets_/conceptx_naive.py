from .base import AbstractDataset
import numpy as np
import os

class ConceptXNaiveDataset(AbstractDataset):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.load_dataset()
            
    @classmethod
    def code(cls):
        return 'conceptx_naive'

    def load_dataset(self):
        data_dir = self.cfg['data_dir']
        data = np.load(os.path.join(data_dir, "clustering_points-layer3.npy"))
        self.data = data
        


