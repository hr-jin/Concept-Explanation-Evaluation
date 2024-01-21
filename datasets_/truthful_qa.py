from .base import AbstractDataset
import os
import datasets
import pandas as pd

class TrustfulQADataset(AbstractDataset):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.load_dataset()
            
    @classmethod
    def code(cls):
        return 'TrustfulQA'

    def load_dataset(self):
        data_dir = self.cfg['data_dir']
        df = pd.read_csv(os.path.join(data_dir, 'TrustfulQA.csv'))
        self.data = df['Question'].tolist()


