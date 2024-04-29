from .base import AbstractDataset
import os
import pandas as pd
import numpy as np

class HarmFulQADataset(AbstractDataset):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.load_dataset()
            
    @classmethod
    def code(cls):
        return 'HarmfulQA'

    def load_dataset(self):
        data_dir = self.cfg['data_dir']
        df = pd.read_csv(os.path.join(data_dir, 'advbench+harmfulqa.csv'))
        pos_data = df['question_1'].tolist()
        neg_data = df['question_2'].tolist()   
        pos_labels = np.ones((len(pos_data), ))  
        neg_labels = np.zeros((len(neg_data),))      
        self.data = (pos_data, neg_data, pos_labels, neg_labels)


