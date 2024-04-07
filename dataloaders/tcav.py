from .base import AbstractDataloader
import torch

class TCAVDataloader(AbstractDataloader):
    
    def __init__(self, cfg, data, model):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.pos_data, self.neg_data, self.pos_labels, self.neg_labels = data

        
    def __len__(self):
        return 1
    
    @classmethod
    def code(cls):
        return 'tcav'
    
    def next(self):
        return self.pos_data, self.neg_data, self.pos_labels, self.neg_labels
    
    @torch.no_grad()
    def get_processed_random_batch(self):
        inputs = self.model.tokenizer(self.pos_data + self.neg_data, max_length=128, truncation=True, padding=True,return_tensors="pt")
        inputs = inputs['input_ids'].to('cpu')
        return inputs, inputs