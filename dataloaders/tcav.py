from .base import AbstractDataloader

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