from .base import AbstractDataset
import os
import datasets
import numpy as np

class PileDataset(AbstractDataset):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.load_dataset()
            
    @classmethod
    def code(cls):
        return 'pile'

    def load_dataset(self):
        data_dir = self.cfg['data_dir']
        if not os.path.exists(os.path.join(data_dir, 'pile.hf')):
            data = datasets.load_dataset('pile', split="train", cache_dir=data_dir)
            data.save_to_disk(os.path.join(data_dir, 'pile.hf'))
        if self.cfg['model_to_interpret']=="llama-7b-chat":
            data = datasets.load_from_disk(os.path.join(data_dir, 'pile-by-llama2.hf'))
        elif self.cfg['model_to_interpret']=="gpt2-small":
            data = datasets.load_from_disk(os.path.join(data_dir, 'pile-by-gpt2.hf'))
        elif self.cfg['model_to_interpret']=="pythia-70m":
            data = datasets.load_from_disk(os.path.join(data_dir, 'pile.hf'))
        else:
            data = None
        print('seqlen of dataset:', np.array(data[0]['tokens']).shape)
        self.data = data
        


