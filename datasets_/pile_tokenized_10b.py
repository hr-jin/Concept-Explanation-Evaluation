from .base import AbstractDataset
import os
import datasets

class PileTokenized10BDataset(AbstractDataset):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.load_dataset()
            
    @classmethod
    def code(cls):
        return 'pile-tokenized-10b'

    def load_dataset(self):
        data_dir = self.cfg['data_dir']
        if not os.path.exists(os.path.join(data_dir, 'pile-tokenized-10b.hf')):
            data = datasets.load_dataset('pile-tokenized-10b', split="train", cache_dir=data_dir)
            data.save_to_disk(os.path.join(data_dir, 'pile-tokenized-10b.hf'))
        data = datasets.load_from_disk(os.path.join(data_dir, 'pile-tokenized-10b.hf'))
        self.data = data
        


