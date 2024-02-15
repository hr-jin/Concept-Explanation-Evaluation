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
        if self.cfg['model_to_interpret']=="llama-2-7b-chat":
            data = datasets.load_from_disk(os.path.join(data_dir, 'pile-tokenized-10b-by-llama2.hf'))
        elif self.cfg['model_to_interpret']=="gpt2-small":
            data = datasets.load_from_disk(os.path.join(data_dir, 'pile-tokenized-10b-by-gpt2.hf'))
        else:
            data = datasets.load_from_disk(os.path.join(data_dir, 'pile-tokenized-10b.hf'))
        self.data = data
        


