import os
import random
import json
import gzip
import numpy as np
import torch
import scipy.sparse as sp
import nltk
from torch.utils.data import Dataset
from torch.nn import functional as F


class Dataloader_base():
    """
    This is a dataloader used to generate minibatch data for concept extraction or concept evaluation.
    """
    def __init__(self, **kwargs):
        pass
    
    def __len__(self, **kwargs):
        """
        Returns the number of batches in an epoch.
        """
        pass

    @torch.no_grad()
    def next(self):
        """
        Returns a batch of data.
        """
        pass
    
class AE_Dataloader(Dataloader_base):
    """
    This is a dataloader used to generate minibatch data for concept extraction or concept evaluation.
    """
    def __init__(self, cfg, data, model):
        super(AE_Dataloader, self).__init__()
        self.buffer = torch.zeros((cfg["buffer_size"], cfg["act_size"]), dtype=torch.bfloat16, requires_grad=False)
        self.cfg = cfg
        self.token_pointer = 0
        self.data = data
        self.model = model
        self.refresh()
        
    def __len__(self):
        return self.cfg['num_batches']
        
    def refresh(self):
        print("\nbuffer refreshing...")
        self.pointer = 0
        with torch.autocast("cuda", torch.float16):
            num_batches = self.cfg["buffer_batches"]
            with torch.no_grad():
                for _ in range(0, num_batches, self.cfg["model_batch_size"]):
                    tokens = self.data[self.token_pointer:self.token_pointer+self.cfg["model_batch_size"]]['tokens']
                    tokens = torch.tensor(tokens)
                    tokens = tokens[:, :self.cfg['seq_len']]
                    tokens[:, 0] = self.model.tokenizer.bos_token_id
                    _, cache = self.model.run_with_cache(tokens, names_filter=self.cfg["act_name"], remove_batch_dim=False)
                    acts = cache[self.cfg["act_name"]].reshape(-1, self.cfg["act_size"])

                    self.buffer[self.pointer: self.pointer+acts.shape[0]] = acts.cpu()
                    self.pointer += acts.shape[0]
                    self.token_pointer += self.cfg["model_batch_size"]

        self.pointer = 0
        self.buffer = self.buffer[torch.randperm(self.buffer.shape[0])]

    @torch.no_grad()
    def next(self):
        out = self.buffer[self.pointer:self.pointer+self.cfg["batch_size"]]
        self.pointer += self.cfg["batch_size"]
        if self.pointer > self.buffer.shape[0] - self.cfg["batch_size"]:
            self.refresh()
        return out
    
    def reinit(self):
        self.token_pointer = 0
        self.refresh()
        
    