from .base import AbstractDataloader

import torch
from logger import logger


class ConceptXDataloader(AbstractDataloader):
    def __init__(self, cfg, data, model):
        super().__init__()
        self.cfg = cfg
        self.data = data
        self.model = model
        self.tokenizer = model.tokenizer
        self.token_pointer = 0

    @classmethod
    def code(cls):
        return "conceptx"

    def next(self):
        out = torch.tensor(self.data[self.token_pointer], device=self.cfg["device"]).unsqueeze(0)
        self.token_pointer += 1
        return out

    def get_pointer_batch(self):
        out = torch.tensor(self.data[self.token_pointer], device=self.cfg["device"]).unsqueeze(0)
        return out

    def __len__(self):
        return len(self.data)
