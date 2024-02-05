from .base import AbstractDataset
from transformers import AutoTokenizer
import os
import datasets


class ConceptXData(AbstractDataset):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.load_dataset()

    @classmethod
    def code(cls):
        return "conceptx_data"

    def load_dataset(self):
        os.system("pwd")
        with open("./data/text.in") as f:
            raw_texts = f.readlines()
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
        self.data = tokenizer(raw_texts)["input_ids"]
