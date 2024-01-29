from .base import AbstractDataloader

import torch
from logger import logger
import numpy as np
import time

def count_occurrences(input_tensor):
    logger.info("counting occurrences...")
    # print(input_tensor.shape)
    start_time = time.time()
    
    # Flatten the input tensor
    flat_tensor = input_tensor.flatten()

    # Get unique elements and their counts
    unique_elements, counts = torch.unique(flat_tensor, return_counts=True)

    # Create a dictionary to map unique elements to their counts
    counts_dict = dict(zip(unique_elements.tolist(), counts.tolist()))
    for w in counts_dict:
        counts_dict[w] = np.sqrt(counts_dict[w])
        
    end_time = time.time()
    logger.info("counting occurrences takes {} seconds".format(end_time - start_time))

    return counts_dict

class ConvexOptimDataloader(AbstractDataloader):
    """
    This is a dataloader used to generate minibatch data for concept extraction or concept evaluation.
    There's a built-in buffer to scramble the activation vector, which may be helpful for training Autoencoder.
    Add word frequency targeted for convex optimation baseline
    """
    def __init__(self, cfg, data, model):
        super().__init__()
        self.cfg = cfg
        self.buffer = torch.zeros((cfg["buffer_size"], cfg["act_size"]), dtype=torch.float16, requires_grad=False)
        self.freq_buffer = torch.zeros((cfg["buffer_size"]), dtype=torch.float16, requires_grad=False)
        self.cfg = cfg
        self.token_pointer = 0
        self.data = data
        
        occurence_tensor = torch.tensor(data[:cfg["freq_sample_range"]]['tokens'])
        occurence_tensor = occurence_tensor[:, :self.cfg['seq_len']]
        self.counts_dict = count_occurrences(occurence_tensor)
        del occurence_tensor
        
        self.model = model
        self.tokenizer = model.tokenizer
        self.empty_flag = 0
        self.refresh()
        
        
    def __len__(self):
        return self.cfg['num_batches']
    
    @classmethod
    def code(cls):
        return 'convex_optim'

    @torch.no_grad()
    def next(self):
        out = self.buffer[self.pointer:self.pointer+self.cfg["batch_size"]]
        out_freq = self.freq_buffer[self.pointer:self.pointer+self.cfg["batch_size"]]
        self.pointer += self.cfg["batch_size"]
        if self.pointer > self.buffer.shape[0] - self.cfg["batch_size"]:
            self.refresh()
        return out, out_freq
    
    def reinit(self):
        self.token_pointer = 0
        self.empty_flag = 0
        self.refresh()
        
    def refresh(self):
        logger.info("buffer refreshing...\n")
        self.pointer = 0
        with torch.autocast("cuda", torch.float16):
            num_batches = self.cfg["buffer_batches"]
            with torch.no_grad():
                for _ in range(0, num_batches, self.cfg["model_batch_size"]):
                    if self.token_pointer+self.cfg["model_batch_size"] <= len(self.data):
                        if self.cfg['tokenized']:
                            tokens = self.data[self.token_pointer:self.token_pointer+self.cfg["model_batch_size"]]['tokens']
                            tokens = torch.tensor(tokens)
                        else:    
                            sentences = self.data[self.token_pointer:self.token_pointer+self.cfg["model_batch_size"]]
                            inputs = self.tokenizer(sentences, max_length=128, truncation=True, padding=True,return_tensors="pt")
                            inputs = inputs.to('cpu')
                            tokens = inputs['input_ids']
                        tokens = tokens[:, :self.cfg['seq_len']]
                        freq_tensor = torch.tensor([[self.counts_dict[element.item()] if element.item() in self.counts_dict else 1 for element in row] for row in tokens])
                        freq_tensor = freq_tensor.reshape(-1)
                        
                        tokens[:, 0] = self.model.tokenizer.bos_token_id
                        _, cache = self.model.run_with_cache(tokens, names_filter=self.cfg["act_name"], remove_batch_dim=False)
                        acts = cache[self.cfg["act_name"]].reshape(-1, self.cfg["act_size"])
                        self.buffer[self.pointer: self.pointer+acts.shape[0]] = acts.cpu()
                        self.freq_buffer[self.pointer: self.pointer+acts.shape[0]] = freq_tensor.cpu()
                        self.pointer += acts.shape[0]
                        self.token_pointer += self.cfg["model_batch_size"]
                    else:
                        self.empty_flag = 1
        self.pointer = 0
        new_order = torch.randperm(self.buffer.shape[0])
        self.buffer = self.buffer[new_order]
        self.freq_buffer = self.freq_buffer[new_order]
        
    @torch.no_grad()
    def get_random_batch(self):
        if self.cfg['tokenized']:
            tokens = self.data[torch.randperm(len(self.data))[:self.cfg['model_batch_size']]]['tokens']
            tokens = torch.tensor(tokens)
        else:    
            sentences = self.data[torch.randperm(len(self.data))[:self.cfg['model_batch_size']]]
            inputs = self.tokenizer(sentences, max_length=128, truncation=True, padding=True,return_tensors="pt")
            inputs = inputs.to('cpu')
            tokens = inputs['input_ids']
        
        tokens = tokens[:, :self.cfg['seq_len']]
        return tokens
    
    @torch.no_grad()
    def get_processed_random_batch(self):
        tokens = self.get_random_batch()
        tokens = tokens[:, :self.cfg['seq_len']]
        tokens[:, 0] = self.model.tokenizer.bos_token_id
        freq_tensor = torch.tensor([[self.counts_dict[element.item()] if element.item() in self.counts_dict else 1 for element in row] for row in tokens])
        return tokens, freq_tensor
    
    @torch.no_grad()
    def get_batch(self):
        if self.cfg['tokenized']:
            tokens = self.data[:self.cfg['model_batch_size']]['tokens']
            tokens = torch.tensor(tokens)
        else:    
            sentences = self.data[:self.cfg['model_batch_size']]
            inputs = self.tokenizer(sentences, max_length=128, truncation=True, padding=True,return_tensors="pt")
            inputs = inputs.to('cpu')
            tokens = inputs['input_ids']
        return tokens
    
    @torch.no_grad()
    def get_processed_batch(self):
        tokens = self.get_batch()
        tokens = tokens[:, :self.cfg['seq_len']]
        tokens[:, 0] = self.model.tokenizer.bos_token_id
        freq_tensor = torch.tensor([[self.counts_dict[element.item()] if element.item() in self.counts_dict else 1 for element in row] for row in tokens])
        return tokens, freq_tensor