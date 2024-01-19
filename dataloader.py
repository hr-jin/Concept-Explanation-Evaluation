import torch
from logger import logger

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

    def next(self):
        """
        Returns a batch of data.
        """
        pass
    
class TCAV_Dataloader(Dataloader_base):
    
    def __init__(self, cfg, positive_examples, negative_examples):
        super(TCAV_Dataloader, self).__init__()
        self.cfg = cfg
        self.positive_examples = positive_examples
        self.negative_examples = negative_examples
        
    def __len__(self):
        return 1
    
    def next(self):
        return self.positive_examples, self.negative_examples
    
class AE_Dataloader(Dataloader_base):
    """
    This is a dataloader used to generate minibatch data for concept extraction or concept evaluation.
    There's a built-in buffer to scramble the activation vector, which may be helpful for training Autoencoder.
    """
    def __init__(self, cfg, data, model):
        super(AE_Dataloader, self).__init__()
        self.cfg = cfg
        self.buffer = torch.zeros((cfg["buffer_size"], cfg["act_size"]), dtype=torch.bfloat16, requires_grad=False)
        self.cfg = cfg
        self.token_pointer = 0
        self.data = data
        self.model = model
        self.tokenizer = model.tokenizer
        self.refresh()
        
    def __len__(self):
        return self.cfg['num_batches']
        
    def refresh(self):
        logger.info("buffer refreshing...\n")
        self.pointer = 0
        with torch.autocast("cuda", torch.float16):
            num_batches = self.cfg["buffer_batches"]
            with torch.no_grad():
                for _ in range(0, num_batches, self.cfg["model_batch_size"]):
                    if self.cfg['tokenized']:
                        tokens = self.data[self.token_pointer:self.token_pointer+self.cfg["model_batch_size"]]['tokens']
                        tokens = torch.tensor(tokens)
                    else:    
                        sentences = self.data[self.token_pointer:self.token_pointer+self.cfg["model_batch_size"]]
                        inputs = self.tokenizer(sentences, max_length=128, truncation=True, padding=True,return_tensors="pt")
                        inputs = inputs.to('cpu')
                        tokens = inputs['input_ids']
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
    def get_random_batch(self):
        if self.cfg['tokenized']:
            tokens = self.data[torch.randperm(len(self.data))[:self.cfg['model_batch_size']]]['tokens']
            tokens = torch.tensor(tokens)
        else:    
            sentences = self.data[torch.randperm(len(self.data))[:self.cfg['model_batch_size']]]
            inputs = self.tokenizer(sentences, max_length=128, truncation=True, padding=True,return_tensors="pt")
            inputs = inputs.to('cpu')
            tokens = inputs['input_ids']
        return tokens

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
        
