import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import pprint
from utils import *
import time
from dataloader import *
from evaluator import *

class Concept_Extractor(nn.Module):
    """
    Extract concept.
    """
    def __init__(self, **kwargs):
        super().__init__()
        pass

    def extract_concepts(self, **kwargs):
        """
        Return: shaped [N, V], a set of concepts, where N is the number of concepts and V 
            is the dimension of the concept vector. 
        """
        pass
    
class CAV_Extractor(Concept_Extractor):
    """
    Extract concept.
    """
    def __init__(self, cfg):
        super().__init__()
        if  "pythia" in self.cfg['model_to_interpret']:
            ...
        elif "gpt" in self.cfg['model_to_interpret']:
            ...
        elif "llama" in self.cfg['model_to_interpret']:
            ...
        
    def forward(self, x):
        if  "pythia" in self.cfg['model_to_interpret']:
            ...
        elif "gpt" in self.cfg['model_to_interpret']:
            ...
        elif "llama" in self.cfg['model_to_interpret']:
            ...

    def save(self, save_dir, ckpt_name=None):
        ...

    @classmethod
    def load(cls, save_dir, ckpt_name=None):
        ...
    
    @classmethod
    def load_from_file(cls, path, device="cuda"):
        """
        Loads the saved autoencoder from file.
        """
        ...

    
    def extract_concepts(self, model, dataloader, save_dir):
        """
        Return: shaped [N, V], a set of concepts, where N is the number of concepts and V 
            is the dimension of the concept vector. 
        """
        if  "pythia" in self.cfg['model_to_interpret']:
            ...
        elif "gpt" in self.cfg['model_to_interpret']:
            ...
        elif "llama" in self.cfg['model_to_interpret']:
            ...
    
class AutoEncoder(Concept_Extractor):
    """
    Extract concept.
    """
    def __init__(self, cfg):
        super().__init__()
        if  "pythia" in cfg['model_to_interpret']:
            if cfg['site'] == 'mlp_post':
                d_out = cfg["d_mlp"]
            else:
                d_out = cfg["d_model"]
            d_hidden = cfg["dict_size"]
            dtype = torch.float32
            
            if cfg['init_type'] == 'xavier_uniform':
                self.W_enc = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(d_out, d_hidden, dtype=dtype)))
                self.W_dec = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(d_hidden, d_out, dtype=dtype)))
            if cfg['init_type'] == 'kaiming_uniform':
                self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_out, d_hidden, dtype=dtype)))
                self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_hidden, d_out, dtype=dtype)))
                
            self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
            self.b_dec = nn.Parameter(torch.zeros(d_out, dtype=dtype))

            self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
            
            self.cfg = cfg
            self.d_hidden = d_hidden
        
            
        elif "gpt" in self.cfg['model_to_interpret']:
            ...
        elif "llama" in self.cfg['model_to_interpret']:
            ...
        
        
        
    def forward(self, x):
        if  "pythia" in self.cfg['model_to_interpret']:
            x_cent = x - self.b_dec
            if self.cfg['tied_enc_dec'] == 1:
                acts = F.relu(x_cent @ self.W_dec.T + self.b_enc)
            else:
                acts = F.relu(x_cent @ self.W_enc + self.b_enc)
            x_reconstruct = acts @ self.W_dec + self.b_dec
            return x_reconstruct, acts
        elif "gpt" in self.cfg['model_to_interpret']:
            ...
        elif "llama" in self.cfg['model_to_interpret']:
            ...

    @torch.no_grad()
    def re_init(self, indices):
        if self.cfg['init_type'] == 'xavier_uniform':
            new_W_enc = (torch.nn.init.xavier_uniform_(torch.zeros_like(self.W_enc)))
            new_W_dec = (torch.nn.init.xavier_uniform_(torch.zeros_like(self.W_dec)))
        if self.cfg['init_type'] == 'kaiming_uniform':
            new_W_enc = (torch.nn.init.kaiming_uniform_(torch.zeros_like(self.W_enc)))
            new_W_dec = (torch.nn.init.kaiming_uniform_(torch.zeros_like(self.W_dec)))
        
        new_b_enc = (torch.zeros_like(self.b_enc))
        self.W_enc.data[:, indices] = new_W_enc[:, indices]
        self.W_dec.data[indices, :] = new_W_dec[indices, :]
        self.b_enc.data[indices] = new_b_enc[indices]

    @torch.no_grad()
    def remove_parallel_component_of_grads(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    def get_version(self, save_dir):
        return 1+max([int(file.name.split(".")[0]) for file in list(save_dir.iterdir()) if "pt" in str(file)])

    def save(self, save_dir, ckpt_name=None):
        if ckpt_name is None:
            torch.save(self.state_dict(), os.path.join(save_dir, "model.pt"))
            with open(os.path.join(save_dir, "cfg.json"), "w") as f:
                json.dump(self.cfg, f)
        else:
            torch.save(self.state_dict(), os.path.join(save_dir, ckpt_name + ".pt"))
            with open(os.path.join(save_dir, ckpt_name+".json"), "w") as f:
                json.dump(self.cfg, f)

    @classmethod
    def load(cls, save_dir, ckpt_name=None):
        if ckpt_name is None:
            cfg = (json.load(open(os.path.join(save_dir, "cfg.json"), "r")))
            pprint.pprint(cfg)
            self = cls(cfg=cfg)
            self.load_state_dict(torch.load( os.path.join(save_dir, "model.pt")))
        else:
            cfg = (json.load(open(os.path.join(save_dir, ckpt_name + ".json"), "r")))
            pprint.pprint(cfg)
            self = cls(cfg=cfg)
            self.load_state_dict(torch.load( os.path.join(save_dir, ckpt_name + ".pt")))
        return self
    
    @classmethod
    def load_cfg_only(cls, save_dir):
        cfg = (json.load(open(os.path.join(save_dir, "cfg.json"), "r")))
        pprint.pprint(cfg)
        self = cls(cfg=cfg)
        return self
    
    @classmethod
    def load_from_file(cls, path, device="cuda"):
        """
        Loads the saved autoencoder from file.
        """
        cfg = json.load(open(os.path.join(path, "cfg.json"), "r"))
        pprint.pprint(cfg)
        self = cls(cfg=cfg)
        state_dict = torch.load(os.path.join(path, "model.pt"), map_location=device)
        self.load_state_dict(state_dict)
        return self

    
    def extract_concepts(self, model, dataloader, save_dir):
        """
        Return: shaped [N, V], a set of concepts, where N is the number of concepts and V 
            is the dimension of the concept vector. 
        """
        if  "pythia" in self.cfg['model_to_interpret']:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg["lr"], betas=(self.cfg["beta1"], self.cfg["beta2"]))
            best_reconstruct = 0
            time_start=time.time()
            for epoch in range(self.cfg['epoch']):
                print('epoch:', epoch)
                if epoch > 0:
                    dataloader.reinit()
                for iter in range(dataloader.__len__()): 
                    activations = dataloader.next()
                    activations = activations.to(self.cfg["device"])
                    acti_reconstruct, mid_acts = self.forward(activations)
                    l2_loss =  (acti_reconstruct.float() - activations.float()).pow(2).sum(-1).mean()
                    l1_loss = self.cfg['l1_coeff'] * (mid_acts.float().abs().sum(-1).mean())
                    loss = l2_loss + l1_loss
                    loss.backward()
                    if self.cfg['remove_parallel']:
                        self.remove_parallel_component_of_grads()
                    optimizer.step()
                    optimizer.zero_grad()
                    loss_dict = {"loss": loss.item(), "l2_loss": l2_loss.item(), "l1_loss": l1_loss.item()}
                    
                    if (iter + 1) % 100 == 0:
                        time_end=time.time()
                        print('\nEpoch:', (epoch+1)," Iteration:", iter+1, " Total time:", time_end-time_start, "s")
                        print(f"Finished: {(100 * (iter+1) / self.cfg['num_batches']):.3f}", "% of Epoch ", epoch)
                        print(loss_dict)
                        x = AE_Evaluator.get_recons_loss(self, dataloader, model, self.cfg, num_batches=5)
                        l0 = AE_Evaluator.get_l0_norm(self, dataloader, self.cfg, num_batches=5)
                        
                        print("l0 norm:", l0)
                        
                        if x[0] > best_reconstruct:
                            best_reconstruct = x[0]
                            self.save(save_dir, ckpt_name="best_reconstruct")
                            
                    if (iter + 1) % 10000 == 0:
                        print('saved at:', save_dir)
                        self.save(save_dir, ckpt_name="Iteration" + str(iter+1) + "_Epoch" + str(epoch+1))
                        freqs, num_dead = AE_Evaluator.get_freqs(self, dataloader, self.cfg, num_batches=25)
                        if self.cfg['reinit'] == 1:
                            to_be_reset = (freqs<10**(-5.5))
                            self.re_init(self, to_be_reset)
                    del loss, acti_reconstruct, mid_acts, l2_loss, l1_loss
                self.save(save_dir, ckpt_name="Iteration" + str(iter) + "_Epoch" + str(epoch+1))
            return self.W_dec.detach()
        elif "gpt" in self.cfg['model_to_interpret']:
            ...
        elif "llama" in self.cfg['model_to_interpret']:
            ...
            
