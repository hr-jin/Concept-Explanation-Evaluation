from .base import BaseExtractor
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import pprint
from logger import logger
import os
import time

class AutoEncoder(nn.Module, BaseExtractor):
    """
    An unsupervised learning method that hopes to learn sparsely activated concepts through AutoEncoder
    """
    def __init__(self, cfg, dataloader):
        super().__init__()
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
        if cfg['use_bias_d'] == 0:
            self.b_dec = nn.Parameter(torch.zeros(d_out, dtype=dtype), requires_grad=False) 
        else:
            self.b_dec = nn.Parameter(torch.zeros(d_out, dtype=dtype))
        
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.cfg = cfg
        self.d_hidden = d_hidden
        self.dataloader = dataloader
    
    @classmethod
    def code(cls):
        return 'ae'
        
    def forward(self, x):
        x_cent = x - self.b_dec
        if self.cfg['tied_enc_dec'] == 1:
            acts = F.relu(x_cent @ self.W_dec.T + self.b_enc)
        else:
            acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        return x_reconstruct, acts
    
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

    def get_version(self):
        return 1+max([int(file.name.split(".")[0]) for file in list(self.cfg['save_dir'].iterdir()) if "pt" in str(file)])

    def save(self, ckpt_name=None):
        if ckpt_name is None:
            torch.save(self.state_dict(), os.path.join(self.cfg['save_dir'], "model.pt"))
            with open(os.path.join(self.cfg['save_dir'], "cfg.json"), "w") as f:
                json.dump(self.cfg, f)
        else:
            torch.save(self.state_dict(), os.path.join(self.cfg['save_dir'], ckpt_name + ".pt"))
            with open(os.path.join(self.cfg['save_dir'], ckpt_name+".json"), "w") as f:
                json.dump(self.cfg, f)
    
    @classmethod
    def load_from_file(cls, dataloader, path=None, cfg=None):
        """
        Loads the saved autoencoder from file.
        """
        if cfg == None:
            cfg = json.load(open(path + ".json", "r"))
        if path == None:
            path = cfg['load_path']
        pprint.pprint(cfg)
        self = cls(cfg=cfg, dataloader=dataloader)
        state_dict = torch.load(path + ".pt")
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'model' not in k:
                new_state_dict[k] = v 
        self.load_state_dict(new_state_dict)
        return self

    def extract_concepts(self, model):
        """
        Returns:
            A dictionary with the shape [d_hidden, d_out], each line contains a concept
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg["lr"], betas=(self.cfg["beta1"], self.cfg["beta2"]))
        best_reconstruct = 0
        time_start=time.time()
        for epoch in range(self.cfg['epoch']):
            logger.info('epoch:{}'.format(epoch + 1))
            logger.info('total iterations:{}'.format(self.dataloader.__len__()))
            logger.info('\n')
            if epoch > 0:
                self.dataloader.reinit()
            for iter in range(self.dataloader.__len__()): 
                activations = self.dataloader.next() 
                if self.dataloader.empty_flag == 1:
                    logger.info('All training data in dataloader has been passed through.')
                    break
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
                loss_dict = {"AE_loss": loss.item(), "l2_loss": l2_loss.item(), "l1_loss": l1_loss.item()}
                
                if (iter + 1) % self.cfg['val_freq'] == 0:
                    time_end=time.time()
                    logger.info('Epoch: {} Iteration: {} Total time: {:.4f}s'.format(epoch+1, iter+1, time_end-time_start))
                    logger.info("Finished: {:.3f} % of Epoch {}".format((100 * (iter+1) / self.cfg['num_batches']), epoch + 1))
                    logger.info(" ".join(["{}: {:.4f}".format(metric_name, metric_val) for metric_name, metric_val in loss_dict.items()]))
                    x = self.get_recons_loss(self.dataloader, model, self.cfg, num_batches=5)
                    l0 = self.get_l0_norm(self.dataloader, self.cfg, num_batches=5)
                    logger.info("l0 norm: {:.4f}".format(l0))
                    if x[0] > best_reconstruct:
                        best_reconstruct = x[0]
                        self.save(ckpt_name="best_reconstruct")
                    print('\n')
                        
                if (iter + 1) % 10000 == 0:
                    logger.info('saved at:' + self.cfg['save_dir'])
                    self.save(ckpt_name="Iteration" + str(iter+1) + "_Epoch" + str(epoch+1))
                    freqs, num_dead = self.get_freqs(self.dataloader, self.cfg, num_batches=25)
                    if self.cfg['reinit'] == 1:
                        to_be_reset = (freqs<10**(-5.5))
                        self.re_init(self, to_be_reset)
                del loss, acti_reconstruct, mid_acts, l2_loss, l1_loss
            self.save(ckpt_name="Iteration" + str(iter) + "_Epoch" + str(epoch+1))
        self.concepts = self.W_dec.clone().detach()
    
    def get_concepts(self):
        self.concepts = self.W_dec.clone().detach()
        return self.concepts
    
    @staticmethod
    def replacement_hook(mlp_post, hook, encoder):
        mlp_post_reconstr = encoder(mlp_post)[0]
        return mlp_post_reconstr

    @staticmethod
    def zero_ablate_hook(mlp_post, hook):
        mlp_post[:] = 0.
        return mlp_post

    def get_freqs(self, dataloader, cfg, num_batches=25):
        with torch.no_grad():
            act_freq_scores = torch.zeros(self.d_hidden, dtype=torch.float32).to(cfg['device'])
            total = 0
            for i in range(num_batches):
                hidden_states = dataloader.buffer[torch.randperm(dataloader.buffer.shape[0])[:cfg["batch_size"]]]
                hidden_states = hidden_states.to(cfg['device'])
                _, acts = self(hidden_states)
                act_freq_scores += (acts > 0).sum(0)
                total+=acts.shape[0]
            act_freq_scores /= total
            num_dead = (act_freq_scores==0).float().mean()
            return act_freq_scores, num_dead

    def get_recons_loss(self, dataloader, model, cfg, num_batches=5):
        with torch.no_grad():
            loss_list = []
            for i in range(num_batches):
                tokens, _ = dataloader.get_processed_random_batch()
                loss = model(tokens, return_type="loss")
                recons_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(cfg["act_name"], partial(self.replacement_hook, encoder=self))])
                zero_abl_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(cfg["act_name"], self.zero_ablate_hook)])
                loss_list.append((loss, recons_loss, zero_abl_loss))
            losses = torch.tensor(loss_list)
            loss, recons_loss, zero_abl_loss = losses.mean(0).tolist()
            logger.info(f"loss: {loss:.4f} recons_loss: {recons_loss:.4f} zero_abl_loss: {zero_abl_loss:.4f}")
            score = ((zero_abl_loss - recons_loss)/(zero_abl_loss - loss))
            logger.info(f"Reconstruction Score: {score:.2%}")
            return score, loss, recons_loss, zero_abl_loss

    def get_l0_norm(self, dataloader, cfg, num_batches=5):
        with torch.no_grad():
            num_feature = self.d_hidden
            l0_norms = []
            for i in range(num_batches):
                hidden_states = dataloader.buffer[torch.randperm(dataloader.buffer.shape[0])[:cfg["batch_size"]]]
                hidden_states = hidden_states.to(cfg['device'])
                _, acts = self(hidden_states)
                acts = acts.reshape(-1, num_feature)
                l0_norm = torch.linalg.norm(acts, ord=0, dim=-1).sum() / acts.shape[0]
                l0_norms.append(l0_norm)
            l0_norm = sum(l0_norms) / len(l0_norms)
            return l0_norm
        
    # @torch.no_grad()
    # def activation_func(self, tokens, model, concept=None, concept_idx=None):
    #     _, cache = model.run_with_cache(tokens, stop_at_layer=self.cfg["layer"]+1, names_filter=self.cfg["act_name"])
    #     hidden_states = cache[self.cfg["act_name"]]
    
        
    #     assert tokens.shape[1] == hidden_states.shape[1]
        
    #     if self.cfg['site'] == 'mlp_post':
    #         hidden_states = hidden_states.reshape(-1, self.cfg['d_mlp'])
    #     else: 
    #         hidden_states = hidden_states.reshape(-1, self.cfg['d_model'])
            
    #     if concept_idx == None:
    #         results = self.forward(hidden_states)[1]
    #     else:
    #         hidden_acts = self.forward(hidden_states)[1]
    #         results = hidden_acts[:, concept_idx]
    #     return results
    
    @torch.no_grad()
    def activation_func(self, tokens, model, concept=None, concept_idx=None):    
        _, cache = model.run_with_cache(tokens, stop_at_layer=self.cfg["layer"]+1, names_filter=self.cfg["act_name"])
        hidden_states = cache[self.cfg["act_name"]]
    
        assert tokens.shape[1] == hidden_states.shape[1]
        
        if self.cfg['site'] == 'mlp_post':
            hidden_states = hidden_states.reshape(-1, self.cfg['d_mlp'])
        else: 
            hidden_states = hidden_states.reshape(-1, self.cfg['d_model'])
            
        if concept_idx == None:
            results = torch.cosine_similarity(hidden_states, concept, dim=-1)
            # results = results * (results > 0.)
        else:
            results = torch.cosine_similarity(hidden_states, self.concepts[concept_idx, :], dim=-1)
            # results = results * (results > 0.)
        return results