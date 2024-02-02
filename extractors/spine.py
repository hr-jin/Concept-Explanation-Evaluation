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
from sklearn.datasets import make_blobs
import numpy as np
from tqdm import tqdm


class SPINEModel(nn.Module):
    
    def __init__(self, params):
        super(SPINEModel, self).__init__()
		
		# params
        self.input_dim = params['act_size']
        self.hidden_dim = params['dict_size']
        self.noise_level = params['noise_level']
        self.getReconstructionLoss = nn.MSELoss()
        self.rho_star = 1.0 - params['sparsity']

        # autoencoder
        logger.info("Building model ")
        self.linear1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.input_dim)


    def forward(self, batch_x, batch_y):
		
        # forward
        # batch: [b, d]
        batch_size = batch_x.size(0)
        linear1_out = self.linear1(batch_x)
        h = linear1_out.clamp(min=0, max=1) # capped relu
        out = self.linear2(h)

        # different terms of the loss
        reconstruction_loss = self.getReconstructionLoss(out, batch_y) # reconstruction loss
        psl_loss = self._getPSLLoss(h, batch_size) 		# partial sparsity loss
        asl_loss = self._getASLLoss(h)    	# average sparsity loss
        total_loss = reconstruction_loss + psl_loss + asl_loss
        
        return out, h, total_loss, [reconstruction_loss,psl_loss, asl_loss]


    def _getPSLLoss(self,h, batch_size):
        return torch.sum(h*(1-h))/ (batch_size * self.hidden_dim)


    def _getASLLoss(self, h):
        temp = torch.mean(h, dim=0) - self.rho_star
        temp = temp.clamp(min=0)
        return torch.sum(temp * temp) / self.hidden_dim
   
   
def get_noise_features(n_samples, n_features, noise_amount):
	noise_x,  _ =  make_blobs(n_samples=n_samples, n_features=n_features, 
			cluster_std=noise_amount,
			centers=np.array([np.zeros(n_features)]))
	return torch.from_numpy(noise_x).type(torch.FloatTensor)
 
    
class SpineExtractor(nn.Module, BaseExtractor):
    """
    paper: SPINE: SParse Interpretable Neural Embeddings
    """
    def __init__(self, cfg, dataloader):
        super(SpineExtractor, self).__init__()
        self.cfg = cfg
        self.dataloader = dataloader
        self.model = SPINEModel(cfg)
        self.is_trained = False
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg["lr"], betas=(self.cfg["beta1"], self.cfg["beta2"]))
        
    @classmethod
    def code(cls):
        return "spine"
    
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
        Loads the saved extractor from file.
        """
        # if ckpt_name is None:
        #     cfg = json.load(open(os.path.join(save_dir, "cfg.json"), "r"))
        #     state_dict = torch.load(os.path.join(save_dir, "model.pt"))
        # else:
        #     cfg = json.load(open(os.path.join(save_dir, ckpt_name+".json"), "r"))
        #     state_dict = torch.load(os.path.join(save_dir, ckpt_name+".pt"))
        # self = cls(cfg, dataloader)
        # self.load_state_dict(state_dict)
        if cfg == None:
            cfg = json.load(open(path + ".json", "r"))
        if path == None:
            path = cfg['load_path']
        pprint.pprint(cfg)
        self = cls(cfg=cfg, dataloader=dataloader)
        state_dict = torch.load(path + ".pt")
        self.load_state_dict(state_dict)
        return self
        return self
    
    def train(self, explained_model):
        log_freq = self.cfg['val_freq']
        best_recons = 0
        for epoch in range(self.cfg["epoch"]):
            logger.info('epoch:{}'.format(epoch + 1))
            logger.info('total iterations:{}'.format(self.dataloader.__len__()))
            logger.info('\n')
            
            loss_dict = {
                "total_loss": [],
                "recons_loss": [],
                "psl_loss": [],
                "asl_loss": [],
            }
            for iter in tqdm(range(len(self.dataloader))):
                # get batch
                activations = self.dataloader.next()
                if self.dataloader.empty_flag == 1:
                    logger.info('All training data in dataloader has been passed through.')
                    break
                activations = activations.type(torch.FloatTensor).to(self.cfg["device"])
                # add noise
                noised_acts = activations + get_noise_features(activations.shape[0], activations.shape[1], self.cfg['noise_level']).to(self.cfg["device"])
                
                # forward
                out, h, loss, loss_terms = self.model(noised_acts, activations)
                recons_loss, psl_loss, asl_loss = loss_terms
                
                # optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # log outputs
                loss_dict["total_loss"].append(loss.item())
                loss_dict["asl_loss"].append(asl_loss.item())
                loss_dict["psl_loss"].append(psl_loss.item())
                loss_dict["recons_loss"].append(recons_loss.item())
                
                # log outputs and save best model
                if (iter+1) % log_freq == 0:
                    logger.info('iter:{}'.format(iter + 1))
                    logger.info('total_loss:{:.4f}'.format(sum(loss_dict["total_loss"]) / log_freq))
                    logger.info('recons_loss:{:.4f}'.format(sum(loss_dict["recons_loss"]) / log_freq))
                    logger.info('psl_loss:{:.4f}'.format(sum(loss_dict["psl_loss"]) / log_freq))
                    logger.info('asl_loss:{:.4f}'.format(sum(loss_dict["asl_loss"]) / log_freq))
                    loss_dict = {
                        "total_loss": [],
                        "recons_loss": [],
                        "psl_loss": [],
                        "asl_loss": [],
                    }
                    
                    recons_score, _, _, _ = get_recons_loss(self.dataloader, explained_model, self.cfg, self.model, num_batches=5)
                    if recons_score > best_recons:
                        best_recons = recons_score
                        logger.info(f"Saving best model with reconstruction score: {recons_score:.2%}")
                        self.save('spine_best_reconstruct')
        self.is_trained = True               
    
    def extract_concepts(self, model):
        if self.is_trained:
            return
        self.train(model)
    
    
    def get_concepts(self):                   
        return self.model.linear2.weight.T.clone().detach()   

    
    @torch.no_grad()
    def activation_func(self, x, concept_idx):
        out, h, loss, loss_terms = self.model(x, x)
        return h[:, concept_idx]


def replacement_hook(mlp_post, hook, encoder):
    mlp_post_reconstr = encoder(mlp_post, mlp_post)[0]
    return mlp_post_reconstr

def zero_ablate_hook(mlp_post, hook):
    mlp_post[:] = 0.
    return mlp_post


def get_recons_loss(dataloader, model, cfg, encoder, num_batches=5):
    with torch.no_grad():
        loss_list = []
        for i in range(num_batches):
            tokens = dataloader.get_processed_random_batch()
            loss = model(tokens, return_type="loss")
            recons_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(cfg["act_name"], partial(replacement_hook, encoder=encoder))])
            zero_abl_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(cfg["act_name"], zero_ablate_hook)])
            loss_list.append((loss, recons_loss, zero_abl_loss))
        losses = torch.tensor(loss_list)
        loss, recons_loss, zero_abl_loss = losses.mean(0).tolist()
        logger.info(f"loss: {loss:.4f} recons_loss: {recons_loss:.4f} zero_abl_loss: {zero_abl_loss:.4f}")
        score = ((zero_abl_loss - recons_loss)/(zero_abl_loss - loss))
        logger.info(f"Reconstruction Score: {score:.2%}")
        return score, loss, recons_loss, zero_abl_loss