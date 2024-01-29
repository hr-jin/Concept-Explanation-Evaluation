from .base import BaseExtractor
from functools import partial
import scipy as cp

import torch
import torch.nn as nn
import json
from logger import logger
import os
import numpy as np
from tqdm import tqdm


class ConvexOptimModel(nn.Module):
    
    def __init__(self, params):
        super(ConvexOptimModel, self).__init__()
		
		# params
        self.input_dim = params['act_size']
        self.hidden_dim = params['dict_size']
        self.device = params['device']
        self.reg = params["reg"]
        # The regularization factor for sparse coding. You should use the same one you used in inference 

        # autoencoder
        logger.info("Building model ")
        self.PHI_SIZE = [self.input_dim, self.hidden_dim]
        self.PHI = torch.randn(self.PHI_SIZE).to(self.device)
        self.PHI = self.PHI.div_(self.PHI.norm(2,0))
        
        self.lambd = 1.0
        self.ACT_HISTORY_LEN = 300
        self.HessianDiag = torch.zeros(self.hidden_dim).to(self.device)
        self.ActL1 = torch.zeros(self.hidden_dim).to(self.device)
        self.signalEnergy = 0.
        self.noiseEnergy = 0.
        


    def forward(self, batch, batch_freq):
        I_cuda = batch.T.to(self.device)
        frequency = batch_freq.to(self.device)
        ahat, Res, recons = FISTA(I_cuda, self.PHI, self.reg, 500, self.device)

        #Statistics Collection
        self.ActL1 = self.ActL1.mul((self.ACT_HISTORY_LEN-1.0)/self.ACT_HISTORY_LEN) + ahat.abs().mean(1)/self.ACT_HISTORY_LEN
        self.HessianDiag = self.HessianDiag.mul((self.ACT_HISTORY_LEN-1.0)/self.ACT_HISTORY_LEN) + torch.pow(ahat,2).mean(1)/self.ACT_HISTORY_LEN

        self.signalEnergy = self.signalEnergy*((self.ACT_HISTORY_LEN-1.0)/self.ACT_HISTORY_LEN) + torch.pow(I_cuda,2).sum()/self.ACT_HISTORY_LEN
        self.noiseEnergy = self.noiseEnergy*((self.ACT_HISTORY_LEN-1.0)/self.ACT_HISTORY_LEN) + torch.pow(Res,2).sum()/self.ACT_HISTORY_LEN
        snr = self.signalEnergy/self.noiseEnergy

        #Dictionary Update
        self.PHI = quadraticBasisUpdate(self.PHI, Res*(1/frequency), ahat, 0.001, self.HessianDiag, 0.005)
        
        return recons, ahat, (snr, self.ActL1.max(), self.ActL1.min())
    
    def inference(self, batch):
        I_cuda = batch.T.to(self.device)
        ahat, Res, recons = FISTA(I_cuda, self.PHI, self.reg, 500, self.device)
        return recons, ahat
 
    
class ConvexOptimExtractor(nn.Module, BaseExtractor):
    """
    paper: Transformer visualization via dictionary learning: contextualized embedding as a linear superposition of transformer factors
    """
    def __init__(self, cfg, dataloader):
        super(ConvexOptimExtractor, self).__init__()
        self.cfg = cfg
        self.dataloader = dataloader
        self.model = ConvexOptimModel(cfg)
        self.is_trained = False
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg["lr"], betas=(self.cfg["beta1"], self.cfg["beta2"]))
        
    @classmethod
    def code(cls):
        return "convex_optim"
    
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
    def load_from_file(cls, dataloader, save_dir, ckpt_name=None):
        """
        Loads the saved extractor from file.
        """
        if ckpt_name is None:
            cfg = json.load(open(os.path.join(save_dir, "cfg.json"), "r"))
            state_dict = torch.load(os.path.join(save_dir, "model.pt"))
        else:
            cfg = json.load(open(os.path.join(save_dir, ckpt_name+".json"), "r"))
            state_dict = torch.load(os.path.join(save_dir, ckpt_name+".pt"))
        self = cls(cfg, dataloader)
        self.load_state_dict(state_dict)
        return self
    
    def train(self, explained_model):
        logger.info('start training convex optim...')
        log_freq = self.cfg['val_freq']
        best_recons = 0
        for epoch in range(self.cfg["epoch"]):
            logger.info('epoch:{}'.format(epoch + 1))
            logger.info('total iterations:{}'.format(self.dataloader.__len__()))
            logger.info('\n')
            
            loss_dict = {
                "snr": [],
                "act_max": [],
                "act_min": [],
            }
            for iter in tqdm(range(len(self.dataloader))):
                # get batch
                activations, freq = self.dataloader.next()
                if self.dataloader.empty_flag == 1:
                    logger.info('All training data in dataloader has been passed through.')
                    break
                activations = activations.type(torch.FloatTensor).to(self.cfg["device"])
                
                # forward
                recons, ahat, observations = self.model(activations, freq)            
                snr, act_max, act_min = observations
                
                # log outputs
                loss_dict["snr"].append(snr.item())
                loss_dict["act_max"].append(act_max.item())
                loss_dict["act_min"].append(act_min.item())
                
                # log outputs and save best model
                if (iter+1) % log_freq == 0:
                    logger.info('iter:{}'.format(iter + 1))
                    logger.info('snr:{:.4f}'.format(sum(loss_dict["snr"]) / log_freq))
                    logger.info('act_max:{:.4f}'.format(sum(loss_dict["act_max"]) / log_freq))
                    logger.info('act_min:{:.4f}'.format(sum(loss_dict["act_min"]) / log_freq))
                    loss_dict = {
                        "snr": [],
                        "act_max": [],
                        "act_min": [],
                    }
                            
                    recons_score, _, _, _ = get_recons_loss(self.dataloader, explained_model, self.cfg, self.model, num_batches=5)
                    if recons_score > best_recons:
                        best_recons = recons_score
                        logger.info(f"Saving best model with reconstruction score: {recons_score:.2%}")
                        # save function
        self.is_trained = True               
    
    def extract_concepts(self, model):
        if self.is_trained:
            return
        self.train(model)
    
    
    def get_concepts(self):                   
        return self.model.PHI.T 

    
    @torch.no_grad()
    def activation_func(self, x, concept_idx):
        out, h, loss, loss_terms = self.model(x, x)
        return h[:, concept_idx]


def replacement_hook(mlp_post, hook, encoder):
    # mlp_post [b, l, d]
    mlp_post_shape = mlp_post.shape
    mlp_post = mlp_post.reshape((-1, mlp_post_shape[-1]))
    mlp_post_reconstr = encoder.inference(mlp_post)[0]
    mlp_post_reconstr = mlp_post_reconstr.reshape(mlp_post_shape)
    return mlp_post_reconstr

def zero_ablate_hook(mlp_post, hook):
    mlp_post[:] = 0.
    return mlp_post

def get_recons_loss(dataloader, model, cfg, encoder, num_batches=5):
    with torch.no_grad():
        loss_list = []
        for i in range(num_batches):
            tokens, token_freq = dataloader.get_processed_random_batch()
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
    
    
def quadraticBasisUpdate(basis, Res, ahat, lowestActivation, HessianDiag, stepSize = 0.001,constraint = 'L2', Noneg = False):
    """
    This matrix update the basis function based on the Hessian matrix of the activation.
    It's very similar to Newton method. But since the Hessian matrix of the activation function is often ill-conditioned, we takes the pseudo inverse.

    Note: currently, we can just use the inverse of the activation energy.
    A better idea for this method should be caculating the local Lipschitz constant for each of the basis.
    The stepSize should be smaller than 1.0 * min(activation) to be stable.
    """
    dBasis = stepSize*torch.mm(Res, ahat.t())/ahat.size(1)
    dBasis = dBasis.div_(HessianDiag+lowestActivation)
    basis = basis.add_(dBasis)
    if Noneg:
        basis = basis.clamp(min = 0.)
    if constraint == 'L2':
        basis = basis.div_(basis.norm(2,0))
    return basis

def ISTA_PN(I,basis,lambd,num_iter,eta=None, useMAGMA=True):
    # This is a positive-negative PyTorch-Ver ISTA solver
    # MAGMA uses CPU-GPU hybrid method to solve SVD problems, which is great for single task. When running multiple jobs, this flag should be turned off to leave the svd computation on only GPU.
    dtype = basis.type()
    batch_size=I.size(1)
    M = basis.size(1)
    if eta is None:
        if useMAGMA:
            L = torch.max(torch.linalg.eigvalsh(torch.mm(basis,basis.t())))
            eta = 1./L
        else:
            eta = 1./cp.linalg.eigvalsh(cp.asarray(torch.mm(basis,basis.t()).cpu().numpy())).max().get().reshape(1)
            eta = torch.from_numpy(eta.astype('float32')).cuda()

    #Res = torch.zeros(I.size()).type(dtype)
    #ahat = torch.zeros(M,batch_size).type(dtype)
    Res = torch.cuda.FloatTensor(I.size()).fill_(0)
    ahat = torch.cuda.FloatTensor(M,batch_size).fill_(0)

    for t in range(num_iter):
        ahat = ahat.add(eta * basis.t().mm(Res))
        ahat_sign = torch.sign(ahat)
        ahat.abs_()
        ahat.sub_(eta * lambd).clamp_(min = 0.)
        ahat.mul_(ahat_sign)
        Res = I - torch.mm(basis,ahat)
    return ahat, Res

def FISTA(I,basis,lambd,num_iter, device, eta=None, useMAGMA=True):
    # This is a positive-only PyTorch-Ver FISTA solver
    dtype = basis.type()
    batch_size=I.size(1)
    M = basis.size(1)
    if eta is None:
        if useMAGMA:
            L = torch.max(torch.linalg.eigvalsh(torch.mm(basis,basis.t())))
            eta = 1./L
        else:
            eta = 1./cp.linalg.eigvalsh(cp.asarray(torch.mm(basis,basis.t()).cpu().numpy())).max().get().reshape(1)
            eta = torch.from_numpy(eta.astype('float32')).to(device)

    tk_n = 1.
    tk = 1.
    Res = torch.cuda.FloatTensor(I.size()).fill_(0).to(device)
    ahat = torch.cuda.FloatTensor(M,batch_size).fill_(0).to(device)
    ahat_y = torch.cuda.FloatTensor(M,batch_size).fill_(0).to(device)

    for t in range(num_iter):
        tk = tk_n
        tk_n = (1+np.sqrt(1+4*tk**2))/2
        ahat_pre = ahat
        Res = I - torch.mm(basis,ahat_y)
        ahat_y = ahat_y.add(eta * basis.t().mm(Res))
        ahat = ahat_y.sub(eta * lambd).clamp(min = 0.)
        ahat_y = ahat.add(ahat.sub(ahat_pre).mul((tk-1)/(tk_n)))
    Res = I - torch.mm(basis,ahat)
    recons = torch.mm(basis,ahat)
    return ahat, Res, recons

def ISTA(I,basis,lambd,num_iter,eta=None, useMAGMA=True):
    # This is a positive-only PyTorch-Ver ISTA solver
    dtype = basis.type()
    batch_size=I.size(1)
    M = basis.size(1)
    if eta is None:
        if useMAGMA:
            L = torch.max(torch.linalg.eigvalsh(torch.mm(basis,basis.t())))
            eta = 1./L
        else:
            eta = 1./cp.linalg.eigvalsh(cp.asarray(torch.mm(basis,basis.t()).cpu().numpy())).max().get().reshape(1)
            eta = torch.from_numpy(eta.astype('float32')).cuda()

    #Res = torch.zeros(I.size()).type(dtype)
    #ahat = torch.zeros(M,batch_size).type(dtype)
    Res = torch.cuda.FloatTensor(I.size()).fill_(0)
    ahat = torch.cuda.FloatTensor(M,batch_size).fill_(0)

    for t in range(num_iter):
        ahat = ahat.add(eta * basis.t().mm(Res))
        ahat = ahat.sub(eta * lambd).clamp(min = 0.)
        Res = I - torch.mm(basis,ahat)
    return ahat, Res