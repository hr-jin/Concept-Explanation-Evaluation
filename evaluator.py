import torch
import torch.nn.functional as F
import numpy as np
from utils import *
from dataloader import *
from functools import partial

class Concept_Evaluator:
    """
    Evaluate concept.
    """
    def __init__(self, **kwargs):
        pass
    
class AE_Evaluator(Concept_Evaluator):
    def __init__(self, AE):
        super().__init__()
        if  "pythia" in self.cfg['model_to_interpret']:
            self.AE = AE
        elif "gpt" in self.cfg['model_to_interpret']:
            ...
        elif "llama" in self.cfg['model_to_interpret']:
            ...

    @staticmethod
    def replacement_hook(mlp_post, hook, encoder):
        mlp_post_reconstr = encoder(mlp_post)[0]
        return mlp_post_reconstr

    @staticmethod
    def zero_ablate_hook(mlp_post, hook):
        mlp_post[:] = 0.
        return mlp_post

    @staticmethod
    def get_freqs(AE, dataloader, cfg, num_batches=25):
        with torch.no_grad():
            print('get freqs...')
            act_freq_scores = torch.zeros(AE.d_hidden, dtype=torch.float32).to(cfg['device'])
            total = 0
            for i in range(num_batches):
                mlp_acts = dataloader.buffer[torch.randperm(dataloader.buffer.shape[0])[:cfg["batch_size"]]]
                mlp_acts = mlp_acts.to(cfg['device'])

                _, acts = AE(mlp_acts)

                act_freq_scores += (acts > 0).sum(0)
                total+=acts.shape[0]
            act_freq_scores /= total
            num_dead = (act_freq_scores==0).float().mean()
            return act_freq_scores, num_dead

    @staticmethod
    def get_recons_loss(AE, dataloader, model, cfg, num_batches=5):
        with torch.no_grad():
            loss_list = []
            for i in range(num_batches):
                tokens = dataloader.data[torch.randperm(len(dataloader.data))[:cfg["model_batch_size"]]]['tokens']
                tokens = torch.tensor(tokens)
                tokens = tokens[:, :cfg['seq_len']]
                tokens[:, 0] = model.tokenizer.bos_token_id
                loss = model(tokens, return_type="loss")
                recons_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(cfg["act_name"], partial(AE_Evaluator.replacement_hook, encoder=AE))])
                zero_abl_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(cfg["act_name"], AE_Evaluator.zero_ablate_hook)])
                loss_list.append((loss, recons_loss, zero_abl_loss))
            losses = torch.tensor(loss_list)
            loss, recons_loss, zero_abl_loss = losses.mean(0).tolist()

            print(f"loss: {loss:.4f}, recons_loss: {recons_loss:.4f}, zero_abl_loss: {zero_abl_loss:.4f}")
            score = ((zero_abl_loss - recons_loss)/(zero_abl_loss - loss))
            print(f"Reconstruction Score: {score:.2%}")
            return score, loss, recons_loss, zero_abl_loss

    @staticmethod
    def get_l0_norm(AE, dataloader, cfg, num_batches=5):
        with torch.no_grad():
            print('get l0 norm...')
            num_feature = AE.d_hidden
            l0_norms = []
            for i in range(num_batches):
                mlp_acts = dataloader.buffer[torch.randperm(dataloader.buffer.shape[0])[:cfg["batch_size"]]]
                mlp_acts = mlp_acts.to(cfg['device'])

                _, acts = AE(mlp_acts)
                acts = acts.reshape(-1, num_feature)
                l0_norm = torch.linalg.norm(acts, ord=0, dim=-1).sum() / acts.shape[0]
                l0_norms.append(l0_norm)
            l0_norm = sum(l0_norms) / len(l0_norms)
            return l0_norm
    
    