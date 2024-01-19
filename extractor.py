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
import os
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from logger import logger

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
    
class AutoEncoder(Concept_Extractor):
    """
    An unsupervised learning method that hopes to learn sparsely activated concepts through AutoEncoder
    """
    def __init__(self, cfg):
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
        self.b_dec = nn.Parameter(torch.zeros(d_out, dtype=dtype))
        
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.cfg = cfg
        self.d_hidden = d_hidden
        
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
        Returns:
            A dictionary with the shape [d_hidden, d_out], each line contains a concept
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg["lr"], betas=(self.cfg["beta1"], self.cfg["beta2"]))
        best_reconstruct = 0
        time_start=time.time()
        for epoch in range(self.cfg['epoch']):
            logger.info('epoch:{}'.format(epoch + 1))
            logger.info('total iterations:{}'.format(dataloader.__len__()))
            logger.info('\n')
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
                loss_dict = {"AE_loss": loss.item(), "l2_loss": l2_loss.item(), "l1_loss": l1_loss.item()}
                
                if (iter + 1) % self.cfg['val_freq'] == 0:
                    time_end=time.time()
                    logger.info('Epoch: {} Iteration: {} Total time: {:.4f}s'.format(epoch+1, iter+1, time_end-time_start))
                    logger.info("Finished: {:.3f} % of Epoch {}".format((100 * (iter+1) / self.cfg['num_batches']), epoch + 1))
                    logger.info(" ".join(["{}: {:.4f}".format(metric_name, metric_val) for metric_name, metric_val in loss_dict.items()]))
                    x = AE_Evaluator.get_recons_loss(self, dataloader, model, self.cfg, num_batches=5)
                    l0 = AE_Evaluator.get_l0_norm(self, dataloader, self.cfg, num_batches=5)
                    logger.info("l0 norm: {:.4f}".format(l0))
                    if x[0] > best_reconstruct:
                        best_reconstruct = x[0]
                        self.save(save_dir, ckpt_name="best_reconstruct")
                    print('\n')
                        
                if (iter + 1) % 10000 == 0:
                    logger.info('saved at:', save_dir)
                    self.save(save_dir, ckpt_name="Iteration" + str(iter+1) + "_Epoch" + str(epoch+1))
                    freqs, num_dead = AE_Evaluator.get_freqs(self, dataloader, self.cfg, num_batches=25)
                    if self.cfg['reinit'] == 1:
                        to_be_reset = (freqs<10**(-5.5))
                        self.re_init(self, to_be_reset)
                del loss, acti_reconstruct, mid_acts, l2_loss, l1_loss
            self.save(save_dir, ckpt_name="Iteration" + str(iter) + "_Epoch" + str(epoch+1))
        return self.W_dec.clone().detach()
            
class TCAV_Extractor(Concept_Extractor):
    
    def __init__(self, cfg, model, token_idx = -1):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.tokenizer = model.tokenizer
        self.cavs      = []
        self.token_idx = token_idx
        self.act_name = cfg['act_name']
    
    def get_reps(self, concept_examples):
        with torch.no_grad():
            inputs = self.tokenizer(concept_examples, max_length=128, truncation=True, padding=True,return_tensors="pt")
            inputs = inputs.to('cpu')
            _, cache = self.model.run_with_cache(inputs['input_ids'], names_filter=[self.act_name])
            concept_repres = cache[self.act_name].cpu().detach().numpy()
            concept_repres = concept_repres[np.arange(concept_repres.shape[0]),(inputs['attention_mask'].sum(-1)-1),:]
        return concept_repres
   

    def extract_concepts(self, dataloader, delta=None, num_runs=1):
        positive_concept_examples, negative_concept_examples = dataloader.next()
        reps = self.get_reps(positive_concept_examples + negative_concept_examples)
        
        positive_embedding = reps[:reps.shape[0]//2]
        negative_embedding = reps[reps.shape[0]//2:]
        
        positive_labels = np.ones((len(positive_concept_examples), ))  
        negative_labels = np.zeros((len(negative_concept_examples),))  
        
        X = np.vstack((positive_embedding, negative_embedding))
        Y = np.concatenate((positive_labels, negative_labels))
        cavs = []
        accuracy_train_list = []
        accuracy_test_list = []

        for i in range(num_runs):
            x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=i)
            if delta is None:
                log_reg = LogisticRegression(solver='saga', max_iter=10000)
            else:
                log_reg = LogisticRegression(penalty='l1', C=delta, solver='saga', max_iter=10000)
            log_reg.fit(x_train, y_train)
            
            predictions_test = log_reg.predict(x_test)
            predictions_train = log_reg.predict(x_train)
            accuracy_test = accuracy_score(y_test, predictions_test)
            accuracy_train = accuracy_score(y_train, predictions_train)
            accuracy_train_list.append(accuracy_train)
            accuracy_test_list.append(accuracy_test)
            cav = log_reg.coef_[0]
            cavs.append(cav)

        acc = np.mean(accuracy_test_list)
        self.cavs = cavs
        logger.info('Acc in training set: {:.2f}, in test set: {:.2f}'.format(np.mean(accuracy_train_list), np.mean(accuracy_test_list)))
        return cavs, acc
    
    def save_cav(self, path):
        torch.save(self.cavs, path) 