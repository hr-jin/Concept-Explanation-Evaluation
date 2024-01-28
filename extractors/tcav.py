from .base import BaseExtractor

import torch
import numpy as np
import torch.nn as nn
from utils import *
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from logger import logger

class TCAVExtractor(nn.Module, BaseExtractor):
    
    def __init__(self, cfg, dataloader, token_idx = -1):
        super().__init__()
        self.cfg = cfg
        self.dataloader = dataloader
        self.model = dataloader.model
        self.tokenizer = self.model.tokenizer
        self.cavs = []
        self.token_idx = token_idx
        self.act_name = cfg['act_name']
    
    @classmethod
    def code(cls):
        return 'tcav'
    
    def get_reps(self, concept_examples):
        with torch.no_grad():
            inputs = self.tokenizer(concept_examples, max_length=128, truncation=True, padding=True,return_tensors="pt")
            inputs = inputs.to('cpu')
            _, cache = self.model.run_with_cache(inputs['input_ids'], names_filter=[self.act_name])
            concept_repres = cache[self.act_name].cpu().detach().numpy()
            concept_repres = concept_repres[np.arange(concept_repres.shape[0]),(inputs['attention_mask'].sum(-1)-1),:]
        return concept_repres
    
    def activation_func(self, tokens, model, concept, concept_idx):
        with torch.no_grad():
            return self.classifier.predict(tokens)
   
    def extract_concepts(self, model):
        pos_examples, neg_examples, pos_labels, neg_labels = self.dataloader.next()
        reps = self.get_reps(pos_examples + neg_examples)
        
        positive_embedding = reps[:reps.shape[0]//2]
        negative_embedding = reps[reps.shape[0]//2:]
        
        X = np.vstack((positive_embedding, negative_embedding))
        Y = np.concatenate((pos_labels, neg_labels))

        x_train, x_val, y_train, y_val = train_test_split(X, Y, random_state=0)
        self.classifier = LogisticRegression(solver='saga', max_iter=10000)
        self.classifier.fit(x_train, y_train)
        
        predictions_val = self.classifier.predict(x_val)
        predictions_train = self.classifier.predict(x_train)
        accuracy_val = accuracy_score(y_val, predictions_val)
        accuracy_train = accuracy_score(y_train, predictions_train)
        cav = self.classifier.coef_[0]
    
        self.coef_ = self.classifier.coef_
        self.intercept_ = self.classifier.intercept_

        self.concept = cav
        logger.info('Acc in training set: {:.2f}, in val set: {:.2f}'.format(np.mean(accuracy_train), np.mean(accuracy_val)))
        return cav, accuracy_val
    
    def get_concepts(self):
        return self.concept
    
    def get_log_reg_params(self):
        return self.coef_, self.intercept_
    
    def save_cav(self, path):
        torch.save(self.cavs, path) 
