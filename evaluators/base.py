from abc import *
import torch
import time
import numpy as np
from transformer_lens import utils
import pandas as pd
from tqdm import tqdm
from logger import logger

class BaseEvaluator(metaclass=ABCMeta):
    def __init__(self, cfg, activation_func, model):
        super().__init__()
        self.activation_func = activation_func
        self.cfg = cfg
        self.model = model

    @classmethod
    @abstractmethod
    def code(cls):
        pass
    
    @torch.no_grad
    def get_hidden_states(self, tokens):
        _, cache = self.model.run_with_cache(tokens, stop_at_layer=self.cfg["layer"]+1, names_filter=self.cfg["act_name"])
        hidden_states = cache[self.cfg["act_name"]]
        return hidden_states
    
    @torch.no_grad
    def get_most_critical_tokens(self, all_tokens, concept=None, concept_idx=-1):   
        """
        Args:
            tokens (tensor): B, seqlen
            concept (_type_): An index for AE, or an vector for other
        """
             
        T1 = time.time()
        _, maxlen = all_tokens.shape[0], all_tokens.shape[1]
        minibatch = 128
        all_tokens = all_tokens.split(minibatch, dim=0)
            
        results = []
        for tokens in tqdm(all_tokens, desc='Searching the corpus for the most critical token for the current concept'):
            num_tokens = minibatch * maxlen
            concept_acts = self.activation_func(tokens, self.model, concept, concept_idx) # minibatch * maxlen
            origin_acts = concept_acts # minibatch * maxlen
            

            most_imp_actis = np.zeros([num_tokens])
            most_imp_pos = np.zeros([num_tokens])
            most_imp_tokens = np.zeros([num_tokens])
            imp_this_token = np.zeros([num_tokens])

            padding_id = self.model.tokenizer.unk_token_id
            for padding_position in range(maxlen):
                tmp_tokens = tokens.clone()
                tmp_tokens[:,padding_position] = padding_id
                concept_acts = self.activation_func(tmp_tokens, self.model, concept, concept_idx) # minibatch * maxlen
                
                acti_diff = origin_acts - concept_acts
                acti_diff = acti_diff.detach().cpu().numpy()

                mask = np.ones((minibatch, maxlen))
                mask[:, padding_position] = 0
                mask = mask.reshape((minibatch * maxlen))
                imp_this_token = np.where(mask == 0, acti_diff , imp_this_token)
                
                acti_diff =  acti_diff * mask
                
                now_pos = np.array([padding_position for _ in range(minibatch * maxlen)])
                padding_token_id = np.tile(tokens[:,padding_position].unsqueeze(1).cpu().numpy(), [1, maxlen])
                padding_token_id = padding_token_id.reshape((minibatch * maxlen))

                indices = most_imp_actis > acti_diff
                most_imp_pos = np.where(indices, most_imp_pos , now_pos)
                most_imp_tokens = np.where(indices, most_imp_tokens , padding_token_id)
                most_imp_actis = np.where(indices, most_imp_actis , acti_diff)

            tokens_reshaped = tokens.reshape((-1)).cpu().numpy()
            df = pd.DataFrame(data=self.model.to_str_tokens(tokens_reshaped.astype(np.int32)), columns=["token"])
            df_ctxt = pd.DataFrame(data=self.model.to_str_tokens(most_imp_tokens.astype(np.int32)), columns=["token"])
            
            
            df["imp"] = imp_this_token
            df_ctxt["imp"] = most_imp_actis
            df_agg = df.groupby('token').agg(['max'])['imp'].sort_values(by=['max'],ascending=False)
            df_ctxt_agg = df_ctxt.groupby('token').agg(['max'])['imp'].sort_values(by=['max'],ascending=False)
            
            result = pd.merge(df_agg, df_ctxt_agg, on=['token'], how='left').reset_index()
            result.fillna(0, inplace = True)
                        
            result['imp'] = result['max_x'] + result['max_y']
            result = result.sort_values(by=['imp'],ascending=False)[['token','imp']][:20]
            results.append(result)
            
        
        df_final = pd.concat(results).groupby('token').agg('max').sort_values(by=['imp'],ascending=False).reset_index()[['token','imp']][:10]
        print('df_final:', df_final)
        max_value = df_final['imp'].values.max()
        df_most_critical_tokens = df_final[df_final['imp']>=max_value*0.9]['token'].apply(lambda x:x.strip().lower()).values
        df_most_critical_tokens = df_most_critical_tokens[df_most_critical_tokens != '']
        logger.info('The most critical tokens(after removing the spaces and changing to lower case): {}'.format(','.join(df_most_critical_tokens)))
        return df_most_critical_tokens
    

