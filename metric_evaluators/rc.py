from .base import BaseMetricEvaluator

import torch
import torch.nn as nn
from logger import logger
from audtorch.metrics.functional import pearsonr
import torch.nn.functional as F
import time
from tqdm import tqdm

class ReliabilityConsistencyEvaluator(nn.Module, BaseMetricEvaluator):
    def __init__(self, cfg):
        nn.Module.__init__(self)
        BaseMetricEvaluator.__init__(self, cfg)
        
    @classmethod
    def code(cls):
        return 'rc'
    
    def get_metric(
        self, 
        eval_tokens, 
        evaluator_dict=dict(), 
        concepts=[], 
        concept_idxs=[], 
        origin_tokens=None,
        most_imp_tokens=None,
        most_imp_idxs=None,
        origin_imp_idxs=None,
        **kwargs
    ):            
        evaluator_names = list(evaluator_dict.keys())        
        minibatch = self.cfg['metric_eval_batchsize']
        origin_tokens = eval_tokens
        print('origin_tokens.shape:',origin_tokens.shape)
        eval_tokens = eval_tokens.split(minibatch, dim=0) 
        print('len(eval_tokens):',len(eval_tokens))
          
        metric_list = []
        topic_tokens = [[None for i in range(len(concept_idxs))] for j in range(len(eval_tokens))]
        topic_idxs = [[None for i in range(len(concept_idxs))] for j in range(len(eval_tokens))]
        origin_critical_idxs = [[None for i in range(len(concept_idxs))] for j in range(len(eval_tokens))]

        origin_dfs = [[None for i in range(len(concept_idxs))] for j in range(len(eval_tokens))]
        most_preferred_tokens = [[None for i in range(len(concept_idxs))] for j in range(len(eval_tokens))]
   
        pre_metrics = dict()
        pre_concept_acts = dict()
        for i, tokens in enumerate(eval_tokens):
            logger.info('Metric evaluation on subdataset {}...\n'.format(i+1))
            token_freq_dict, freq_threshold = BaseMetricEvaluator.get_token_freq(tokens)
            tmp_metric_list = []
            hidden_states_all = []
            if ('itc' in name) and (len(hidden_states_all) == 0):
                _, maxlen = tokens.shape[0], tokens.shape[1]
                minibatch = self.cfg['concept_eval_batchsize']
                tokens_tuple = tokens.split(minibatch, dim=0)
                padding_id = evaluator.model.tokenizer.unk_token_id
                T1 = time.time()
                for tokens_ in tqdm(tokens_tuple, desc='model forward: get hidden states'):
                    hidden_states_tmp = [evaluator.hidden_state_func(tokens_, evaluator.model)]
                    for padding_position in range(maxlen):
                        tmp_tokens = tokens_.clone().to(self.cfg['device'])
                        tmp_tokens[:,padding_position] = padding_id
                        hidden_states_tmp.append(evaluator.hidden_state_func(tmp_tokens, evaluator.model).to('cpu'))
                    print(len(hidden_states_tmp))
                    hidden_states_all.append(hidden_states_tmp)
                print(len(hidden_states_all))
                T2 = time.time()
                print('\nhidden states calculating cost:%s s' % ((T2-T1)))
                
            for name, evaluator in evaluator_dict.items():  
                logger.info('Evaluating {} ...'.format(name))
                concept_metric_list = []
                
                for j, concept_idx in enumerate(concept_idxs):
                    concept = concepts[j]
                    evaluator.update_concept(concept, concept_idx) 
                    if 'itc' in name:
                        if topic_tokens[i][j] is None:
                            tmp_tokens, tmp_idxs, origin_df, origin_critical_idxs_tmp = evaluator.get_most_critical_tokens(
                                                                                        eval_tokens, 
                                                                                        concept, 
                                                                                        concept_idx,
                                                                                        token_freq_dict,
                                                                                        freq_threshold,
                                                                                        hidden_states_all
                                                                                        )
                            topic_tokens[i][j] = tmp_tokens
                            topic_idxs[i][j] = tmp_idxs
                            origin_dfs[i][j] = origin_df
                            origin_critical_idxs[i][j] = origin_critical_idxs_tmp
                        concept_metric = evaluator.get_metric(
                            origin_tokens, 
                            topic_tokens[i][j], 
                            topic_idxs[i][j], 
                            origin_critical_idxs[i][j], 
                            origin_df=origin_dfs[i][j], 
                            token_freq_dict=token_freq_dict, 
                            freq_threshold=freq_threshold
                        )
                    elif 'replace-ablation' in name: 
                        abl_str = name.replace('replace-ablation', 'ablation') + str(concept_idx)
                        rep_str = name.replace('replace-ablation', 'replace') + str(concept_idx)
                        tmp_acts = pre_concept_acts[abl_str]
                        tmp_metrics = pre_metrics[rep_str] + pre_metrics[abl_str] # ablation metrics has been inverted
                        concept_metric = evaluator.get_metric(tokens, tmp_metrics, tmp_acts)
                    elif ('replace' in name) or ('ablation' in name):
                        concept_metric, tmp_metrics, tmp_acts = evaluator.get_metric(tokens, return_metric_and_acts=True)
                        pre_metrics[name + str(concept_idx)] = tmp_metrics
                        pre_concept_acts[name + str(concept_idx)] = tmp_acts
                    else:
                        concept_metric = evaluator.get_metric(tokens)
                    concept_metric_list.append(concept_metric)
                tmp_metric_list.append(concept_metric_list)
            metric_list.append(tmp_metric_list)
        separate_metrics = torch.tensor(metric_list) # n_minibatch, n_metrics, n_concepts
        separate_metrics = separate_metrics.permute(1,0,2) # n_metrics, n_minibatch, n_concepts
        print('separate_metrics:\n',separate_metrics)
        separate_vars_agg = torch.var(separate_metrics, dim=-1)
        print('separate_vars:\n',separate_vars_agg)
        separate_vars_agg = separate_vars_agg.sum(-1) # n_metrics
        integral_vars = torch.var(separate_metrics.sum(1), dim=-1) # n_metrics
        
        J = separate_metrics.shape[1] 
        final_metrics = J / (J - 1) * (integral_vars - separate_vars_agg) / integral_vars # n_metrics
        print('J:', J)
        print('separate_vars_agg:', separate_vars_agg)
        print('integral_vars:', integral_vars)
        logger.info('Metric Consistency: \n{}'.format(
            ' '.join(
                ['{}:{:4f}'.format(evaluator_names[i],final_metrics[i]) for i in range(final_metrics.shape[0])]
                )
            ))    
        return final_metrics
        
            