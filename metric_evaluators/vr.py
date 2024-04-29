from .base import BaseMetricEvaluator
import torch
import torch.nn as nn
from logger import logger
import numpy as np
import datetime
from scipy.stats import kendalltau, pearsonr
from tqdm import tqdm
import time

class ValidityRelevanceEvaluator(nn.Module, BaseMetricEvaluator):
    def __init__(self, cfg):
        nn.Module.__init__(self)
        BaseMetricEvaluator.__init__(self, cfg)
        
    @classmethod
    def code(cls):
        return 'vr'
    
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
        logger.info('Metric evaluation ...\n')
        metric_list = []
        topic_tokens = [None for i in range(len(concept_idxs))]
        topic_idxs = [None for i in range(len(concept_idxs))]
        origin_critical_idxs = [None for i in range(len(concept_idxs))]
        if most_imp_tokens is not None:
            topic_tokens = [most_imp_tokens[i] for i in range(len(concept_idxs))]
            topic_idxs = [most_imp_idxs[i].values for i in range(len(concept_idxs))]
            origin_critical_idxs = [origin_imp_idxs[i].values for i in range(len(concept_idxs))]
        origin_dfs = [None for i in range(len(concept_idxs))]
        
        most_preferred_tokens = [None for i in range(len(concept_idxs))]
        pre_metrics = dict()
        pre_concept_acts = dict()
        print('get token freq...')
        token_freq_dict, freq_threshold = BaseMetricEvaluator.get_token_freq(eval_tokens)
        print('finished getting token freq...')
        hidden_states_reada = []
        hidden_states_faith = []
        for name, evaluator in evaluator_dict.items():     
             
            if (len(hidden_states_reada) == 0):
                _, maxlen = eval_tokens.shape[0], eval_tokens.shape[1]
                minibatch = self.cfg['concept_eval_batchsize']
                eval_tokens_tuple = eval_tokens.split(minibatch, dim=0)
                padding_id = evaluator.model.tokenizer.unk_token_id
                total_movetime = 0
                T1 = time.time()
                for tokens in tqdm(eval_tokens_tuple, desc='model forward: get hidden states'):
                    hidden_states_tmp = [evaluator.hidden_state_func(tokens, evaluator.model)]
                    for padding_position in range(maxlen):
                        tmp_tokens = tokens.clone().to(self.cfg['device'])
                        tmp_tokens[:,padding_position] = padding_id
                        hidden_states = evaluator.hidden_state_func(tmp_tokens, evaluator.model)
                        T3 = time.time()
                        hidden_states_tmp.append(hidden_states.to('cpu'))
                        T4 = time.time()
                        total_movetime += T4 - T3
                    print(len(hidden_states_tmp))
                    hidden_states_reada.append(hidden_states_tmp)
                print(len(hidden_states_reada))
                T2 = time.time()
                print('\nhidden states calculating cost:%s s' % ((T2-T1)))
                print('first device moving cost:%s s' % ((total_movetime)))
                
            if (len(hidden_states_faith) == 0):
                _, maxlen = eval_tokens.shape[0], eval_tokens.shape[1]
                minibatch = self.cfg['concept_eval_batchsize']
                eval_tokens_tuple = eval_tokens.split(minibatch, dim=0)
                padding_id = evaluator.model.tokenizer.unk_token_id
                total_movetime = 0
                T1 = time.time()
                for tokens in tqdm(eval_tokens_tuple, desc='model forward: get hidden states'):
                    hidden_states_tmp = [evaluator.hidden_state_func(tokens, evaluator.model)]
                    hidden_states_reada.append(hidden_states_tmp)
                print(len(hidden_states_reada))
                T2 = time.time()
                print('\nhidden states calculating cost:%s s' % ((T2-T1)))
                
            logger.info('Evaluating {} ...'.format(name))   
            concept_metric_list = [] 
                    
            for j, concept_idx in enumerate(concept_idxs):
                concept = concepts[j]
                evaluator.update_concept(concept, concept_idx) 
                if 'itc' in name:
                    if topic_tokens[j] is None:
                        (tmp_tokens, 
                         tmp_idxs, 
                         origin_df, 
                         origin_critical_idxs_tmp) = evaluator.get_most_critical_tokens(eval_tokens, 
                                                                                        concept, 
                                                                                        concept_idx,
                                                                                        token_freq_dict,
                                                                                        freq_threshold,
                                                                                        hidden_states_reada)
                        topic_tokens[j] = tmp_tokens
                        topic_idxs[j] = tmp_idxs
                        origin_dfs[j] = origin_df
                        origin_critical_idxs[j] = origin_critical_idxs_tmp
                    concept_metric = evaluator.get_metric(
                        origin_tokens, topic_tokens[j], topic_idxs[j], 
                        origin_critical_idxs[j],origin_df=origin_dfs[j],
                        token_freq_dict=token_freq_dict, freq_threshold=freq_threshold
                    )
                elif 'replace-ablation' in name: 
                    abl_str = name.replace('replace-ablation', 'ablation') + str(concept_idx)
                    rep_str = name.replace('replace-ablation', 'replace') + str(concept_idx)
                    tmp_acts = pre_concept_acts[abl_str]
                    tmp_metrics = pre_metrics[rep_str] + pre_metrics[abl_str] # ablation metrics has been inverted
                    concept_metric = evaluator.get_metric(eval_tokens, tmp_metrics, tmp_acts, hidden_states_all=hidden_states_all)
                elif ('replace' in name) or ('ablation' in name):
                    concept_metric, tmp_metrics, tmp_acts = evaluator.get_metric(eval_tokens, return_metric_and_acts=True, hidden_states_all=hidden_states_all)
                    pre_metrics[name + str(concept_idx)] = tmp_metrics
                    pre_concept_acts[name + str(concept_idx)] = tmp_acts
                elif 'otc' in name:
                    concept_metric, tmp_preferred_tokens = evaluator.get_metric(eval_tokens, return_tokens=True)
                    most_preferred_tokens[j] = tmp_preferred_tokens
                else:
                    assert 0, name + ' is not supported yet'
                concept_metric_list.append(concept_metric)
            metric_list.append(concept_metric_list)
        metrics = torch.tensor(metric_list) # n_metrics, n_concepts 
        
        dtime = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        
        np.save(self.cfg['output_dir'] + '/vr_data/' + str(dtime).replace(' ','_') + 'origin_metrics.npy',metrics)
        
        np.save(self.cfg['output_dir'] + '/vr_data/' + str(dtime).replace(' ','_') + 'most_imp_tokens.npy',np.array(topic_tokens))
        np.save(self.cfg['output_dir'] + '/vr_data/' + str(dtime).replace(' ','_') + 'most_imp_idxs.npy',np.array(topic_idxs))
        np.save(self.cfg['output_dir'] + '/vr_data/' + str(dtime).replace(' ','_') + 'most_pref_tokens.npy',np.array(most_preferred_tokens))
        np.save(self.cfg['output_dir'] + '/vr_data/' + str(dtime).replace(' ','_') + 'concept_idxs.npy',np.array(concept_idxs))
        np.save(self.cfg['output_dir'] + '/vr_data/' + str(dtime).replace(' ','_') + 'origin_dfs.npy',np.array(origin_dfs))
        
        pearsonr_list = []
        pearsonr_p_list = []
        for i in metrics:
            tmp_list = []
            tmp_p_list = []
            for j in metrics:
                r, pvalue = pearsonr(i, j)
                tmp_list.append(r)
                tmp_p_list.append(pvalue)
            pearsonr_list.append(tmp_list)
            pearsonr_p_list.append(tmp_p_list)
            
        kendalltau_list = []
        kendalltau_p_list = []
        for i in metrics:
            tmp_list = []
            tmp_p_list = []
            for j in metrics:
                tau, pvalue = kendalltau(i, j)
                tmp_list.append(tau)
                tmp_p_list.append(pvalue)
            kendalltau_list.append(tmp_list)
            kendalltau_p_list.append(tmp_p_list)

        pearsonr_metrics = torch.tensor(pearsonr_list).cpu().numpy() # n_metrics * n_metrics
        kendalltau_metrics = torch.tensor(kendalltau_list).cpu().numpy() # n_metrics * n_metrics
        pearson_p_metrics = torch.tensor(pearsonr_p_list).cpu().numpy() # n_metrics * n_metrics
        kendall_p_metrics = torch.tensor(kendalltau_p_list).cpu().numpy() # n_metrics * n_metrics
        
        
        
        np.save(self.cfg['output_dir'] + '/vr_data/' + str(dtime).replace(' ','_') + 'kendalltau_metrics.npy',kendalltau_metrics)
        np.save(self.cfg['output_dir'] + '/vr_data/' + str(dtime).replace(' ','_') + 'kendall_p_metrics.npy',kendall_p_metrics)
        logger.info('Metrics: '.format(str(list(evaluator_dict.keys()))))
        logger.info('Metric Validity Relevance (pearsonr): \n{}'.format(str(pearsonr_metrics)))   
        logger.info('Metric Validity Relevance (kendalltau): \n{}'.format(str(kendalltau_metrics)))   
        logger.info('P-value (pearsonr): \n{}'.format(str(pearson_p_metrics)))   
        logger.info('P-value (kendalltau): \n{}'.format(str(kendall_p_metrics)))   
        return kendalltau_metrics, pearsonr_metrics
        
            