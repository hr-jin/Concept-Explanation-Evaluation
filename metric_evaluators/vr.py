from .base import BaseMetricEvaluator
import torch
import torch.nn as nn
from logger import logger
import numpy as np
import datetime
from scipy.stats import kendalltau, pearsonr

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
        for name, evaluator in evaluator_dict.items():   
            logger.info('Evaluating {} ...'.format(name))   
            concept_metric_list = []    
            for j, concept_idx in enumerate(concept_idxs):
                concept = concepts[j]
                evaluator.update_concept(concept, concept_idx) 
                if 'itc' in name:
                    if topic_tokens[j] is None:
                        tmp_tokens, tmp_idxs, origin_df, origin_critical_idxs_tmp = evaluator.get_most_critical_tokens(eval_tokens, concept, concept_idx)
                        topic_tokens[j] = tmp_tokens
                        topic_idxs[j] = tmp_idxs
                        origin_dfs[j] = origin_df
                        origin_critical_idxs[j] = origin_critical_idxs_tmp
                    concept_metric = evaluator.get_metric(origin_tokens, topic_tokens[j], topic_idxs[j], origin_critical_idxs[j])
                elif 'replace-ablation' in name: 
                    abl_str = name.replace('replace-ablation', 'ablation') + str(concept_idx)
                    rep_str = name.replace('replace-ablation', 'replace') + str(concept_idx)
                    tmp_acts = pre_concept_acts[abl_str]
                    tmp_metrics = pre_metrics[rep_str] + pre_metrics[abl_str] # ablation metrics has been inverted
                    concept_metric = evaluator.get_metric(eval_tokens, tmp_metrics, tmp_acts)
                elif ('replace' in name) or ('ablation' in name):
                    concept_metric, tmp_metrics, tmp_acts = evaluator.get_metric(eval_tokens, return_metric_and_acts=True)
                    pre_metrics[name + str(concept_idx)] = tmp_metrics
                    pre_concept_acts[name + str(concept_idx)] = tmp_acts
                elif 'otc' in name:
                    concept_metric, tmp_preferred_tokens = evaluator.get_metric(eval_tokens, return_tokens=True)
                    most_preferred_tokens[j] = tmp_preferred_tokens
                else:
                    concept_metric = evaluator.get_metric(eval_tokens)
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
        # cosine_sim_metrics = torch.tensor(cosine_sim_list).cpu().numpy() # n_metrics * n_metrics
        
        
        
        # np.save(self.cfg['output_dir'] + '/vr_data/' + str(dtime).replace(' ','_') + 'pearsonr_metrics.npy',pearsonr_metrics)
        np.save(self.cfg['output_dir'] + '/vr_data/' + str(dtime).replace(' ','_') + 'kendalltau_metrics.npy',kendalltau_metrics)
        # np.save(self.cfg['output_dir'] + '/vr_data/' + str(dtime).replace(' ','_') + 'pearson_p_metrics.npy',pearson_p_metrics)
        np.save(self.cfg['output_dir'] + '/vr_data/' + str(dtime).replace(' ','_') + 'kendall_p_metrics.npy',kendall_p_metrics)
        
        
        # np.save(self.cfg['output_dir'] + '/vr_data/' + str(dtime).replace(' ','_') + 'cosine_sim_metrics.npy',cosine_sim_metrics)
        
        logger.info('Metrics: '.format(str(list(evaluator_dict.keys()))))
        # logger.info('Metric Validity Relevance (cosine similarity): \n{}'.format(str(cosine_sim_metrics))) 
        logger.info('Metric Validity Relevance (pearsonr): \n{}'.format(str(pearsonr_metrics)))   
        logger.info('Metric Validity Relevance (kendalltau): \n{}'.format(str(kendalltau_metrics)))   
        logger.info('P-value (pearsonr): \n{}'.format(str(pearson_p_metrics)))   
        logger.info('P-value (kendalltau): \n{}'.format(str(kendall_p_metrics))) 
        
        
        return kendalltau_metrics, pearsonr_metrics
        
            