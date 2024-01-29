import argparse
from utils import *
from config import cfg as default_cfg
import logging
from dataloaders import dataloader_factory
from extractors import extractor_factory
from datasets_ import dataset_factory
from models import model_factory
from evaluators import evaluator_factory
from metric_evaluators import metric_evaluator_factory
import json
from prettytable import PrettyTable


def main():
    """
    Here we take the process of concept extraction by autoencoder as an example.
    """
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO, force=True)
    logger = logging.getLogger('logger')

    parser = argparse.ArgumentParser()
    cfg, args = arg_parse_update_cfg(default_cfg, parser)

    model = model_factory(cfg)
    
    logger.info('loaded model...')
    
    cfg = process_cfg(cfg, model)
    print(json.dumps(cfg, indent=2))
    
    train_data = dataset_factory(cfg)
    dataloader = dataloader_factory(cfg, train_data, model)
    extractor = extractor_factory(cfg, dataloader)
    if cfg['load_extractor']:
        logger.info('loading extractor...')
        extractor = extractor.load_from_file(dataloader, cfg['load_path'], cfg).to(cfg['device'])
    else:
        logger.info('extract concepts...')
        extractor.extract_concepts(model)
        
    
    concepts = extractor.get_concepts()
    print('concept vectors:', concepts)
    
        
    # cfg['data_dir'] = "/root/autodl-tmp/haoran/SparseAE-pythia-pile/data/pile_neel/"
    # cfg['dataset_name'] = "pile-tokenized-10b"
    # cfg['extractor'] = "ae"
    # eval_data = dataset_factory(cfg)
    # dataloader = dataloader_factory(cfg, eval_data, model)
    
    # concept_idx = 2919
    # concept_idx = 0
    
    concept_idxs = [ 735, 3821, 2811, 3998,  766, 1546,   34, 2122, 1413, 4094,3065, 1120, 3856,   37, 1672, 2057, 4064, 2950,  521, 2561,1436, 2230,  155, 3133, 2500, 2078, 3203, 2243,   39, 1600]
        
    # concept_idxs = [ 735, 3821, 2811, 3998,  766, 1546,]
    
    token_list = []
    for _ in range(1):
        tokens = dataloader.get_processed_random_batch()
        token_list.append(tokens)
    tokens = torch.cat(token_list, 0)
    
    evaluator_dict = dict()
                
                    
    # cfg['evaluator'] = 'faithfulness'
    # for disturb in ['gradient']:
    #     for measure_obj in ['pred_logit']:
    #         extractor_str = disturb + '_' + measure_obj
    #         evaluator_dict[extractor_str] = evaluator_factory(
    #             cfg, 
    #             extractor.activation_func, 
    #             model,
    #             concept=None, 
    #             concept_idx=None, 
    #             disturb=disturb, # ['ablation', 'gradient', 'replace']
    #             measure_obj=measure_obj, # ['loss', 'class_logit', 'logits']
    #             corr_func='pearson', # ['pearson', 'KL_div', 'openai_var']
    #             class_idx=None, 
    #             logits_corr_topk=None,
    #         )
            
    cfg['evaluator'] = 'otc'
    evaluator_dict.update({
        'otc_emb_dist': evaluator_factory(
            cfg, 
            extractor.activation_func, 
            model,
            concept=None, 
            concept_idx=None, 
            pmi_type='emb_dist', # ['uci', 'umass', 'silhouette']
        ),
        'otc_emb_cos': evaluator_factory(
            cfg, 
            extractor.activation_func, 
            model,
            concept=None, 
            concept_idx=None, 
            pmi_type='emb_cos', # ['uci', 'umass', 'silhouette']
        ),
        'otc_silhouette': evaluator_factory(
            cfg, 
            extractor.activation_func, 
            model,
            concept=None, 
            concept_idx=None, 
            pmi_type='silhouette', # ['uci', 'umass', 'silhouette']
        ),
        
    })
        
    cfg['evaluator'] = 'itc'
    evaluator_dict.update({
        'itc_emb_dist': evaluator_factory(
            cfg, 
            extractor.activation_func, 
            model,
            concept=None, 
            concept_idx=None, 
            pmi_type='emb_dist', # ['uci', 'umass', 'silhouette']
        ),
        'itc_emb_cos': evaluator_factory(
            cfg, 
            extractor.activation_func, 
            model,
            concept=None, 
            concept_idx=None, 
            pmi_type='emb_cos', # ['uci', 'umass', 'silhouette']
        ),
        'itc_silhouette': evaluator_factory(
            cfg, 
            extractor.activation_func, 
            model,
            concept=None, 
            concept_idx=None, 
            pmi_type='silhouette', # ['uci', 'umass', 'silhouette']
        ),
        'itc_uci': evaluator_factory(
            cfg, 
            extractor.activation_func, 
            model,
            concept=None, 
            concept_idx=None, 
            pmi_type='uci', # ['uci', 'umass', 'silhouette']
        ),
        'itc_umass': evaluator_factory(
            cfg, 
            extractor.activation_func, 
            model,
            concept=None, 
            concept_idx=None, 
            pmi_type='umass', # ['uci', 'umass', 'silhouette']
        ),
        
    })
    
    # cfg['evaluator'] = 'faithfulness'
    # for disturb in ['gradient','replace-ablation','replace','ablation']:
    #     for measure_obj in ['logits', 'loss']: # , 'class_logit'
    #         if measure_obj == 'logits':
    #             for corr_func in ['KL_div', 'pearson']:
    #                 if disturb == 'gradient':
    #                     continue
    #                 for topk in [10, 1000]:
    #                     extractor_str = disturb + '_' + measure_obj + '_' + corr_func + '_top' + str(topk)
    #                     evaluator_dict[extractor_str] = evaluator_factory(
    #                     cfg, 
    #                     extractor.activation_func, 
    #                     model,
    #                     concept=None, 
    #                     concept_idx=None, 
    #                     disturb=disturb, # ['ablation', 'gradient', 'replace']
    #                     measure_obj=measure_obj, # ['loss', 'class_logit', 'logits']
    #                     corr_func=corr_func, # ['pearson', 'KL_div', 'openai_var']
    #                     class_idx=7000, 
    #                     logits_corr_topk=topk,
    #                     )
    #         else:
    #             extractor_str = disturb + '_' + measure_obj
    #             evaluator_dict[extractor_str] = evaluator_factory(
    #                 cfg, 
    #                 extractor.activation_func, 
    #                 model,
    #                 concept=None, 
    #                 concept_idx=None, 
    #                 disturb=disturb, # ['ablation', 'gradient', 'replace']
    #                 measure_obj=measure_obj, # ['loss', 'class_logit', 'logits']
    #                 corr_func=corr_func, # ['pearson', 'KL_div', 'openai_var']
    #                 class_idx=7000, 
    #                 logits_corr_topk=None,
    #             )
    
    concept_idxs = concept_idxs[:5]
    metric_dict = dict()
    itc_critical_tokens_dict = dict()
    otc_critical_tokens_dict = dict()
    for name, evaluator in evaluator_dict.items():
        metric_dict[name] = []
        for concept_idx in concept_idxs:
            concept = concepts[concept_idx]
            evaluator.update_concept(concept, concept_idx) 
            metric, critical_tokens = evaluator.get_metric(tokens, return_tokens=True)
            if evaluator.code() == 'itc':
                itc_critical_tokens_dict[concept_idx] = critical_tokens
            elif evaluator.code() == 'otc':
                otc_critical_tokens_dict[concept_idx] = critical_tokens
            metric_dict[name].append(metric)
    
    table = PrettyTable()
    table.title = 'Metrics Info'
    table.field_names = ['Metric'] + [str(i) for i in concept_idxs]

    
    for concept_idx in concept_idxs: 
        print('idx:', concept_idx, ' ,itc tokens: ', itc_critical_tokens_dict[concept_idx], ' ,otc tokens: ', otc_critical_tokens_dict[concept_idx])
    for name, evaluator in evaluator_dict.items():
        metric_arr = np.array(metric_dict[name])
        table.add_row([name] + list(np.argsort(-metric_arr)))
    print(table)
        
        
    
    # metric_evaluator = metric_evaluator_factory(cfg)
    # metric_of_metrics = metric_evaluator.get_metric(
    #     tokens, 
    #     evaluator_dict, 
    #     concepts=concepts[concept_idxs],
    #     concept_idxs=concept_idxs
    # )
    
    # metric_evaluator = metric_evaluator_factory(cfg)
    # metric_of_metrics = metric_evaluator.get_metric(
    #     tokens, 
    #     evaluator_dict, 
    #     concepts=concepts,
    #     concept_idxs=[0]
    # )
    
    # print('metric_of_metrics:\n',metric_of_metrics)
    
    
if __name__ == "__main__":
    print('hello')
    main()