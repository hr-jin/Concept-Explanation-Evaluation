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


def main():
    """
    Here we take the process of concept extraction by autoencoder as an example.
    """
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO, force=True)
    logger = logging.getLogger('logger')


    torch.set_default_dtype(torch.float32)

    parser = argparse.ArgumentParser()
    cfg, args = arg_parse_update_cfg(default_cfg, parser)
    
    set_seed(cfg['seed'])

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
    print('concepts.shape:', concepts.shape)
    
    concept_idxs = [i for i in range(200)]
    
    token_list = []
    origin_token_list = []
    print('\nconcept idxs:', concept_idxs)
    for _ in range(5):
        tokens, origin_tokens = dataloader.get_processed_random_batch()
        token_list.append(tokens)
        origin_token_list.append(origin_tokens)
    tokens = torch.cat(token_list, 0)
    origin_tokens = torch.cat(origin_token_list, 0)
    print('len(origin_token_list):',len(origin_token_list))
    evaluator_dict = dict()
    
    cfg['evaluator'] = 'faithfulness'
    for disturb in ['gradient','ablation']: 
        for measure_obj in ['loss','next_logit','pred_logit','logits']: 

            if measure_obj == 'logits':
                for corr_func in ['KL_div']:
                    if disturb == 'gradient':
                        continue
                    for topk in [1000]:
                        extractor_str = disturb + '_' + measure_obj + '_' + corr_func + '_top' + str(topk)
                        evaluator_dict[extractor_str] = evaluator_factory(
                        cfg, 
                        extractor.activation_func, 
                        model,
                        concept=None, 
                        concept_idx=None, 
                        disturb=disturb, 
                        measure_obj=measure_obj, 
                        corr_func=corr_func, 
                        class_idx=-1, 
                        logits_corr_topk=topk,
                        )
            else:
                extractor_str = disturb + '_' + measure_obj
                evaluator_dict[extractor_str] = evaluator_factory(
                    cfg, 
                    extractor.activation_func, 
                    model,
                    concept=None, 
                    concept_idx=None, 
                    disturb=disturb, 
                    measure_obj=measure_obj, 
                    corr_func='KL_div', 
                    class_idx=-1, 
                    logits_corr_topk=None,
                )
    
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
        
    })


    metric_evaluator = metric_evaluator_factory(cfg)
    metric_of_metrics = metric_evaluator.get_metric(
        tokens, 
        evaluator_dict, 
        concepts=concepts[concept_idxs],
        concept_idxs=concept_idxs,
        origin_tokens=origin_tokens,

    )
    print('metric_of_metrics:\n',metric_of_metrics)
    
if __name__ == "__main__":
    print('Hello guys!')
    main()