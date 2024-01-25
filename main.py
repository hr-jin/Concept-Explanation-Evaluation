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
    
    concept_idx = 2919
        
    # # readability
    # evaluator = evaluator_factory(
    #     cfg, 
    #     extractor.activation_func, 
    #     model,
    #     concept=concepts[concept_idx], 
    #     concept_idx=concept_idx, 
    #     pmi_type='silhouette'
    # )
    
    # faithfulness
    # evaluator = evaluator_factory(
    #     cfg, 
    #     extractor.activation_func, 
    #     model,
    #     concept=concepts[concept_idx], 
    #     concept_idx=concept_idx, 
    #     disturb='replace', # ['ablation', 'gradient', 'replace']
    #     measure_obj='logits', # ['loss', 'class_logit', 'logits']
    #     corr_func='KL_div', # ['cosine', 'KL_div', 'openai_var']
    #     class_idx=7000, 
    #     logits_corr_topk=10
    # )
    
    token_list = []
    for _ in range(5):
        tokens = dataloader.get_processed_random_batch()
        token_list.append(tokens)
    tokens = torch.cat(token_list, 0)
    
    # metric = evaluator.get_metric(tokens)
    metric_evaluator = metric_evaluator_factory(cfg)
    # evaluator_dict = {
    #     'replace_logits_KL_div': evaluator_factory(
    #         cfg, 
    #         extractor.activation_func, 
    #         model,
    #         concept=concepts[concept_idx], 
    #         concept_idx=concept_idx, 
    #         disturb='replace', # ['ablation', 'gradient', 'replace']
    #         measure_obj='logits', # ['loss', 'class_logit', 'logits']
    #         corr_func='KL_div', # ['cosine', 'KL_div', 'openai_var']
    #         class_idx=7000, 
    #         logits_corr_topk=10,
    #     ),
    #     'ablation_logits_KL_div': evaluator_factory(
    #         cfg, 
    #         extractor.activation_func, 
    #         model,
    #         concept=concepts[concept_idx], 
    #         concept_idx=concept_idx, 
    #         disturb='ablation', # ['ablation', 'gradient', 'replace']
    #         measure_obj='logits', # ['loss', 'class_logit', 'logits']
    #         corr_func='KL_div', # ['cosine', 'KL_div', 'openai_var']
    #         class_idx=7000, 
    #         logits_corr_topk=10,
    #     ),
    #     'replace_loss_KL_div': evaluator_factory(
    #         cfg, 
    #         extractor.activation_func, 
    #         model,
    #         concept=concepts[concept_idx], 
    #         concept_idx=concept_idx, 
    #         disturb='replace', # ['ablation', 'gradient', 'replace']
    #         measure_obj='loss', # ['loss', 'class_logit', 'logits']
    #         corr_func='KL_div', # ['cosine', 'KL_div', 'openai_var']
    #         class_idx=7000, 
    #         logits_corr_topk=10,
    #     ),
    #     'ablation_loss_KL_div': evaluator_factory(
    #         cfg, 
    #         extractor.activation_func, 
    #         model,
    #         concept=concepts[concept_idx], 
    #         concept_idx=concept_idx, 
    #         disturb='ablation', # ['ablation', 'gradient', 'replace']
    #         measure_obj='loss', # ['loss', 'class_logit', 'logits']
    #         corr_func='KL_div', # ['cosine', 'KL_div', 'openai_var']
    #         class_idx=7000, 
    #         logits_corr_topk=10,
    #     ),
    # }
    evaluator_dict = {
        'itc_silhouette': evaluator_factory(
            cfg, 
            extractor.activation_func, 
            model,
            concept=concepts[concept_idx], 
            concept_idx=concept_idx, 
            pmi_type='silhouette', # ['uci', 'umass', 'silhouette']
        ),
        'itc_uci': evaluator_factory(
            cfg, 
            extractor.activation_func, 
            model,
            concept=concepts[concept_idx], 
            concept_idx=concept_idx, 
            pmi_type='uci', # ['uci', 'umass', 'silhouette']
        ),
        'itc_umass': evaluator_factory(
            cfg, 
            extractor.activation_func, 
            model,
            concept=concepts[concept_idx], 
            concept_idx=concept_idx, 
            pmi_type='umass', # ['uci', 'umass', 'silhouette']
        ),
    }
    metric_of_metrics = metric_evaluator.get_metric(
        tokens, 
        evaluator_dict, 
        concepts=concepts[concept_idx-20:concept_idx+10],
        concept_idxs=[i for i in range(concept_idx-20,concept_idx+10)]
    )
    
    print('metric_of_metrics:\n',metric_of_metrics)
    
    
if __name__ == "__main__":
    main()