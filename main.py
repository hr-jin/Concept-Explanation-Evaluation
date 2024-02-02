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
    
        
    # cfg['data_dir'] = "/root/autodl-tmp/haoran/SparseAE-pythia-pile/data/pile_neel/"
    # cfg['dataset_name'] = "pile-tokenized-10b"
    # cfg['extractor'] = "ae"
    # eval_data = dataset_factory(cfg)
    # dataloader = dataloader_factory(cfg, eval_data, model)
    
    # concept_idx = 2919
    # concept_idx = 0
    
    
    # concept_idxs = [3676, 2397, 1889,  884, 1949, 3966,   49, 3665,  350, 3936,  461,
    #    3627,  263, 4076,  208, 2797, 1802, 3541, 3604, 2762, 2657, 1676,
    #    3336, 3875, 3307, 3798, 1842, 1571, 2749,  695, 2130, 1003, 3749,
    #    1066, 2784, 1437, 1537, 2084, 3038, 3455, 1484, 1884, 2110, 2582,
    #    1662, 3103, 3583,  516,  388, 3809, 2897, 4011, 1278, 1512, 1481,
    #    3897, 3509,  820,  859, 3406, 2907, 2178, 3064, 2583, 1394, 1474,
    #    2831,  515, 3763, 1081,  125, 3969, 1837,  514,  113, 2250, 3047,
    #    4089, 3877, 1823,  758,  430, 1166, 1247, 1739, 2555, 1141, 2993,
    #     560, 2021,  323, 3220, 3500, 3712, 2687, 2799, 2578, 1290, 1404,
    #    4057]
    
    # concept_idxs = [2841, 1411, 1284, 3689, 3863, 1986,  357,  366,  754, 1073, 4078,
    #    2217, 2245,   29, 1801,  236,  641,   31, 4019, 2208, 2809, 2398,
    #    2723, 3849,  565, 1733, 3429, 2695,  627, 2007, 2872,  668, 2690,
    #    1699,   19, 1408, 3003, 1448,  347, 2345, 1516, 2276, 1623, 3010,
    #    1093,  979, 2361, 1159,  258, 1558, 2423,  601, 1720, 2573, 2560,
    #    1149, 3850, 1249,  289,   20, 2662, 4069, 3395, 3562, 2828, 2352,
    #    3342, 1727, 3883, 1852, 2142, 3922, 1826, 2192, 2337, 1771, 4053,
    #      47, 2864, 1108,  453,  418, 1932, 1893, 2473,  855, 1075,  255,
    #    1144, 2513, 2808, 1131,  247, 2703,  513, 3149, 3909, 1566,   52,
    #    1256]
    
    concept_idxs = [2538, 3986, 3300, 3681,  531, 2384, 3914,  998,  636,  791, 2970,
       1651, 3295, 1235, 1389, 3137,  500,  160, 1520, 4010, 3673, 2663,
       3311, 3786, 1113,  404, 2852, 1115, 1114, 1337, 2474, 2306,  687,
       1638, 3536, 1528, 1882,  765, 3566, 1622, 2708, 1724, 1139, 3051,
        691, 2859,  288, 3145,  760, 2070, 2042,  442, 2246, 3484, 2227,
       1280, 3113, 1024, 2331,  443, 3974, 1352, 1489, 1098, 3449, 2999,
       1241, 3942, 2259, 2129,  718, 2992, 1106, 3352, 2874, 2120, 2545,
        910, 2660, 3411,   38, 3896, 2568, 3065, 2174, 2078, 3574, 3095,
       2189, 1104, 2514, 3150,  880, 2697, 3186, 1557, 3077,  779, 3872,
       3769]
    
    concept_idxs = concept_idxs[0:3] + concept_idxs[20:23] + concept_idxs[40:43]
    # concept_idxs = concept_idxs[0:7] + concept_idxs[20:27] + concept_idxs[40:47]
    # concept_idxs = concept_idxs[0:20]
    # concept_idxs = concept_idxs[20:40]
    # concept_idxs = concept_idxs[40:60]
        
    token_list = []
    for _ in range(5):
        tokens = dataloader.get_processed_random_batch()
        token_list.append(tokens)
    tokens = torch.cat(token_list, 0)
    
    evaluator_dict = dict()
    
    # cfg['evaluator'] = 'faithfulness'
    # for disturb in ['gradient','replace','ablation','replace-ablation']: # 'gradient','replace','ablation','replace-ablation'
    #     for measure_obj in ['pred_logit','logits', 'loss']: # 'pred_logit','logits', 'loss'
    #         if measure_obj == 'logits':
    #             for corr_func in ['KL_div', 'pearson']:
    #                 if disturb == 'gradient':
    #                     continue
    #                 for topk in [1000]:
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
    #                 corr_func='pearson', # ['pearson', 'KL_div', 'openai_var']
    #                 class_idx=7000, 
    #                 logits_corr_topk=None,
    #             )
                
    
        
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
    
    concept_idxs = concept_idxs
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
            print('idx:', concept_idx, ' ', name, ':', metric) 
    
    table = PrettyTable()
    table.title = 'Metrics Info'
    table.field_names = ['Metric'] + [str(i) for i in concept_idxs]

    
    for concept_idx in concept_idxs: 
        print('idx:', concept_idx, ' ,itc tokens: ', itc_critical_tokens_dict[concept_idx], ' ,otc tokens: ', otc_critical_tokens_dict[concept_idx])
    for name, evaluator in evaluator_dict.items():
        metric_arr = np.array(metric_dict[name])
        metric_arr = list(zip(np.argsort(-metric_arr), [i for i in range(len(metric_arr))]))
        metric_arr.sort(key=lambda item:item[0])
        table.add_row([name] + [x[1] for x in metric_arr])
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