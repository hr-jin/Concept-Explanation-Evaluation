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
    
    # concept_idxs = [2538, 3986, 3300, 3681,  531, 2384, 3914,  998,  636,  791, 2970,
    #    1651, 3295, 1235, 1389, 3137,  500,  160, 1520, 4010, 3673, 2663,
    #    3311, 3786, 1113,  404, 2852, 1115, 1114, 1337, 2474, 2306,  687,
    #    1638, 3536, 1528, 1882,  765, 3566, 1622, 2708, 1724, 1139, 3051,
    #     691, 2859,  288, 3145,  760, 2070, 2042,  442, 2246, 3484, 2227,
    #    1280, 3113, 1024, 2331,  443, 3974, 1352, 1489, 1098, 3449, 2999,
    #    1241, 3942, 2259, 2129,  718, 2992, 1106, 3352, 2874, 2120, 2545,
    #     910, 2660, 3411,   38, 3896, 2568, 3065, 2174, 2078, 3574, 3095,
    #    2189, 1104, 2514, 3150,  880, 2697, 3186, 1557, 3077,  779, 3872,
    #    3769]
    
    # 200 1
    concept_idxs = [3676, 1647, 3046, 2810, 1519,  763, 3264,  884,  740, 2036,   24,       3599, 3610,  310,   49,  217, 2594,  405, 3431,  978,  619, 3936,       3116, 2600, 3778,   81, 2877,  413,  263,  496,  982, 1583, 3161,        497,  843, 2797,  379, 3253,  329, 2541, 1863, 1956, 3604, 1996,        316, 3326,  545,  519, 1092, 1676, 2087,  265, 2035, 2426, 2027,         28, 3307, 4073,  448, 3992, 1032, 1398,  947, 1571,  750, 1022,       2162, 1792, 1717,  259, 2130,  137,  966, 3545, 1529, 2851,  956,       1066,  484,  375, 1297,  540, 3349, 2472, 1537, 3328, 3510, 1891,       3820, 2972, 2030, 3455, 3323, 1160, 1774, 2308,  449, 3059, 2110,       3002,  885, 3029, 2654, 2634, 3699, 3103,  773, 2400,  147, 2535,       2074, 3420,  388, 3503, 3847, 1661, 2981, 1017, 1246, 4011,  282,        660,  844, 3880, 3865, 1109, 1481, 1548, 2755, 1078, 1937, 1460,       3708,  820, 2277, 2609, 1070,  200, 3978,  250, 2907, 1912,  414,       1035, 1669, 2616, 2040, 2583,  466, 2310, 1013, 3393, 2320,  254,       2831,  332, 3428,  337, 3709, 3473, 2734, 1081,   10, 1749, 2699,       2849, 3575, 1088, 1837, 1514, 2968, 1356, 3343, 4080, 3104, 2250,        693,  926, 1706,  106, 2791, 1868, 3877, 2858, 2177, 2286, 2436,       3758, 1363,  430, 2532, 1903, 2597,  616, 2682, 1014, 1739,  244,       2700, 2137]
    
    # # 200 2
    # concept_idxs = [2841,  338, 1344, 1997, 2641, 2866, 3344, 3689, 2889, 3066, 3233,3005, 3550, 4091,  357, 1215, 1983, 1921, 3669, 3912, 3450, 1073,       3076, 3851, 2735, 1944,  590, 3961, 2245, 3488, 4040, 2540, 3845,       1148, 1347,  236, 2656, 2257, 3624, 3976, 1927, 2926, 4019,  336,       1829, 1812, 2658, 3134, 4003, 2398, 1180, 1624, 2904, 3477, 2350,       4043,  565, 2343, 1813, 2586, 2832, 2888, 1594, 2695, 2590,  711,        729, 1498,  297, 2763, 2872, 1916, 3324, 3413,  159,  633,  389,       1699, 2258,  856, 2413,  700, 1691, 1138, 3003,   32,  213, 3836,        274, 1469, 1839, 2345, 1914,  114, 1834, 1090, 2191,  251, 1623,       1429, 3179, 3996,  650, 2857, 3959,  979, 2777, 2409,   71,  118,       3260, 2140,  258,  469, 3466,  386, 2528,  792, 2085,  601,  380,       1110, 1076, 3026,   18, 2642, 2560, 3119, 3422, 3848, 3199, 1705,       1029, 1249, 3442,  140, 2935, 2358,  925, 1502, 2662, 2124, 1058,        553, 3791, 3775, 3218, 3562, 2235, 3513,  814,  520, 1867, 2219,       3342,   35, 1918, 1378, 4083, 3554,  701, 1852, 1856, 3701, 3197,        187, 3340, 2925, 1826,  975, 3292, 1111, 1177, 2238, 2263, 1771,       2914,  262, 2048, 2248,  266,  811, 2864, 3740, 3405, 1643, 2938,       2133, 1419,  418,  411, 1784, 2816, 2121, 1374, 3024, 2473, 3153,        853, 2218]
    
    # # 200 3
    # concept_idxs = [2538,  706, 2903, 3829,  481,  747, 1414, 3681, 1616, 2633, 3052,       2099, 3693,  170, 3914, 3501, 1585, 1226,   99, 3892, 3594,  791,       1445, 2704,  858, 2197, 1525,  363, 3295,  977, 1930, 2372, 3028,       3748, 1061, 3137,  901, 1511, 1368, 2878, 2265, 2563, 1520, 2196,         17, 2901, 3090,   90, 2147, 2663, 3618, 1877, 2063,  799, 2821,       3074, 1113, 1034, 3214, 1260, 3390, 1574,  179, 1115, 1504, 1416,       3265, 1785,  557,  493, 2474,  790, 4006,   75, 1656,   70, 3947,       1638, 1958, 3421,  874,  959,  214, 1607, 1882, 3084, 1390, 1901,        345, 1966, 2517, 1622, 3631, 1055,  238,  696, 3184, 1364, 1139,       1642,  354,  862, 1874, 1096, 2822, 2859,  318, 3861,  882, 1917,       3573, 3247,  760, 3055, 2463, 1158, 4050, 3508, 1027,  442, 2684,        580, 3641, 1383, 1738,  312, 2227,  905, 2776,  498, 4020, 3529,       1701, 1024,  997, 2268, 1428, 3261, 2706, 3350, 3974,  270, 3020,        699, 2902, 2014, 3332, 1098, 1841,   12, 3514,  591, 2603, 2255,       1241,  424, 3039, 1708,  551, 1620, 2462, 2129,  981, 3788, 3062,        886, 2056,  732, 1106, 2579, 3945, 3989,  757, 2732, 2199, 2120,       3170,  678, 2607,  483, 2522, 2145, 2660,  185, 1204, 2939,  286,       1961, 2249, 3896,  197, 2052, 3940, 3387, 3057, 1721, 2174,  212,       3317, 2924]
    
    # most_imp_tokens = np.load(
    #     '/root/autodl-tmp/haoran/SparseAE-pythia-pile/codebase/data/output/vr_data/200concepts_uci5tokens_1_most_imp_tokens.npy',
    #     allow_pickle=True
    # )
    # most_imp_idxs = np.load(
    #     '/root/autodl-tmp/haoran/SparseAE-pythia-pile/codebase/data/output/vr_data/200concepts_uci5tokens_1_most_imp_idxs.npy',
    #     allow_pickle=True
    # )
    
    # concept_idxs = [i+0*128 for i in range(128)]
    # concept_idxs = [i+0*200 for i in range(200)]
    # concept_idxs = [i+0*250 for i in range(250)]
    # concept_idxs = [0]
    
    # concept_idxs = concept_idxs[0:7] + concept_idxs[20:27] + concept_idxs[40:47]
    # concept_idxs = concept_idxs[0:20]
    # concept_idxs = concept_idxs[20:40]
    # concept_idxs = concept_idxs[40:60]
        
    token_list = []
    origin_token_list = []
    for _ in range(5): # 之前把batch_size从32768改为8192，所以测ae concept要把5改回20，或者batch_size改回32768。
        tokens, origin_tokens = dataloader.get_processed_random_batch()
        token_list.append(tokens)
        origin_token_list.append(origin_tokens)
    tokens = torch.cat(token_list, 0)
    origin_tokens = torch.cat(origin_token_list, 0)
    
    evaluator_dict = dict()
    
    # cfg['evaluator'] = 'faithfulness'
    # for disturb in ['gradient','replace','ablation','replace-ablation']: # 'gradient','replace','ablation','replace-ablation'
    #     for measure_obj in ['loss','pred_logit','logits']: # 'pred_logit','logits', 'loss'
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
    
    # concept_idxs = concept_idxs
    # metric_dict = dict()
    # itc_critical_tokens_dict = dict()
    # otc_critical_tokens_dict = dict()
    # for name, evaluator in evaluator_dict.items():
    #     metric_dict[name] = []
    #     for concept_idx in concept_idxs:
    #         concept = concepts[concept_idx]
    #         evaluator.update_concept(concept, concept_idx) 
    #         metric, critical_tokens = evaluator.get_metric(tokens, return_tokens=True)
    #         if evaluator.code() == 'itc':
    #             itc_critical_tokens_dict[concept_idx] = critical_tokens
    #         elif evaluator.code() == 'otc':
    #             otc_critical_tokens_dict[concept_idx] = critical_tokens
    #         metric_dict[name].append(metric)
    #         print('idx:', concept_idx, ' ', name, ':', metric) 
    
    # table = PrettyTable()
    # table.title = 'Metrics Info'
    # table.field_names = ['Metric'] + [str(i) for i in concept_idxs]

    
    # for concept_idx in concept_idxs: 
    #     print('idx:', concept_idx, ' ,itc tokens: ', itc_critical_tokens_dict[concept_idx], ' ,otc tokens: ', otc_critical_tokens_dict[concept_idx])
    # for name, evaluator in evaluator_dict.items():
    #     metric_arr = np.array(metric_dict[name])
    #     metric_arr = list(zip(np.argsort(-metric_arr), [i for i in range(len(metric_arr))]))
    #     metric_arr.sort(key=lambda item:item[0])
    #     table.add_row([name] + [x[1] for x in metric_arr])
    # print(table)
        
        
    metric_evaluator = metric_evaluator_factory(cfg)
    metric_of_metrics = metric_evaluator.get_metric(
        tokens, 
        evaluator_dict, 
        concepts=concepts[concept_idxs],
        concept_idxs=concept_idxs,
        origin_tokens=origin_tokens,
        # most_imp_tokens=most_imp_tokens,
        # most_imp_idxs=most_imp_idxs,
    )
    
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