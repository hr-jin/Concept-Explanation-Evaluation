import os
import argparse
from utils import *
from extractor import *
from dataloader import *
from config import cfg as default_cfg
import pandas as pd
import logging

def main():

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO, force=True)
    logger = logging.getLogger('logger')

    parser = argparse.ArgumentParser()
    
    cfg, args = arg_parse_update_cfg(default_cfg, parser)

    model_to_interpret = load_model(args, device_list=[1,2,3,5,6,7])
    
    cfg = process_cfg(cfg, model_to_interpret)
    
    save_path = f"model_{args.model_to_interpret}_layer_{cfg['layer']}_dictSize_{cfg['dict_size']}_site_{cfg['site']}"
    save_dir = os.path.join(args.output_dir, save_path)
    os.makedirs(save_dir, exist_ok=True)
    
    df = pd.read_csv('/root/autodl-tmp/haoran/SparseAE-pythia-pile/data/HarmfulQA.csv')
    harmful_concept_examples = df['question_1'].tolist()

    df = pd.read_csv('/root/autodl-tmp/haoran/SparseAE-pythia-pile/data/HarmfulQA.csv')
    negative_concept_examples = df['question_2'].tolist()

    df = pd.read_csv('/root/autodl-tmp/haoran/SparseAE-pythia-pile/data/TrustfulQA.csv')
    input_examples = df['Question'].tolist()
    random_input_examples = input_examples
    
    # define dataloader
    dataloader = TCAV_Dataloader(cfg=cfg, positive_examples=harmful_concept_examples, negative_examples=negative_concept_examples)
    
    # define concept extractor
    extractor = TCAV_Extractor(cfg, model=model_to_interpret)
    
    print(json.dumps(cfg, indent=2))
    logger.info(model_to_interpret.cfg.device)
    
    cavs, acc = extractor.extract_concepts(dataloader) #, train_loader)
        
    evaluator = TCAV_Evaluator(cfg, model_to_interpret, cavs, logit_token_idx=7420) # 7420(low) 7423(high)
    tcavs_score, positive_mean_effects, negative_mean_effects = evaluator.get_tcav_score(random_input_examples)
    
if __name__ == "__main__":
    main()