import os
import argparse
from transformer_lens import HookedTransformer
from utils import *
from extractor import *
from dataloader import *
from config import cfg as default_cfg
import pandas as pd

def main():
    """
    Here we take the process of concept extraction by autoencoder as an example.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_to_interpret", type=str, default="Llama-2-7b-chat-hf")
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/haoran/SparseAE-pythia-pile/data/") 
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/haoran/SparseAE-pythia-pile/codebase/data/output")
    parser.add_argument("--subname", type=str, required=True) 
    parser.add_argument("--model_dir", type=str, default='/root/autodl-tmp/haoran/SparseAE-pythia-pile/data/Llama-2-7b-chat-hf')
    parser.add_argument("--data_from_hf", type=bool, default=True)
    
    cfg = default_cfg
    cfg, args = arg_parse_update_cfg(cfg, parser)

    model_to_interpret = load_model(args, device_list=[1,2,3,5,6,7])
    
    cfg = process_cfg(cfg, model_to_interpret)
    
    save_path = f"{args.subname}_model_{args.model_to_interpret}_layer_{cfg['layer']}_dictSize_{cfg['dict_size']}_site_{cfg['site']}"
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
    print(model_to_interpret.cfg.device)
    
    cavs, acc = extractor.extract_concepts(dataloader) #, train_loader)
    
    # print('cavs:', cavs)
    
    evaluator = TCAV_Evaluator(cfg, model_to_interpret, cavs, logit_token_idx=7423)
    tcavs_score, positive_mean_effects, negative_mean_effects = evaluator.get_tcav_score(random_input_examples[-100:])
    
if __name__ == "__main__":
    main()