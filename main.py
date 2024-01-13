import os
import argparse
from transformer_lens import HookedTransformer
from utils import *
from extractor import *
from dataloader import *
from config import cfg as default_cfg

def main():
    """
    Here we take the process of concept extraction by autoencoder as an example.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_to_interpret", type=str, default="pythia-70m")
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/haoran/SparseAE-pythia-pile/data/") 
    parser.add_argument("--dataset_name", type=str, default="pile_neel/pile-tokenized-10b.hf")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/haoran/SparseAE-pythia-pile/codebase/data/output")
    parser.add_argument("--subname", type=str, required=True) 
    parser.add_argument("--model_dir", type=str, default="/root/autodl-tmp/haoran/SparseAE-pythia-pile/data/cac/models--EleutherAI--pythia-70m/snapshots/a39f36b100fe8a5377810d56c3f4789b9c53ac42")
    
    
    cfg = default_cfg
    cfg, args = arg_parse_update_cfg(cfg, parser)

    model_to_interpret = HookedTransformer.from_pretrained(args.model_dir).to(args.device) # to(DTYPES["fp32"])
    
    cfg = process_cfg(cfg, model_to_interpret)
    
    save_path = f"{args.subname}_model_{args.model_to_interpret}_layer_{cfg['layer']}_dictSize_{cfg['dict_size']}_site_{cfg['site']}"
    save_dir = os.path.join(args.output_dir, save_path)
    os.makedirs(save_dir, exist_ok=True)
    
    # define dataloader
    dataloader = AE_Dataloader(cfg=cfg, data=load_dataset(args.data_dir, args.dataset_name), model=model_to_interpret)
    
    # define concept extractor
    extractor = AutoEncoder(cfg).to(cfg['device'])
    
    print(json.dumps(cfg, indent=2))
    print(model_to_interpret.cfg.device)
    
    concepts = extractor.extract_concepts(model_to_interpret, 
                                          dataloader, 
                                          save_dir) #, train_loader)
    
    print(concepts)
    
if __name__ == "__main__":
    main()