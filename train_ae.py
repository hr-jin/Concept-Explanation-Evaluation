import os
import argparse
from utils import *
from extractor import *
from dataloader import *
import importlib
from config import cfg as default_cfg


def main():
    """
    Here we take the process of concept extraction by autoencoder as an example.
    """
    parser = argparse.ArgumentParser()
    cfg, args = arg_parse_update_cfg(default_cfg, parser)
    

    model_to_interpret = load_model(args)
    
    print('model_to_interpret.cfg:',model_to_interpret.cfg)
    cfg = process_cfg(cfg, model_to_interpret)
    
    save_path = f"model_{args.model_to_interpret}_layer_{cfg['layer']}_dictSize_{cfg['dict_size']}_site_{cfg['site']}"
    save_dir = os.path.join(args.output_dir, save_path)
    os.makedirs(save_dir, exist_ok=True)
    
    # define dataloader
    loaded_data = load_dataset(args.data_dir, args.dataset_name, args.data_from_hf)
    dataloader = AE_Dataloader(cfg=cfg, data=loaded_data, model=model_to_interpret)
    
    # define concept extractor
    extractor = AutoEncoder(cfg).to(cfg['device'])
    
    print(json.dumps(cfg, indent=2))
    print(model_to_interpret.cfg.device)
    
    print('extract concepts...')
    concepts = extractor.extract_concepts(model_to_interpret, 
                                          dataloader, 
                                          save_dir) #, train_loader)
    
    print('finished extract concepts...')
    
    print(concepts)
    
if __name__ == "__main__":
    main()