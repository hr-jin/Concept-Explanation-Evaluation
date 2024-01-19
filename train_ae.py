import os
import argparse
from utils import *
from extractor import *
from dataloader import *
from config import cfg as default_cfg
import logging

def main():
    """
    Here we take the process of concept extraction by autoencoder as an example.
    """
    
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO, force=True)
    logger = logging.getLogger('logger')

    parser = argparse.ArgumentParser()
    cfg, args = arg_parse_update_cfg(default_cfg, parser)

    model_to_interpret = load_model(args)
    
    
    print('model_to_interpret.cfg:',model_to_interpret.cfg)
    logger.info('loaded model...')
    
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
    logger.info(model_to_interpret.cfg.device)
    
    logger.info('extract concepts...')
    concepts = extractor.extract_concepts(model_to_interpret, 
                                          dataloader, 
                                          save_dir) #, train_loader)
    
    logger.info('finished extract concepts...')
    
    print(concepts)
    
if __name__ == "__main__":
    main()