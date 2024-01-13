from transformer_lens import utils
import torch
import json
import datasets
import numpy as np
import random
import os
import torch.nn.functional as F

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
def load_dataset(data_dir, dataset_name):
    if not os.path.exists(data_dir + dataset_name):
        data = datasets.load_dataset("NeelNanda/pile-tokenized-10b", split="train", cache_dir=data_dir)
        data.save_to_disk(os.path.join(data_dir, dataset_name))
    data = datasets.load_from_disk(data_dir + dataset_name)
    return data


def post_init_cfg(cfg):
    cfg["model_batch_size"] = cfg["batch_size"] // cfg["seq_len"] # * 16
    cfg["buffer_size"] = cfg["batch_size"] * cfg["buffer_mult"]
    cfg["buffer_batches"] = cfg["buffer_size"] // cfg["seq_len"]
    cfg["act_name"] = utils.get_act_name(cfg["site"], cfg['layer'], cfg['layer_type'], cfg['name_only'])
    cfg["dict_size"] = cfg["act_size"] * cfg["dict_mult"]
    return cfg

def arg_parse_update_cfg(default_cfg, parser):
    """
    Helper function to take in a dictionary of arguments, convert these to command line arguments, look at what was passed in, and return an updated dictionary.
    """
    cfg = dict(default_cfg)
    for key, value in default_cfg.items():
        if type(value) == bool:
            if value:
                parser.add_argument(f"--{key}", action="store_false")
            else:
                parser.add_argument(f"--{key}", action="store_true")
        elif value is None:
            continue
        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)
    args = parser.parse_args()
    parsed_args = vars(args)
    cfg.update(parsed_args)
    
    return cfg, args

def process_cfg(cfg, model_to_interpret):
    d_model = model_to_interpret.cfg.d_model
    d_mlp = model_to_interpret.cfg.d_mlp
    d_model = model_to_interpret.cfg.d_model
    
    cfg['d_mlp'] = d_mlp
    cfg['d_model'] = d_model
    
    if cfg['site'] == 'mlp_post':
        cfg["dict_size"] = cfg["dict_mult"] * cfg["d_mlp"]
        cfg["act_size"] = cfg["d_mlp"]
    else:
        cfg["dict_size"] = cfg["dict_mult"] * cfg["d_model"]
        cfg["act_size"] = cfg["d_model"]
        
    cfg["num_batches"] = cfg["num_tokens"] // cfg["batch_size"] 
    cfg = post_init_cfg(cfg)
    
    print("Updated config")
    
    return cfg