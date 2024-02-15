from transformer_lens import utils
import torch
import datasets
import numpy as np
import random
import os
from logger import logger


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
def run_with_cache_onesentence(
        *model_args,
        model,
        names_filter,
        device=None,
        remove_batch_dim=False,
        incl_bwd=True,
        reset_hooks_end=True,
        clear_contexts=False,
        seq_len=0,
        logit_token_idx=-1,
        concept_act=None,
        **model_kwargs,
    ):
    cache_dict, fwd, bwd = model.get_caching_hooks(
        names_filter, incl_bwd, device, remove_batch_dim=remove_batch_dim
    )
    
    with model.hooks(
        fwd_hooks=fwd,
        bwd_hooks=bwd,
        reset_hooks_end=reset_hooks_end,
        clear_contexts=clear_contexts,
    ):
        model_out = model(*model_args, **model_kwargs)
        last_token_logit = model_out[0, torch.argmax(concept_act), :] 
        if logit_token_idx == -1:
            value, logit_token_idx = torch.topk(last_token_logit, k=1)
        if incl_bwd:
            last_token_logit[logit_token_idx].backward()
            
    return model_out, cache_dict

def run_with_cache_top1logit_bkwd(
        tokens,
        *model_args,
        model,
        names_filter,
        device=None,
        remove_batch_dim=False,
        incl_bwd=True,
        reset_hooks_end=True,
        clear_contexts=False,
        seq_len=0,
        cfg=None,
        logit_token_idx=-1,
        concept_act=None,
        **model_kwargs,
    ):
    cache_dict, fwd, bwd = model.get_caching_hooks(
        names_filter, incl_bwd, device, remove_batch_dim=remove_batch_dim
    )
    
    with model.hooks(
        fwd_hooks=fwd,
        bwd_hooks=bwd,
        reset_hooks_end=reset_hooks_end,
        clear_contexts=clear_contexts,
    ):
        model_out = model(tokens, *model_args, **model_kwargs)
        if logit_token_idx == -1:
            value, _ = torch.topk(model_out, k=1, dim=-1) 
            
        if logit_token_idx == -2:
            true_next_indices = tokens[:,1:].clone().detach().to(cfg['device'])
            value = torch.gather(model_out[:,:-1,:], dim=-1, index=true_next_indices.unsqueeze(-1)).squeeze()
        if incl_bwd:
            (value).sum().backward()
            
    return model_out, cache_dict
    
def load_dataset(data_dir, dataset_name, data_from_hf):
    if data_from_hf:
        if not os.path.exists(data_dir + dataset_name + '.hf'):
            data = datasets.load_dataset(dataset_name, split="train", cache_dir=data_dir)
            data.save_to_disk(os.path.join(data_dir, dataset_name + '.hf'))
        data = datasets.load_from_disk(data_dir + dataset_name + '.hf')
    else:
        ...
    return data


def post_init_cfg(cfg):
    cfg["model_batch_size"] = cfg["batch_size"] // cfg["seq_len"]
    cfg["buffer_size"] = cfg["batch_size"] * cfg["buffer_mult"]
    cfg["buffer_batches"] = cfg["buffer_size"] // cfg["seq_len"]
    if cfg['name_only']:
        cfg["act_name"] = cfg["site"]
    else:
        cfg["act_name"] = utils.get_act_name(cfg["site"], cfg['layer'], cfg['layer_type'])
    cfg["dict_size"] = cfg["act_size"] * cfg["dict_mult"]
    return cfg

def arg_parse_update_cfg(default_cfg, parser):
    """
    Helper function to take in a dictionary of arguments, 
        convert these to command line arguments, look at what was passed in, and return an updated dictionary.
    """
    cfg = dict(default_cfg)
    for key, value in default_cfg.items():
        if key == 'extractor':
            parser.add_argument(f"--{key}", choices=["ae", "tcav", "neuron", "conceptx", "conceptx_ori"], default="ae")
        elif key == 'model_to_interpret':
            parser.add_argument(f"--{key}", choices=["llama-2-7b-chat", "pythia-70m", 'gpt2-small'], default="pythia-70m")
        elif type(value) == bool:
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
    
    save_path = f"model_{cfg['model_to_interpret']}_layer_{cfg['layer']}_dictSize_{cfg['dict_size']}_site_{cfg['site']}"
    save_dir = os.path.join(cfg['output_dir'], save_path)
    os.makedirs(save_dir, exist_ok=True)
    cfg['save_dir'] = save_dir
    
    logger.info("Updated config")
    return cfg
