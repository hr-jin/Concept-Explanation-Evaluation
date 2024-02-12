from .base import BaseModel
from transformer_lens import HookedTransformer
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import torch

class Llama2Chat7B(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.load_model()
    
    @classmethod
    def code(cls):
        return 'llama-2-7b-chat'
    
    def load_model(self):
        model_name = self.cfg['model_to_interpret']
        model_path = self.cfg['model_dir']
        device = self.cfg['device']
        if model_path != '':
            model = HookedTransformer.from_pretrained(model_path).to(device)
        else:
            model = HookedTransformer.from_pretrained(model_name).to(device)
        self.model = model
        
    def load_model(self):
        model_name = self.cfg['model_to_interpret']
        model_path = self.cfg['model_dir']
        device = self.cfg['device']
        n_devices = self.cfg['n_devices']
        device_list = self.cfg['device_list'].split(',')
        if len(device_list) > 0:
            device_list = [int(d) for d in device_list if d != '']
        
        if 'llama' in model_name.lower():
            config = LlamaConfig.from_pretrained(model_path)
            with init_empty_weights():
                model_config = LlamaForCausalLM._from_config(config) 
            
            device_map = {}
            if len(device_list) != 0:
                device_list = [('cuda:' + str(i)) for i in device_list]
            elif device == 'cpu':
                device_list = ['cpu']
            else:
                start_idx = int(device.split(':')[-1])
                device_list = [('cuda:' + str(i)) for i in range(start_idx, start_idx + n_devices)]
                
            print('device_list:', device_list)
            
            if len(device_list) > 1:
                device_map['model.embed_tokens.weight'] = device_list[0]
                device_map['model.norm.weight'] = device_list[-1]
                device_map['lm_head.weight'] = device_list[-1]
                for i in range(32):
                    device_map['model.layers.'+str(i)+'.self_attn'] = device_list[i // (32 // len(device_list) + 1)]
                    device_map['model.layers.'+str(i)+'.mlp'] = device_list[i // (32 // len(device_list) + 1)]
                    device_map['model.layers.'+str(i)+'.input_layernorm'] = device_list[i // (32 // len(device_list) + 1)]
                    device_map['model.layers.'+str(i)+'.post_attention_layernorm'] = device_list[i // (32 // len(device_list) + 1)]
                
                hf_model = load_checkpoint_and_dispatch(
                    model_config, checkpoint=model_path, device_map=device_map, dtype=torch.float16
                )
                hf_model.eval()
                hf_model.requires_grad_(True)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                

                self.model = HookedTransformer.from_pretrained(model_name,
                                                        n_devices=n_devices,
                                                        device=device,
                                                        torch_dtype=torch.float16,
                                                        fold_value_biases=False,
                                                        fold_ln=False, 
                                                        center_writing_weights=False, 
                                                        center_unembed=False, 
                                                        hf_model=hf_model, 
                                                        tokenizer=tokenizer)
                
                self.tokenizer.add_special_tokens({'pad_token': '<unk>'})
                self.model.tokenizer.add_special_tokens({'pad_token': '<unk>'})
            else:
                hf_model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
                hf_model.eval()
                hf_model.requires_grad_(True)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = HookedTransformer.from_pretrained(model_name,
                                                        n_devices=n_devices,
                                                        device=device,
                                                        torch_dtype=torch.float16,
                                                        dtype='float16',
                                                        fold_value_biases=False,
                                                        fold_ln=False, 
                                                        center_writing_weights=False, 
                                                        center_unembed=False, 
                                                        hf_model=hf_model, 
                                                        tokenizer=tokenizer)
                
                self.model.tokenizer.add_special_tokens({'pad_token': '<unk>'})