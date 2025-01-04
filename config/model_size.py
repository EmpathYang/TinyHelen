from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer
)
import os

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

def compute_model_size(config):
    if config.model_type != "mamba":
        model = AutoModel.from_config(config).to("cuda")
    else:
        model = MambaLMHeadModel(config).to("cuda")
    net_n_params = sum({p.data_ptr(): p.numel() for n, p in model.named_parameters() if all(emb not in n for emb in ["emb", "shared", "lm_head"])}.values())
    n_params = sum({p.data_ptr(): p.numel() for n, p in model.named_parameters() if all(emb not in n for emb in ["lm_head"])}.values())
    return net_n_params/2**20, n_params/2**20

def find_param_to_approximate_model_size(config, model_size):
    
    config.num_hidden_layers = 32

    low = 8
    high = 4096

    while low <= high:
        mid = (low + high) // 2
        config.hidden_size = mid
        config.intermediate_size = 4 * config.hidden_size
        mid_value = compute_model_size(config)[0]
        print(mid, 4* mid, mid_value)
        if mid_value == model_size:
            return mid 
        
        if mid_value < model_size:
            low = mid + 1
        else:
            high = mid - 1
    
    return low

if __name__ == "__main__":
    model_family_list = ["bert", "llama", "xlnet", "mamba"]
    model_config_filename_list = ["bert-2K-10M.json", "llama-2K-10M.json", "xlnet-2K-10M.json", "mamba-2K-10M.json"]
    for model_family, model_config_filename in zip(model_family_list, model_config_filename_list):
        config_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_family, model_config_filename)
        config = AutoConfig.from_pretrained(config_name)
        print("*"*32+model_family.upper()+"*"*32)
        net_n_params, n_params = compute_model_size(config=config)
        print(f"Net Size = {net_n_params:.2f}M params\nTotal Size = {n_params:.2f}M params")
