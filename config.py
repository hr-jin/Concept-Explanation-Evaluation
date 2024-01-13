# Parameters written as none will be automatically set in the code, 
# and all properties can be modified through command line parameters.
# e.g. --dict_mult 2
cfg = {
        ## autoencoder
        "dict_mult": 8,
        "d_mlp": None,
        "d_model": None,
        "enc_dtype":"fp32",
        
        ## buffer
        "buffer_size": None,
        "buffer_mult": 400, # to modify
        "act_size": None,
        "buffer_batches": None,
        "model_batch_size":64,
        
        "seed": 49,
        
        ## dataset
        "num_tokens": int(1363348000), #
        "seq_len": 128,
        
        ## subject model setting
        "layer": 0,
        "site": "resid_post",
        "layer_type": None,
        'name_only': 0, 
        
        ## optimizer
        "beta1": 0.9,
        "beta2": 0.99,
        "lr": 1e-3,
        "l1_type": 'default',
        "l2_type": 'KL',

        ## train
        "num_batches": None,
        "device": "cuda:0",
        "batch_size": 8192,
        "l1_coeff": 1.0,
        "epoch": 1,
        "reinit": 1,
        'init_type': 'xavier_uniform',
        'remove_parallel': 1,
        
    }