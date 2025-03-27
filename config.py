STRATEGY = {
    'train': {
        "batch_size": 64,
        "epochs": 10,
        "patience": 200,
        'train_size': None,
        "lr": 1e-3, 
        'weight_decay': 0,
        "threshold": 0.5,
        "CosineAnnealingLR": {"T_max": 10, 'eta_min':1e-6}, # CosineAnnealingLR learning rate scheduler
        "shuffle_splits": 1,
        "kfold_splits": 5,
    },
    'tune': {
        "batch_size": 64,
        # "batch_size": 8, # for Bacteria
        "patience": 50,
        'train_size': None,
        "optmizer": "Adam",
        "Adam_params": {"lr": 1e-5},
    },
    'test': {
        "batch_size": 32,
    }
}

NET = {
    'no_kfval':0,
    # Dropouts
    'attn_dropout': 0.1,
    'attn_dropout_s2':0.0,
    'attn_dropout_s3':0.0,
    'relu_dropout':0.1,
    'embed_dropout':0.25,
    'res_dropout':0.1,
    'out_dropout':0.0,
    'layers':6,
    'num_heads' :8,
    'attn_mask':0,
    'orig_d_s1': 900,
    'orig_d_s2': 993,
    'orig_d_s3': 999,
    'layers': 6,
    'nlevels': 6,
    'output_dim': 957,
    'd_s1':64,
    'd_s2':64,
    'd_s3':64,
    'use_crossmodal':True,
    'crossmodal_first':True,
}