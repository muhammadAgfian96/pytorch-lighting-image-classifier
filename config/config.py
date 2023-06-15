from config.list_optimizer import ListOptimizer

args_data = {
    "ratio_train": 0.8,
    "ratio_val": 0.2,
    "ratio_test": 0.0
}

args_train = {
    "epoch": 10,
    "batch": -1,
    "optimizer": ListOptimizer.AdamW,
    "weight_decay": 0,
    "momentum": 0.9,
    "lr": 1.0e-3,
    "lr_scheduler": "reduce_on_plateau", # step/multistep/reduce_on_plateau
    "lr_step_size": 7,
    "lr_decay_rate": 0.5,
    "precision": 16
}

args_model = {
    "input_size": 224,
    "architecture": "",
    "dropout": 0.0
}

args_custom = {
    "tags_exclude": [],
}