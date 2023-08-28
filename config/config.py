from config.list_optimizer import ListOptimizer

args_data = {
    "ratio_train": 0.8,
    "ratio_val": 0.1,
    "ratio_test": 0.0,
    "augment_type": "custom",
    "augment_preset_train": {
        "hflip_prob": 0.8,
        "ra_magnitude": 9,
        "augmix_severity": 3,
        "random_erase_prob": 0.0
    },

}

args_train = {
    "epoch": 100,
    "batch": 64,
    "optimizer": "adamw",
    "weight_decay": 0,
    "momentum": 0.9,
    "lr": 1.0e-3,
    "lr_scheduler": "reduce_on_plateau", # step/multistep/reduce_on_plateau
    "lr_step_size": 7,
    "lr_decay_rate": 0.5,
    "precision": 16,
    "early_stopping": {
        "patience": 15,
        "min_delta": 0.01,
        "mode": "max",
        "monitor": "val_acc"
    },
    "tuner": {
        "batch_size": True,
        "learning_rate": False
    }
}

args_model = {
    "input_size": 224,
    # "architecture": "timm/tf_mobilenetv3_small_minimal_100.in1k",
    "architecture": "tf_efficientnet_b1",
    "dropout": 0.2,
    "resume": ""
}

args_custom = {
    "tags_exclude": ["drop"],
    "mode": "training",
    # "tags_to_class": {
    #     "keep": ["keep"],
    #     "drop": ["drop", "background", "skip", "blur", "remove"]
    # }
}