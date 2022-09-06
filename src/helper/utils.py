from config.default import TrainingConfig


def override_config(params:dict, conf:TrainingConfig):
    conf.aug.augmentor = params['aug']['augmentor']
    conf.aug.type_executions = params['aug']['type_executions']
    conf.data.batch = params['data']['batch']
    conf.data.dir = params['data']['dir']
    conf.data.input_resize = params['data']['input_resize']
    conf.data.input_size = params['data']['input_size']
    conf.data.random_seed = params['data']['random_seed']
    conf.data.test_ratio = params['data']['test_ratio']
    conf.data.train_ratio = params['data']['train_ratio']
    conf.data.val_ratio = params['data']['val_ratio']
    conf.db.bucket_dataset = params['db']['bucket_dataset']
    conf.db.bucket_experiment = params['db']['bucket_experiment']
    conf.hyp.epoch = params['hyp']['epoch']
    conf.hyp.learning_rate = params['hyp']['learning_rate']
    conf.hyp.opt_momentum = params['hyp']['opt_momentum']
    conf.hyp.opt_name = params['hyp']['opt_name']
    conf.hyp.opt_weight_decay = params['hyp']['opt_weight_decay']
    conf.net.architecture = params['net']['architecture']
    conf.net.checkpoint_model = params['net']['checkpoint_model']
    conf.net.num_class = params['net']['num_class']
    conf.net.pretrained = params['net']['pretrained']
    conf.net.resume = params['net']['resume']
    return conf
