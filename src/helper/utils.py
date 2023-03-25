from config.default import TrainingConfig


def override_config(
        params:dict, 
        conf:TrainingConfig
    ):
    conf.aug.augmentor = params['aug']['augmentor']
    conf.aug.type_executions = params['aug']['type_executions']
    conf.data.batch = params['data']['batch']
    conf.data.dir = params['data']['dir']
    conf.data.dataset = params['data'].get('dataset', conf.data.dataset)
    conf.data.input_resize = params['data']['input_resize']
    conf.data.input_size = params['data']['input_size']
    conf.data.random_seed = params['data']['random_seed']
    conf.data.test_ratio = params['data']['test_ratio']
    conf.data.train_ratio = params['data']['train_ratio']
    conf.data.val_ratio = params['data']['val_ratio']
    # conf.db.bucket_dataset = params['db']['bucket_dataset']
    # conf.db.bucket_experiment = params['db']['bucket_experiment']
    conf.hyp.epoch = params['hyp']['epoch']
    conf.hyp.base_learning_rate = params['hyp']['base_learning_rate']
    conf.hyp.lr_scheduler = params['hyp']['lr_scheduler']
    conf.hyp.lr_step_size = params['hyp']['lr_step_size']
    conf.hyp.lr_decay_rate = params['hyp']['lr_decay_rate']
    # conf.hyp.lr_step_milestones = params['hyp']['lr_step_milestones']
    conf.hyp.opt_momentum = params['hyp']['opt_momentum']
    conf.hyp.opt_name = params['hyp']['opt_name']
    conf.hyp.opt_weight_decay = params['hyp']['opt_weight_decay']
    conf.hyp.precision = params['hyp']['precision']
    conf.net.architecture = params['net']['architecture']
    conf.net.dropout = params['net']['dropout']
    conf.net.checkpoint_model = params['net']['checkpoint_model']
    conf.net.pretrained = params['net']['pretrained']
    conf.net.resume = params['net']['resume']
    # if not args_pipeline['inside_pipeline']:
    #     conf.data.category = sorted(os.listdir(conf.data.dir))
    #     conf.net.num_class = len(os.listdir(conf.data.dir))
    return conf
