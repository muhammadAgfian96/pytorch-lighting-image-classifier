from clearml import StorageManager, Task
from config.default import TrainingConfig
import yaml
import json
from botocore.client import Config

def receive_data_from_pipeline(args_from_pipeline):
    path_config_yaml = StorageManager.download_file(remote_url=args_from_pipeline['config_yaml'], local_folder='./tmp', overwrite=True)
    path_dataset_json = StorageManager.download_file(remote_url=args_from_pipeline['datasets_url'], local_folder='./tmp', overwrite=True)
    path_reports  = StorageManager.download_file(remote_url=args_from_pipeline['reports_url'])
    
    print(args_from_pipeline['datasets_url'])
    path_config_yaml = Task.current_task().connect_configuration(path_config_yaml, 'Config_YAML')
    print('path_config_yaml', path_config_yaml)
    print('path_dataset_json', path_dataset_json)
    
    d_config_yaml = read_yaml(path_config_yaml)
    d_datasets_json = read_json(path_dataset_json)
    
    print('path_config_yaml', path_config_yaml, type(d_config_yaml))
    print('path_dataset_json', path_dataset_json, type(d_datasets_json))
    print('d_config_yaml', d_config_yaml.keys())
    print('d_datasets_json', d_datasets_json.keys())

    # download all data
    print('Download all data...')
    all_dataset = d_datasets_json['dataset']
    small_dataset = d_datasets_json['small_dataset']

    d_report = d_config_yaml(path_reports)
    return d_config_yaml, d_datasets_json, d_report

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

def get_classes_from_s3_folder(dataset_url):
    ls_files = StorageManager.list(remote_url=dataset_url,
        return_full_path=True,
        with_metadata=True
    )
    ls_classe = set()
    for file in ls_files:
        classes = file.get('name').split('/')[-2]
        ls_classe.add(classes)
    return list(ls_classe)

def read_yaml(yaml_path:str) -> dict:
    with open(yaml_path, 'r') as file:
        d = yaml.safe_load(file)
    return d

def read_json(json_path:str) -> dict:
    d = {}
    with open(json_path) as file:
        file_contents = file.read()        
        d = json.loads(file_contents)
    return d

if __name__ == '__main__':
    path_yaml = '../config/datasets.yaml'
    print(read_yaml(path_yaml))