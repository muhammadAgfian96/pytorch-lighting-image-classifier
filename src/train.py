import os
import sys

import torch
cwd = os.getcwd()
os.environ['PYTHONPATH'] = cwd
sys.path.append(cwd)

from dataclasses import asdict
from rich import print
import pytorch_lightning as pl
from src.net import Classifier
from src.data import ImageDataModule
from src.utils import read_json, read_yaml, override_config, receive_data_from_pipeline
from clearml import Task, OutputModel, StorageManager
from clearml import Dataset as DatasetClearML
from config.default import TrainingConfig
from finetuning_scheduler import FinetuningScheduler
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.tuner import tuning

# ----------------------------------------------------------------------------------
# ClearML Setup
# ----------------------------------------------------------------------------------
# Task.add_requirements(f"-r {os.path.join(cwd,'docker/requirements.txt')}")
Task.force_requirements_env_freeze(False, os.path.join(cwd,'docker/requirements.txt'))

task = Task.init(
    project_name='Image Classifier/Template',
    task_name='Training',  
    task_type=Task.TaskTypes.training,
    auto_connect_frameworks=False
)

Task.current_task().set_script(
    repository='https://github.com/muhammadAgfian96/pytorch-lighting-image-classifier.git',
    branch='main',
    commit='',
    working_dir='.',
    entry_point='src/train.py'
)
Task.current_task().set_base_docker(
    docker_image='pytorch/pytorch:latest',
    docker_arguments= ['--ipc=host', '--gpus=all', '-e PYTHONPATH=/workspace'],
    docker_setup_bash_script=['apt install --no-install-recommends -y zip htop screen libgl1-mesa-glx libsm6 libxext6 libxrender-dev']
)
print("""
# ----------------------------------------------------------------------------------
# Manage Configuration
# ----------------------------------------------------------------------------------
""")
# running if inside pipeline
args_from_pipeline = {
    'config_yaml': '',
    'project_name': '',
    'reports_url': '',
    'datasets_url': '',
    'using_pipeline': False
}
task.connect(args_from_pipeline, 'Config-Pipeline')

inside_pipeline = args_from_pipeline['using_pipeline']
if inside_pipeline:
    d_config_yaml, d_datasets_json, d_report = receive_data_from_pipeline(task, args_from_pipeline)


# running for single task
conf = TrainingConfig()
params = asdict(conf).copy()
params['aug'].pop('augmentor_task')
params_general = task.connect(params['default'],  'Project')
params_aug = task.connect(params['aug'], 'Augmentations')
params_db = task.connect(params['db'], 'Database')
params_data = task.connect(params['data'], 'Data')
params_hyp = task.connect(params['hyp'], 'Trainings')
params_net = task.connect(params['net'], 'Models')


new_params = {
    'default': params['default'],
    'aug': params['aug'],
    'db': params['db'],
    'data': params['data'],
    'hyp': params['hyp'],
    'net': params['net']
}

print('CURRENT WORKDIR:', os.getcwd(), ' && ls .')
print(os.listdir(os.getcwd()))
print(new_params)
print(params)

print('torch.cuda.is_available():', torch.cuda.is_available())
conf = override_config(new_params, conf)
print(asdict(conf))

path_yaml_config = '/workspace/config/datasets.yaml'
path_yaml_config = Task.current_task().connect_configuration(path_yaml_config, 'datasets.yaml')
Task.current_task().execute_remotely()


task.rename(new_params['default']['TASK_NAME'])
print("""
# ----------------------------------------------------------------------------------
# Prepare Data, Model, Callbacks For Training 
# ----------------------------------------------------------------------------------
"""
)
pl.seed_everything(conf.data.random_seed)

print("# Data ---------------------------------------------------------------------")
data_module = ImageDataModule(conf=conf, path_yaml_data=path_yaml_config)
data_module.prepare_data()
conf.net.num_class = len(data_module.classes_name)
conf.data.category = data_module.classes_name
task.set_model_label_enumeration({lbl:idx for idx, lbl in enumerate(conf.data.category)})

print("# Callbacks -----------------------------------------------------------------")
checkpoint_callback = ModelCheckpoint(
    monitor="val_acc",
    mode='max',
    save_last= True,
    save_top_k=1,
    save_weights_only=True,
    verbose=True,
    filename='{epoch}-{val_acc:.2f}-{val_loss:.2f}'
)
checkpoint_callback.CHECKPOINT_NAME_LAST = '{epoch}-{val_acc:.2f}-{val_loss:.2f}-last'

early_stop_callback = EarlyStopping(
    monitor="val_acc", 
    min_delta=0.01, 
    patience=4, 
    verbose=False, 
    mode="max", 
    check_on_train_epoch_end=True
)
lr_monitor = LearningRateMonitor(logging_interval='epoch')

# create callbacks
ls_callback = [
    checkpoint_callback,
    lr_monitor,
    # early_stop_callback
    # FinetuningScheduler()
]

accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
model_classifier = Classifier(conf)


print("""
# ----------------------------------------------------------------------------------
# Training and Testing 
# ----------------------------------------------------------------------------------
""")
      
auto_batch = False
auto_lr_find = False
if conf.data.batch == -1:
    auto_batch = True
if conf.hyp.base_learning_rate == -1:
    auto_lr_find = True

trainer = pl.Trainer(
    max_steps=-1,
    max_epochs=conf.hyp.epoch,
    accelerator=accelerator, 
    devices=1,
    logger=True,
    callbacks=ls_callback,
    precision=conf.hyp.precision,
    auto_scale_batch_size=auto_batch,
    auto_lr_find=auto_lr_find,
    log_every_n_steps=4
)
data_module.setup(stage='fit')
trainer.tune(model=model_classifier, datamodule=data_module)
trainer.fit(model=model_classifier, datamodule=data_module)

data_module.setup(stage='test')
trainer.test(datamodule=data_module)

print("""
# ----------------------------------------------------------------------------------
# Reporting 
# ----------------------------------------------------------------------------------
"""
)
print("checkpoint_callback.dirpath: ", checkpoint_callback.dirpath)
output_model_best = OutputModel(
    task=task, 
    name=f'best', 
    framework="Pytorch Lightning", 
)
# output_model_best.update_labels({lbl:idx for idx, lbl in enumerate(conf.data.category)})
print(checkpoint_callback.best_model_path)
print(checkpoint_callback.last_model_path)
output_model_best.update_weights(
    weights_filename=checkpoint_callback.best_model_path,
    target_filename=f'best.ckpt')
output_model_best.update_design(config_dict={'net': conf.net.architecture, 'input_size': conf.data.input_size})

output_model_last = OutputModel(task=task, name=f'latest', framework="Pytorch Lightning")
# output_model_last.update_labels({lbl:idx for idx, lbl in enumerate(conf.data.category)})
output_model_last.update_weights(
    weights_filename=checkpoint_callback.last_model_path,
    target_filename=f'latest.ckpt')
output_model_last.update_design(config_dict={'net': conf.net.architecture, 'input_size': conf.data.input_size})
