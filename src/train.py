import os
import sys
from unicodedata import category
os.environ['PYTHONPATH'] = os.getcwd()
sys.path.append(os.getcwd())
from dataclasses import asdict
from rich import print
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.preare_model import ModelClassifier
from src.prepare_data import ImageDataModule
from config.default import TrainingConfig
from config.list_models import list_models
from config.list_optimizer import ListOptimizer
from src.helper.utils import override_config
from clearml import Task, OutputModel
from clearml import Dataset as DatasetClearML

conf = TrainingConfig()
conf_copy = TrainingConfig()

#region SETUP CLEARML
cwd = os.getcwd()

Task.add_requirements(f"-r {os.path.join(cwd,'docker/requirements.txt')}")
Task.force_requirements_env_freeze(False, os.path.join(cwd,'docker/requirements.txt'))

task = Task.init(
    project_name='Classifier/Test',
    task_name=conf.TASK_NAME,  
    task_type=conf.TYPE_TASK,
    auto_connect_frameworks=False
)

Task.current_task().set_script(
    repository='https://github.com/muhammadAgfian96/pytorch-lighting-image-classifier.git',
    branch='main',
    commit='',
    working_dir='.',
    entry_point='src/train.py'
)


task.upload_artifact('List Models', list_models)
task.upload_artifact('List Optimizer', ListOptimizer, preview=['Adam', 'SGD'])

params = asdict(conf_copy)

params['aug'].pop('augmentor_task')
params.pop('PROJECT_NAME')
params.pop('TASK_NAME')
params.pop('OUTPUT_URI')

params_general = task.connect({'TASK_NAME': params.get('TASK_NAME', conf.TASK_NAME)},  'General')
params_aug = task.connect(params['aug'], 'Augmentations')
params_db = task.connect(params['db'], 'Database')
params_data = task.connect(params['data'], 'Data')
params_hyp = task.connect(params['hyp'], 'Trainings')
params_net = task.connect(params['net'], 'Models')

new_params = {
    'aug': params_aug,
    'data': params_data,
    'db': params_db,
    'hyp': params_hyp,
    'net': params_net
}

print('CURRENT WORKDIR:', os.getcwd(), ' && ls .')
print(os.listdir(os.getcwd()))
print(new_params)

print('download data:', conf.data.dataset_id)
DatasetClearML.get(dataset_id=conf.data.dataset_id).get_mutable_local_copy(target_folder='/workspace/current_dataset')


conf = override_config(new_params, conf)
print(asdict(conf))
#endregion 

task.rename(conf.TASK_NAME)

pl.seed_everything(conf.data.random_seed)
data_module = ImageDataModule(conf)
model_classifier = ModelClassifier(conf)

checkpoint_callback = ModelCheckpoint(
    monitor="val_acc",
    mode='max',
    save_last= True,
    save_top_k=2,
    save_weights_only=True,
    verbose=True,
    filename='{epoch}-{val_acc:.2f}-{val_loss:.2f}'
)
checkpoint_callback.CHECKPOINT_NAME_LAST = '{epoch}-{val_acc:.2f}-{val_loss:.2f}-last'

early_stop_callback = EarlyStopping(
    monitor="val_acc", min_delta=0.01, patience=4, verbose=False, mode="max", check_on_train_epoch_end=True)

# create callbacks
ls_callback = [
    checkpoint_callback,
    early_stop_callback
]

trainer = pl.Trainer(
    max_epochs=conf.hyp.epoch,
    accelerator='gpu', 
    devices=1,
    logger=True,
    callbacks=ls_callback,
    precision=conf.hyp.precision,
)

trainer.fit(model=model_classifier, datamodule=data_module)
trainer.test(datamodule=data_module)

# saving upload / models
print("checkpoint_callback.dirpath: ", checkpoint_callback.dirpath)
output_model_best = OutputModel(task=task, name=f'best-{conf.TASK_NAME}', framework="Pytorch Lightning", comment=f"best model. dataset_id: {conf.data.dataset_id}")
output_model_best.update_labels({lbl:idx for idx, lbl in enumerate(conf.data.category)})
output_model_best.update_weights(
    weights_filename=checkpoint_callback.best_model_path,
    target_filename=f'best-{conf.TASK_NAME}-{task.id}.pt')
output_model_best.update_design(config_dict={'net': conf.net.architecture, 'input_size': conf.data.input_size})

output_model_last = OutputModel(task=task, name=f'latest-{conf.TASK_NAME}', framework="Pytorch Lightning", comment=f"latest model. dataset_id: {conf.data.dataset_id}")
output_model_last.update_labels({lbl:idx for idx, lbl in enumerate(conf.data.category)})
output_model_last.update_weights(
    weights_filename=checkpoint_callback.last_model_path,
    target_filename=f'last-{conf.TASK_NAME}-{task.id}.pt')
output_model_last.update_design(config_dict={'net': conf.net.architecture, 'input_size': conf.data.input_size})
