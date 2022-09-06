import os
os.environ['PYTHONPATH'] = os.getcwd()
from dataclasses import asdict
from pprint import pprint
import torch
import pytorch_lightning as pl
from src.preare_model import ModelClassifier
from src.prepare_data import ImageDataModule
from config.default import TrainingConfig
from pytorch_lightning.callbacks import ModelCheckpoint

from src.helper.utils import override_config

conf = TrainingConfig()
conf_copy = TrainingConfig()

from clearml import Task


# task = Task.create(
#     project_name="PL_Training", 
#     task_name="test-v2",    
#     requirements_file='docker/requirements.txt',
#     repo='https://github.com/muhammadAgfian96/pytorch-lighting-image-classifier.git',
#     script='./src/train.py',
#     add_task_init_call=True
# )
task = Task.init(
    project_name='PL_Training',
    task_name="test-v2",  
)
task.add_requirements('docker/requirements.txt')
pprint(os.getcwd())
# task.started()
params = asdict(conf_copy)
params['aug'].pop('augmentor_task')
params.pop('PROJECT_NAME')
params.pop('TASK_NAME')
params.pop('OUTPUT_URI')
params = task.connect(params)
task.set_parameters_as_dict(params)
pprint(params)
conf = override_config(params, conf)
pl.seed_everything(conf.data.random_seed)
data_module = ImageDataModule(conf)
model_classifier = ModelClassifier(conf)


# create callbacks
trainer = pl.Trainer(
    max_epochs=conf.hyp.epoch,
    accelerator='gpu', 
    devices=1,
    logger=True,
    callbacks=[]
)
trainer.fit(model=model_classifier, datamodule=data_module)
trainer.test(datamodule=data_module)


