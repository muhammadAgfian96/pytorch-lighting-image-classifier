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

task = Task.create(
    project_name="PL_Training", 
    task_name="test-v2",
    output_uri=conf.OUTPUT_URI,
    requirements_file='docker/requirements.txt',
    repo='git@github.com:muhammadAgfian96/pytorch-lighting-image-classifier.git',
    working_directory='/workspace'
)
params = asdict(conf_copy)
params['aug'].pop('augmentor_task')
params.pop('PROJECT_NAME')
params.pop('TASK_NAME')
params.pop('OUTPUT_URI')
params = task.set_parameters_as_dict(params)
pprint(params)
conf = override_config(params, conf)
pl.seed_everything(conf.data.random_seed)
data_module = ImageDataModule(conf)
model_classifier = ModelClassifier(conf)


# create callbacks


trainer = pl.Trainer(
    max_epochs=10,
    accelerator='gpu', 
    devices=1,
    logger=True,
    callbacks=[]
)
trainer.fit(model=model_classifier, datamodule=data_module)
trainer.test(datamodule=data_module)


