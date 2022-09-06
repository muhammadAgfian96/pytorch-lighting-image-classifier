from dataclasses import asdict
import os
from pprint import pprint
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from src.preare_model import ModelClassifier
from src.prepare_data import ImageDataModule
from config.default import TrainingConfig
conf = TrainingConfig()
conf_2 = TrainingConfig()

from clearml import Task
task = Task.init(
    project_name="PL_Training", 
    task_name="test-v1",
    output_uri=conf.OUTPUT_URI
)

pl.seed_everything(0)
data_module = ImageDataModule(conf)
model_classifier = ModelClassifier(conf)

conf_2 = asdict(conf_2)
conf_2['aug'].pop('augmentor_task')
task.set_parameters_as_dict(conf_2)
trainer = pl.Trainer(
    max_epochs=10,
    accelerator='gpu', 
    devices=1,
    logger=True
)
trainer.fit(model=model_classifier, datamodule=data_module)
trainer.test(datamodule=data_module)


