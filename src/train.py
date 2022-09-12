import os
os.environ['PYTHONPATH'] = os.getcwd()
from dataclasses import asdict
from pprint import pprint
import pytorch_lightning as pl
from src.preare_model import ModelClassifier
from src.prepare_data import ImageDataModule
from config.default import TrainingConfig
from config.list_models import list_models
from config.list_optimizer import ListOptimizer
from src.helper.utils import override_config

from clearml import Task

conf = TrainingConfig()
conf_copy = TrainingConfig()

cwd = os.getcwd()

task = Task.init(
    project_name='PL_Training',
    task_name=conf.TASK_NAME,  
    task_type=conf.TYPE_TASK
)
Task.current_task().add_requirements(os.path.join(cwd,'docker/requirements.txt'))
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
pprint(os.listdir(os.getcwd()))
pprint(new_params)
conf = override_config(new_params, conf)
pprint(asdict(conf))

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