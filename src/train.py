import os
import shutil
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

# copy to fix file
import clearml
path_clearml_package = os.path.join(clearml.__path__[0], 'storage', 'helper.py')
from_helper_fix = os.path.join(os.getcwd(), 'clearml-fix-hardcore', 'helper.py')
if os.path.exists(path_clearml_package):
    shutil.copy2(src=from_helper_fix, dst=path_clearml_package)

import clearml
from clearml import Task

conf = TrainingConfig()
conf_copy = TrainingConfig()

cwd = os.getcwd()

Task.add_requirements(os.path.join(cwd,'docker/requirements.txt'))
task = Task.init(
    project_name='PL_Training',
    task_name=conf.TASK_NAME,  
    task_type=conf.TYPE_TASK
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