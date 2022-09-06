import timm 
import torch
from clearml import Task
from config.default import TrainingConfig


conf = TrainingConfig()
task = Task.init(
    project_name=conf.PROJECT_NAME, 
    task_name=conf.TASK_NAME, 
    output_uri=conf.OUTPUT_URI
)


# get data

# preprocessing data

# define hyperparameters model

# define architecture model
model = timm.create_model(
    model_name=conf.net.architecture, 
    pretrained=conf.net.pretrained,
    checkpoint_path=conf.net.checkpoint_model,
    num_classes=conf.net.num_class
)
x     = torch.randn(1, 3, conf.data.input_size, conf.data.input_size)
print(model(x).shape)

# define callbacks

# training

# testing