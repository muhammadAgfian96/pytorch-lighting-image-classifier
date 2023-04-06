import sys
sys.path.append('..')
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List
from clearml import TaskTypes
import albumentations as al
from albumentations.pytorch.transforms import ToTensorV2
from config.list_optimizer import ListOptimizer
import torch

@dataclass
class Config(object):
    PROJECT_NAME:str = 'Template'
    TASK_NAME:str = 'template-training'
    TYPE_TASK:Enum = TaskTypes.training
    OUTPUT_URI:str = 's3://10.8.0.66:9000/clearml-test'

@dataclass
class Storage:
    bucket_experiment:str = f"{Config.OUTPUT_URI}/training/experiment"
    bucket_dataset:str = f"{Config.OUTPUT_URI}/dataset/simple"

@dataclass
class Data:
    random_seed:int = 76
    dir:str = '/workspace/current_dataset'
    dataset:str = 'datasets.yaml'
    category:List[str] = None
    batch:int = -1
    train_ratio:float = 0.80
    val_ratio:float = 0.1
    test_ratio:float = 0.1
    input_size:int = 224
    input_resize:int = input_size + 32
    # mean = (0.5, 0.5, 0.5)
    # std = (0.5, 0.5, 0.5)
    mean = (0.485, 0.456, 0.406) 
    std= (0.229, 0.224, 0.225)

@dataclass
class Augmentations:
    augmentor:str = 'albumentations'
    type_executions:str = 'online' # offline
    augmentor_task:dict = field(init=False)
    
    mean = None
    std = None

    def setmeanstd(self, mean=Data.mean, std=Data.std):
        self.mean = mean
        self.std = std

    def get_ls_train(self):
        _p_default = 0.5
        _p_medium = 0.75
        _p_highest = 0.95

        if self.mean is None and self.std is None:
            self.setmeanstd()
        return [
            # common
            al.OneOf(p=1.0, transforms=[
                al.VerticalFlip(p=_p_highest),
                al.HorizontalFlip(p=_p_highest),
            ]),
            al.ShiftScaleRotate(
                always_apply=True,
                shift_limit= [-0.12, 0.12],
                scale_limit= [-0.05, 0.15],
                rotate_limit= [-45, 45],
                interpolation= 0,
                border_mode= 0,
                value= [0, 0, 0]
            ),
            al.CoarseDropout(
                always_apply=False, p=_p_default,
                min_holes=16, max_holes=25, 
                min_height=0.02, max_height=0.12, 
                min_width=0.02, max_width=0.12,
            ),
            al.RandomBrightnessContrast(
                p=_p_default, 
                brightness_limit=[-0.2, 0.45],
                contrast_limit=[-0.25, 0.35],
                brightness_by_max=False
            ), 
            al.OneOf(p=_p_default, transforms=[
                al.MotionBlur(),
                al.ImageCompression(),
                al.OpticalDistortion(),
                al.MultiplicativeNoise()]
            ),
            al.ZoomBlur(
                always_apply=False, 
                p=_p_medium, max_factor=(1.0, 1.31), 
                step_factor=(0.01, 0.03)
            ),
            al.Resize(height=Data.input_size, width=Data.input_size, always_apply=True),
            al.Normalize(mean=self.mean, std=self.std, always_apply=True, max_pixel_value=255.0),
            ToTensorV2(always_apply=True),
        ]

    def get_ls_val(self):
        if self.mean is None and self.std is None:
            self.setmeanstd()
        return [
            al.Resize(height=Data.input_size, width=Data.input_size, always_apply=True),
            al.Normalize(mean=self.mean, std=self.std, always_apply=True, max_pixel_value=255.0),
            ToTensorV2(transpose_mask=True),
        ]
    
    def __post_init__(self):
        def __to_dict(section='train'):
            if section == 'train':
                ls = self.get_ls_train()
            else:
                ls = self.get_ls_val()
            d_augment = {}
            count_oneof = 0
            for cp in ls:
                d_data = cp._to_dict()
                name_class = d_data['__class_fullname__']
                d_data.pop('__class_fullname__')
                if name_class == 'OneOf':
                    count_oneof+=1
                    name_class += f'_{count_oneof}'
                    d_augment[name_class] = []
                    ls_transforms = d_data['transforms']
                    for transform in ls_transforms:
                        name_children = transform['__class_fullname__']
                        transform.pop('__class_fullname__')
                        d_augment[name_class].append({name_children: transform})
                else:
                    d_augment[name_class] = d_data
            return d_augment
        
        def get_dict_augmentor():
            return {
                'train': __to_dict('train'), 
                'val': __to_dict('val')
            }

        self.augmentor_task = get_dict_augmentor()

@dataclass
class Model:
    # architecture:str = 'edgenext_x_small'
    architecture:str = '4d609a6b07ed447bae47af9a6fbfb999'
    pretrained:bool = True
    dropout:float = 0.0
    resume:bool = False
    checkpoint_model:str = None
    if resume:
        checkpoint_model:str = ''

@dataclass
class HyperParameters(object):
    epoch:int = 5
    
    loss_fn = torch.nn.CrossEntropyLoss()
    # optimizer
    opt_name:str = ListOptimizer.AdamW
    opt_weight_decay:float = 0
    opt_momentum:float = 0.9

    # scheduler lr
    base_learning_rate:float = 1.0e-3
    lr_scheduler:str = 'reduce_on_plateau' # step/multistep/reduce_on_plateau
    lr_step_size:int = 7
    lr_decay_rate:float = 0.5
    # lr_step_milestones:List[int] = [10, 15]
    precision: int = 16


# scheduler

@dataclass
class TrainingConfig():
    default:object = Config()
    db:object = Storage()
    data:object = Data()
    aug:object = Augmentations()
    net:object =  Model()
    hyp:object = HyperParameters()


if __name__ == '__main__':

    from rich import print

    my_conf = TrainingConfig()
    print(asdict(my_conf))