from clearml import Task

from config.default import TrainingConfig
from src.data import ImageDataModule

task = Task.init(
    project_name="Debug/Data-Module",
    task_name="Data Module",
    task_type=Task.TaskTypes.custom,
    auto_connect_frameworks=False,
    tags=["template", "debug"],
)

path_data_yaml = "/workspace/config/datasetsv2.yaml"

conf = TrainingConfig()
data_module = ImageDataModule(conf=conf, path_yaml_data=path_data_yaml)

# prepare data
data_module.prepare_data()

# metadata stored in clearml
conf.net.num_class = len(data_module.classes_name)
conf.data.category = data_module.classes_name
task.set_model_label_enumeration(
    {lbl: idx for idx, lbl in enumerate(conf.data.category)}
)

data_module.visualize_augmented_images_v2("train-augment", num_images=15, augment_policy="imagenet")
data_module.visualize_augmented_images_v2("train-augment", num_images=15, augment_policy="cifar10")
data_module.visualize_augmented_images_v2("train-augment", num_images=15, augment_policy="svhn")
data_module.visualize_augmented_images_v2("train-augment", num_images=15, augment_policy="ra")
data_module.visualize_augmented_images_v2("train-augment", num_images=15, augment_policy="augmix")
data_module.visualize_augmented_images_v2("train-augment", num_images=15, augment_policy="ta_wide")
data_module.visualize_augmented_images("train-augment", num_images=50)
data_module.visualize_augmented_images("val-augment", num_images=15)