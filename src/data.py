import random
import albumentations as al
import numpy as np
import plotly.graph_objects as go
import pytorch_lightning as pl
import torch
from clearml import Task
from PIL import Image
from rich import print
from torch.utils.data import DataLoader, Dataset

from config.default import TrainingConfig

from src.data_controller.downloader.manager import DownloaderManager
from src.data_controller.manipulator.splitter_dataset import splitter_dataset
from src.augment import ClassificationPresetTrain, ClassificationPresetEval
import traceback



class ImageDatasetBinsho(Dataset):
    def __init__(self, data, transform, classes):
        self.data = data
        self.transform = al.Compose(transform)
        self.classes = classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fp_img, y = self.data[index]
        y_label = torch.tensor(int(y))
        x_image = np.array(Image.open(fp_img))  # RGB format!
        x_image = self.transform(image=x_image)["image"]
        return x_image, y_label

class ImageDatasetBinshoV2(Dataset):
    def __init__(self, data, classes, augment_policy="autoaugment", viz_mode=False):
        """
        augment_policy = "autoaugment" or "ra" or "augmix" or "ta_wide"
        """
        self.data = data
        self.classes = classes
        self.transform = ClassificationPresetTrain(
            auto_augment_policy=augment_policy,
            viz_mode=viz_mode,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fp_img, y = self.data[index]
        y_label = torch.tensor(int(y))
        x_image = Image.open(fp_img)  # RGB format!
        x_image = self.transform(img=x_image)
        return x_image, y_label

from src.schema.config import DataConfig, TrainConfig

class ImageDataModule(pl.LightningDataModule):
    def __init__(self, conf: TrainingConfig, d_train:TrainConfig, d_dataset:DataConfig, path_yaml_data=None):
        super().__init__()
        self.prepare_data_has_downloaded = False
        self.d_dataset:DataConfig = d_dataset
        self.d_train:TrainConfig = d_train

        self.data_dir = d_dataset.dir_output
        self.conf = conf
        
        self.batch_size = d_train.batch
        self.path_yaml_dataset = d_dataset.yaml_path
        self.test_local_path = "/workspace/current_dataset_test"
        self.ls_test_map_dedicated = None

    # -------------------------- Main Function --------------------------
    def prepare_data(self) -> None:
        # set clearml and download the data

        if not self.prepare_data_has_downloaded:
            # self.__cleaning_old_data()

            try:
                """
                    purpose:
                    1. download data as dir tree
                    2. self.data_dir = self.conf.data.dir
                    self.ls_train_set = [(fp_img, y), ..., ...]
                    self.ls_val_set  = [(fp_img, y), ..., ...]
                    self.ls_test_set  = [(fp_img, y), ..., ...]
                    there 4 type of dataset input will fetch:
                    1. single link s3
                    2. dict data (coming from pipeline) [NOT YET DONE]
                    3. yaml file
                    4. id dataset clearml
                """
                
                # Download
                output_dir_train, output_dir_test = DownloaderManager().fetch(
                    input_dataset=self.d_dataset.yaml_path,
                    output_dir=self.d_dataset.dir_output
                )
                
                result =  splitter_dataset(
                    d_dataset=self.d_dataset,
                    path_dir_train=output_dir_train,
                    path_dir_test=output_dir_test
                )

                (
                    self.data_train_mapped, 
                    self.ls_train_dataset, 
                    self.ls_val_dataset, 
                    self.ls_test_dataset, 
                    self.d_metadata
                ) = result
                
                self.classes_name = self.d_metadata.get('class_names')
                self.__log_distribution_data_clearml(self.d_metadata)
                self.prepare_data_has_downloaded = True

            except Exception as e:
                print(traceback.format_exc())
                print("🚨 Error:", e)
                print("⛔ Exit Programs")
                exit()
        else:
            print("we has_downloaded your data")

    def setup(self, stage: str):
        # get list of data
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.data_train = ImageDatasetBinsho(
                self.ls_train_dataset,
                transform=self.conf.aug.get_ls_train(),
                classes=self.d_dataset.classes,
            )
            self.data_val = ImageDatasetBinsho(
                self.ls_val_dataset,
                transform=self.conf.aug.get_ls_val(),
                classes=self.d_dataset.classes,
            )
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.data_test = ImageDatasetBinsho(
                self.ls_test_dataset,
                transform=self.conf.aug.get_ls_val(),
                classes=self.d_dataset.classes,
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=6,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val, 
            batch_size=int(self.batch_size), 
            num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test, 
            batch_size=int(self.batch_size), 
            num_workers=4
        )

    # --------------------------- PRIVATE ---------------------------

    def __verify_class_train_and_test(self, d_train, d_test):
        class_name_train = d_train.keys()
        class_name_test = d_test.keys()
        print("class_name_train", class_name_train)
        print("class_name_test", class_name_test)
        for cls_name in class_name_test:
            if cls_name not in class_name_train:
                print(f"[ERROR] {cls_name} not in class_name_train:")
                print(class_name_train)
                print(
                    "Please make sure, the class of dataset-test contain"
                    " in dataset-train!"
                )
                Task.current_task().mark_failed()
                exit()

    # --------------------- LOG DATA ---------------------
    def __log_distribution_data_clearml(self, d_metadata):
        """
        Log distribution data to clearml.
        """
        labels_pie = ["train", "val", "test"]
        values_pie = [
            d_metadata["count_section"]["train"],
            d_metadata["count_section"]["val"],
            d_metadata["count_section"]["test"],
        ]
        fig_dist_train_val_test = go.Figure(
            data=[go.Pie(labels=labels_pie, values=values_pie)]
        )

        Task.current_task().get_logger().report_plotly(
            title="Data Distribution Section",
            series="Train/Val/Test",
            figure=fig_dist_train_val_test,
            iteration=1,
        )

        # Sample data
        x_values = [value for value in d_metadata["count"]["train"].keys()]
        y_train = [value for value in d_metadata["count"]["train"].values()]
        y_val = [value for value in d_metadata["count"]["val"].values()]
        y_test = [value for value in d_metadata["count"]["test"].values()]

        # Create three bar chart traces, one for each section
        trace1 = go.Bar(
            x=x_values,
            y=y_train,
            name="Train",
            text=y_train,
            textposition="auto",
        )
        trace2 = go.Bar(
            x=x_values,
            y=y_val,
            name="Validation",
            text=y_val,
            textposition="auto",
        )
        trace3 = go.Bar(
            x=x_values, y=y_test, name="Test", text=y_test, textposition="auto"
        )

        # Create a layout for the chart
        layout = go.Layout(
            title="Data Distribution",
            xaxis=dict(title="Class Name", showticklabels=True),
            yaxis=dict(title="Counts", showticklabels=True),
            barmode="group",
            legend=dict(x=0, y=1),
        )

        # Create a figure object that contains the traces and layout
        fig_bar_class = go.Figure(data=[trace1, trace2, trace3], layout=layout)

        # Update the font and background colors of the chart
        # fig_bar_class.update_layout(
        #   font=dict(color='white'),
        #   plot_bgcolor='#2c3e50',
        #   paper_bgcolor='#2c3e50')

        Task.current_task().get_logger().report_plotly(
            title="Data Distribution Section",
            series="Class View",
            figure=fig_bar_class,
            iteration=1,
        )

    # --------------------- Vizualisation Augmentation ---------------------
    def visualize_augmented_images(self, section: str, num_images=5):
        print(f"vizualizing sample {section}...")
        ls_viz_data = []
        for label, ls_fp_image in self.data_train_mapped.items():
            ls_viz_data.extend(ls_fp_image[0:num_images])

        random.shuffle(ls_viz_data)
        if "train" in section:
            dataset_viz = ImageDatasetBinsho(
                ls_viz_data,
                transform=self.conf.aug.get_ls_train()[:-2],
                classes=self.classes_name,
            )

        if "val" in section or "test" in section:
            dataset_viz = ImageDatasetBinsho(
                ls_viz_data,
                transform=self.conf.aug.get_ls_val()[:-2],
                classes=self.classes_name,
            )

        for i in range(len(ls_viz_data)):
            image_array, label = dataset_viz[i]
            label_name = self.classes_name[label]
            Task.current_task().get_logger().report_image(
                f"{section}",
                f"{label_name}_{i}",
                iteration=1,
                image=image_array,
            )

    def visualize_augmented_images_v2(self, section: str, num_images=5, augment_policy="autoaugment", prefix_sec=""):
        print(f"vizualizing v2 sample {section}...")
        ls_viz_data = []
        for label, ls_fp_image in self.data_train_mapped.items():
            ls_viz_data.extend(ls_fp_image[0:num_images])

        random.shuffle(ls_viz_data)
        if "train" in section:
            dataset_viz = ImageDatasetBinshoV2(
                ls_viz_data,
                classes=self.classes_name,
                augment_policy=augment_policy,
                viz_mode=True,
            )

        if "val" in section or "test" in section:
            dataset_viz = ImageDatasetBinsho(
                ls_viz_data,
                transform=self.conf.aug.get_ls_val()[:-2],
                classes=self.classes_name,
            )

        for i in range(len(ls_viz_data)):
            image_array, label = dataset_viz[i]
            label_name = self.classes_name[label]
            Task.current_task().get_logger().report_image(
                f"{augment_policy}-{section}",
                f"{label_name}_{i}",
                iteration=1,
                image=image_array,
            )

