import os
import random
import shutil
import time

import albumentations as al
import numpy as np
import plotly.graph_objects as go
import pytorch_lightning as pl
import torch
from clearml import Dataset as DatasetClearML
from clearml import StorageManager, Task
from PIL import Image
from rich import print
from torch.utils.data import DataLoader, Dataset

from config.default import TrainingConfig
from src.helper.data_helper import MinioDatasetDownloader
from src.utils import get_list_data, map_data_to_dict, read_yaml


class ImageDatasetBinsho(Dataset):
    def __init__(self, data, transform, classes):
        self.data = data
        self.transform = al.Compose(transform)
        self.classes = classes

    def __len__(self): return len(self.data)
    
    def __getitem__(self, index):
        fp_img, y  = self.data[index]
        y_label = torch.tensor(int(y))
        x_image = np.array(Image.open(fp_img)) # RGB format!
        x_image = self.transform(image=x_image)["image"] 
        return x_image, y_label

class ImageDatasetBinsho51(Dataset):
    def __init__(self, data, transform, classes):
        self.data = data
        self.transform = al.Compose(transform)
        self.classes = classes

    def __len__(self): return len(self.data)
    
    def __getitem__(self, index):
        fp_img, y  = self.data[index]
        y_label = torch.tensor(int(y))
        x_image = np.array(Image.open(fp_img)) # RGB format!
        x_image = self.transform(image=x_image)["image"] 
        return x_image, y_label

class ImageDataModule(pl.LightningDataModule):
    def __init__(self, conf:TrainingConfig, path_yaml_data=None):
        super().__init__()
        self.data_dir = conf.data.dir
        self.conf = conf
        self.prepare_data_has_downloaded = False
        self.batch_size = self.conf.data.batch
        self.path_yaml_dataset = path_yaml_data
        self.test_local_path = '/workspace/current_dataset_test'

        if os.path.exists('/workspace/current_dataset'):
            shutil.rmtree('/workspace/current_dataset')
        if os.path.exists(self.test_local_path):
            shutil.rmtree(self.test_local_path)
    
    def prepare_data(self) -> None:
        # set clearml and download the data
        if not self.prepare_data_has_downloaded:
            print('cleaning old data...')
            if os.path.exists(self.conf.data.dir):
                shutil.rmtree(self.conf.data.dir)
            if os.path.exists('/workspace/current_dataset_test'):
                shutil.rmtree('/workspace/current_dataset_test')
            print('Downloading data...')
            try:
                os.makedirs(self.conf.data.dir, exist_ok=True)
                print('creted folder, downloading dataset...', )
                if 's3://10.8.0.66:9000' in  self.conf.data.dataset:
                    print('single link s3', self.conf.data.dataset)
                    downloaded_ds_s3 = StorageManager.download_folder(
                        remote_url=self.conf.data.dataset,
                        local_folder=self.conf.data.dir,
                        overwrite=False
                    )
                    path_s3_minio = self.conf.data.dataset.split('s3://10.8.0.66:9000/')[-1]
                    fp_local_download = os.path.join(self.conf.data.dir, path_s3_minio)
                    for folder_name in os.listdir(fp_local_download):
                        folder_path_source = os.path.join(fp_local_download, folder_name)
                        if os.path.exists(folder_path_source):
                            shutil.move(folder_path_source, os.path.join(self.conf.data.dir, folder_name))
                    shutil.rmtree(os.path.join(self.conf.data.dir, path_s3_minio.split('/')[0]))

                    print('path_downloaded:', self.conf.data.dir, downloaded_ds_s3)
                    self.data_dir = self.conf.data.dir
                
                elif type(self.conf.data.dataset) == type({}):
                    print('download dict')
                    s3_api = MinioDatasetDownloader(dataset=self.conf.data.dataset, download_dir='/workspace/current_dataset')
                    start_time = time.time()
                    s3_api.download_dataset()
                    end_time = time.time()
                    self.data_dir = self.conf.data.dir = '/workspace/current_dataset'
                    duration = end_time - start_time
                    print('how long it take :', round(duration, 3), 'seconds')
                
                elif self.conf.data.dataset == 'datasets.yaml':
                    d_train, d_test = self.__extract_list_link_dataset_yaml()
                    class_name_train = d_train.keys() 
                    class_name_test = d_test.keys() 
                    print('class_name_train', class_name_train)
                    print('class_name_test', class_name_test)
                    for cls_name in class_name_test:
                        if cls_name not in class_name_train:
                            print(f"[ERROR] {cls_name} not in class_name_train:")
                            print(class_name_train)
                            print("Please make sure, the class of dataset-test contain in dataset-train!")
                            Task.current_task().mark_failed()
                            exit()

                    s3_api = MinioDatasetDownloader(dataset=d_train, download_dir='/workspace/current_dataset')
                    print('[Downloading] dataset-train')
                    start_time = time.time()
                    s3_api.download_dataset()
                    end_time = time.time()
                    self.data_dir = self.conf.data.dir = '/workspace/current_dataset'
                    duration = end_time - start_time
                    print('[Train] how long it take dataset.yaml :', round(duration, 3), 'seconds')

                    if len(d_test) != 0:
                        s3_api_test = MinioDatasetDownloader(dataset=d_test, download_dir=self.test_local_path)
                        print('[Downloading] dataset-test')
                        start_time = time.time()
                        s3_api_test.download_dataset()
                        end_time = time.time()
                        duration = end_time - start_time
                        print('[Test] how long it take dataset.yaml :', round(duration, 3), 'seconds')

                        self.ls_test_map_dedicated = map_data_to_dict(d_data=d_test, local_path_dir=self.test_local_path)

                else:
                    print('dataset_id is dowloading',)
                    DatasetClearML.get(dataset_id=self.conf.data.dataset).get_mutable_local_copy(target_folder='/workspace/current_dataset', overwrite=True)
                
                print('success download data:', self.conf.data.dir)
                self.prepare_data_has_downloaded = True
                self.ddata_by_label, self.ls_train_set, self.ls_val_set, self.ls_test_set, self.d_metadata, self.classes_name = get_list_data(config=self.conf)
                print('metadata:', self.d_metadata)
                print('classname:', self.classes_name)
                self.__log_distribution_data_clearml(self.d_metadata)
            
            except Exception as e:
                print(e)
                print('out')
                exit()
        else:
            print('we has_downloaded your data')

    def setup(self, stage: str):
        # get list of data        
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.data_train = ImageDatasetBinsho(
                self.ls_train_set, 
                transform=self.conf.aug.get_ls_train(),
                classes=self.classes_name)
            self.data_val = ImageDatasetBinsho(
                self.ls_val_set, 
                transform=self.conf.aug.get_ls_val(),
                classes=self.classes_name)
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.data_test = ImageDatasetBinsho(
                self.ls_test_set, 
                transform=self.conf.aug.get_ls_val(),
                classes=self.classes_name)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=4)
    
    def __log_distribution_data_clearml(self, d_metadata):
        labels_pie = ['train', 'val', 'test']
        values_pie = [d_metadata['train_count'], d_metadata['val_count'], d_metadata['test_count']]
        fig_dist_train_val_test = go.Figure(data=[go.Pie(labels=labels_pie, values=values_pie)])

        Task.current_task().get_logger().report_plotly(
            title='Data Distribution Section', 
            series='Train/Val/Test', 
            figure=fig_dist_train_val_test, 
            iteration=1
        )

        # Sample data
        x_values = [value for value in d_metadata['counts']['train'].keys()]
        y_train = [value for value in d_metadata['counts']['train'].values()]
        y_val = [value for value in d_metadata['counts']['val'].values()]
        y_test = [value for value in d_metadata['counts']['test'].values()]

        # Create three bar chart traces, one for each section
        trace1 = go.Bar(x=x_values, y=y_train, name='Train', text=y_train, textposition='auto')
        trace2 = go.Bar(x=x_values, y=y_val, name='Validation', text=y_val, textposition='auto')
        trace3 = go.Bar(x=x_values, y=y_test, name='Test', text=y_test, textposition='auto')

        # Create a layout for the chart
        layout = go.Layout(title='Data Distribution', xaxis=dict(title='Class Name', showticklabels=True), yaxis=dict(title='Counts', showticklabels=True), barmode='group', legend=dict(x=0, y=1))

        # Create a figure object that contains the traces and layout
        fig_bar_class = go.Figure(data=[trace1, trace2, trace3], layout=layout)

        # Update the font and background colors of the chart
        # fig_bar_class.update_layout(font=dict(color='white'), plot_bgcolor='#2c3e50', paper_bgcolor='#2c3e50')


        Task.current_task().get_logger().report_plotly(
            title='Data Distribution Section', 
            series='Class View', 
            figure=fig_bar_class, 
            iteration=1
        )

    def __extract_list_link_dataset_yaml(self):
        print('path_yaml_config:', self.path_yaml_dataset)
        datasets_yaml = read_yaml(self.path_yaml_dataset)
        print(datasets_yaml)

        ls_url_files_test = []
        # get/download list-data
        ls_url_files_train = self.__get_list_url_from_minio_s3(datasets_yaml, section='train')
        ls_url_files_test = self.__get_list_url_from_minio_s3(datasets_yaml, section='test')

        # datasets_yaml.get('dataset-test', False)
        
        d_train = self.__mapping_to_dict_class(ls_url_files_train)
        d_test = self.__mapping_to_dict_class(ls_url_files_test)

        return d_train, d_test

    def __mapping_to_dict_class(self, ls_url_files_train):
        d_map = {}
        print('mapping to dict_class..')
        if len(ls_url_files_train) == 0:
            return None
        for d_file in ls_url_files_train:
            url_file = d_file['name']
            class_name  = url_file.split('/')[-2]
            if class_name not in d_map.keys(): d_map[class_name] = []
            d_map[class_name].append(url_file)
        return d_map

    def __get_list_url_from_minio_s3(self, datasets_yaml, section='train'):
        ls_urls_files = []
        print(f'Get list dataset-{section}...')
        for path_dataset in datasets_yaml[f'dataset-{section}']:
            if path_dataset is None:
                print('None path_dataset')
                continue
            if 's3://10.8.0.66:9000' not in path_dataset: 
                remote_url = os.path.join('s3://10.8.0.66:9000', path_dataset)
            else: 
                remote_url = path_dataset
            print('<remote_url>', remote_url)

            ls_files = StorageManager.list(
                remote_url=remote_url,
                return_full_path=True,
                with_metadata=True
            )
            print('\tTotal Data:', len(ls_files))
            if len(ls_files) == 0:
                print('CHECK THIS DATA', remote_url)
                continue
            ls_urls_files.extend(ls_files)
            ls_files = None
            print('-----')
        return ls_urls_files
    
    def visualize_augmented_images(self, section:str, num_images=5):
        print(f'vizualizing sample {section}...')
        ls_viz_data = []
        for label, ls_fp_image in self.ddata_by_label.items():
            ls_viz_data.extend(ls_fp_image[0:num_images])

        random.shuffle(ls_viz_data)
        if 'train' in section:
            dataset_viz = ImageDatasetBinsho(
                ls_viz_data, 
                transform=self.conf.aug.get_ls_train()[:-2],
                classes=self.classes_name)
        
        if 'val' in section or 'test' in section:
            dataset_viz = ImageDatasetBinsho(
                ls_viz_data, 
                transform=self.conf.aug.get_ls_val()[:-2],
                classes=self.classes_name)


        for i in range(len(ls_viz_data)):
            image_array, label = dataset_viz[i]
            label_name = self.classes_name[label]
            Task.current_task().get_logger().report_image(f"{section}", f"{label_name}_{i}", iteration=1, image=image_array)

