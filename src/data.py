import random
import shutil
import time
import torch
import pytorch_lightning as pl
import albumentations as al
import os
from os.path import join
from PIL import Image
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import plotly.graph_objects as go

from clearml import (
    Dataset as DatasetClearML, 
    StorageManager, 
    Task)
from config.default import TrainingConfig
from rich import print
from src.helper.data_helper import MinioDatasetDownloader
from src.utils import read_yaml
import matplotlib.pyplot as plt

def get_list_data(conf:TrainingConfig):
    tr = conf.data.train_ratio
    va = conf.data.val_ratio
    te = conf.data.test_ratio

    def check_health_img(fp):
        try:
            img = cv2.imread(fp)
            h,w,c = img.shape
            if h > 0 and w>0:
                return True
        except Exception as e:
            print('[ERROR] Image Corrupt: ', fp, e)
            return False

    def split_list(ls_fp_image):


        count_imgs = len(ls_fp_image)
        tr_count = int(tr*count_imgs)
        va_count = int(va*count_imgs)
        random.shuffle(ls_fp_image)
        train = ls_fp_image[:tr_count]
        val = ls_fp_image[tr_count:tr_count+va_count]
        test = ls_fp_image[tr_count+va_count:]
        return train, val, test
    
    d_metadata = {
        'ratio': [],
        'counts' : {
            'train': {},
            'val': {},
            'test': {},
        }
    }


    classes_name = sorted(os.listdir(conf.data.dir))

    d_data = {lbl:[] for lbl in classes_name}
    ls_train, ls_val, ls_test  = [], [], []

    for label in classes_name:
        fp_folder = join(conf.data.dir, label)
        for file in os.listdir(fp_folder):
            fp_image = join(fp_folder, file)
            if check_health_img(fp_image):
                d_data[label].append((fp_image, classes_name.index(label)))
    
    d_metadata['ratio'] = [tr, va, te]

    ls_train_set, ls_val_set, ls_test_set = [], [], []
    for key, ls_fp_image in d_data.items():
        ls_train, ls_val, ls_test = split_list(ls_fp_image)
        ls_train_set.extend(ls_train)
        ls_val_set.extend(ls_val)
        ls_test_set.extend(ls_test)
        d_metadata['counts']['train'][key] = len(ls_train)
        d_metadata['counts']['val'][key] = len(ls_val)
        d_metadata['counts']['test'][key] = len(ls_test)

    d_metadata['train_count'] = len(ls_train_set)
    d_metadata['val_count'] = len(ls_val_set)
    d_metadata['test_count'] = len(ls_test_set)

    classes_name = sorted(os.listdir(conf.data.dir))
    return d_data, ls_train_set, ls_val_set, ls_test_set, d_metadata, classes_name

class ImageDatasetBinsho(Dataset):
    def __init__(self, data, transform, classes):
        self.data = data
        self.transform = al.Compose(transform)
        self.classes = classes

    def __len__(self): return len(self.data)
    
    def __getitem__(self, index):
        fp_img, y  = self.data[index]
        y_label = torch.tensor(int(y))
        x_image = np.array(Image.open(fp_img)) # rgb format!
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

        if os.path.exists('/workspace/current_dataset'):
            shutil.rmtree('/workspace/current_dataset')
    
    def prepare_data(self) -> None:
        # set clearml and download the data
        if not self.prepare_data_has_downloaded:
            if os.path.exists(self.conf.data.dir):
                shutil.rmtree(self.conf.data.dir)
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
                    d_train = self.__extract_list_link_dataset_yaml()
                    print('test',d_train.keys())
                    s3_api = MinioDatasetDownloader(dataset=d_train, download_dir='/workspace/current_dataset')
                    start_time = time.time()
                    s3_api.download_dataset()
                    end_time = time.time()
                    self.data_dir = self.conf.data.dir = '/workspace/current_dataset'
                    duration = end_time - start_time
                    print('how long it take dataset.yaml :', round(duration, 3), 'seconds')
                else:
                    print('dataset_id is dowloading',)
                    DatasetClearML.get(dataset_id=self.conf.data.dataset).get_mutable_local_copy(target_folder='/workspace/current_dataset', overwrite=True)
                print('success download data:', self.conf.data.dir)
                self.prepare_data_has_downloaded = True
                self.ddata_by_label, self.ls_train_set, self.ls_val_set, self.ls_test_set, self.d_metadata, self.classes_name = get_list_data(conf=self.conf)
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
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=8)
    
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
        y_test = [value for value in d_metadata['counts']['val'].values()]
        y_val = [value for value in d_metadata['counts']['test'].values()]

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

        ls_url_files_train = []
        # get/download list-data
        for path_dataset in datasets_yaml['dataset-train']:
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
                print('CHECK THIS DATA')
                continue
            ls_url_files_train.extend(ls_files)
            ls_files = None
            print('-----')
        
        # datasets_yaml.get('dataset-test', False)
        
        d_train = {}
        print('get_files..')
        for d_file in ls_url_files_train:
            url_file = d_file['name']
            class_name  = url_file.split('/')[-2]
            if class_name not in d_train.keys():
                d_train[class_name] = []
            d_train[class_name].append(url_file)
        return d_train
    

    def __revert_transforms(self, tensor, mean, std):
        # Convert the tensor to a NumPy array and transpose from CHW to HWC format
        image_np = tensor.numpy().transpose(1, 2, 0)

        # Denormalize the pixel values
        image_np = (image_np * std) + mean

        # Convert the float values to uint8
        image_np = (image_np * 255).astype(np.uint8)

        return image_np

    def visualize_augmented_images(self, section:str, num_images=5):
        print(f'vizualizing sample {section}...')
        ls_viz_data = []
        for label, ls_fp_image in self.ddata_by_label.items():
            ls_viz_data.extend(ls_fp_image[0:num_images])

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
            # img_np = self.__revert_transforms(tensor=image, mean=self.conf.aug.mean, std=self.conf.aug.std)
            Task.current_task().get_logger().report_image(f"{section}", f"{label_name}_{i}", iteration=1, image=image_array)

