import random
import torch
import pytorch_lightning as pl
import albumentations as al
import os
from os.path import join
from PIL import Image
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

from clearml import Dataset as DatasetClearML
from config.default import TrainingConfig
from rich import print


def get_list_data(root_path, conf:TrainingConfig):
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


    classes_name = sorted(os.listdir(root_path))

    d_data = {lbl:[] for lbl in classes_name}
    ls_train = []
    ls_val = []
    ls_test = []

    for label in classes_name:
        fp_folder = join(root_path, label)
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
        d_metadata['counts']['val'][key] = len(ls_train)
        d_metadata['counts']['test'][key] = len(ls_train)

    d_metadata['train_count'] = len(ls_train_set)
    d_metadata['val_count'] = len(ls_val_set)
    d_metadata['test_count'] = len(ls_test_set)

    classes_name = sorted(os.listdir(root_path))
    return ls_train_set, ls_val_set, ls_test_set, d_metadata, classes_name

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
    def __init__(self, conf:TrainingConfig):
        super().__init__()
        self.data_dir = conf.data.dir
        self.conf = conf
    
    def prepare_data(self) -> None:
        # set clearml and download the data
        try:
            os.makedirs('/workspace/current_dataset', exist_ok=True)
            print('creted folder, downloading dataset...')
            DatasetClearML.get(dataset_id=self.conf.data.dataset_id).get_mutable_local_copy(target_folder='/workspace/current_dataset', overwrite=True)
        except Exception as e:
            print(e)

    def setup(self, stage: str):
        # get list of data
        ls_train_set, ls_val_set, ls_test_set, d_metadata, classes_name = get_list_data(root_path=self.data_dir, conf=self.conf)
        
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.data_train = ImageDatasetBinsho(
                ls_train_set, 
                transform=self.conf.aug.get_ls_train(),
                classes=classes_name)
            self.data_val = ImageDatasetBinsho(
                ls_val_set, 
                transform=self.conf.aug.get_ls_train(),
                classes=classes_name)
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.data_test = ImageDatasetBinsho(
                ls_test_set, 
                transform=self.conf.aug.get_ls_train(),
                classes=classes_name)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.conf.data.batch, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.conf.data.batch)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.conf.data.batch)