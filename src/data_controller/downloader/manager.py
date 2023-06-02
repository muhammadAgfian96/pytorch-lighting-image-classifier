import shutil
import os
from src.data_controller.downloader.sub_clearml_ds import ClearmlDatasetDownloader
from src.data_controller.downloader.sub_s3_directory_ds import S3DirectoryDownloader
from src.data_controller.utils import MinioDatasetDownloader
from rich import print
import yaml

class DownloaderManager:
    def __init__(self):
        self.clearml_downloader = ClearmlDatasetDownloader()
        self.minio_downloader = MinioDatasetDownloader()
        self.s3_dir_downloader = S3DirectoryDownloader()

    def __create_output_dir(self, output_dir):
        if os.path.exists(output_dir):
            print("🧹 cleaning old dataset...")
            shutil.rmtree(output_dir)
            print("deteled.")

        path_dir_train = os.path.join(output_dir, 'train')
        path_dir_test = os.path.join(output_dir, 'test')
        os.makedirs(path_dir_train, exist_ok=True)
        os.makedirs(path_dir_test, exist_ok=True)
        return path_dir_train, path_dir_test
    
    def read_dataset_yaml(self, yaml_path):
        with open(yaml_path, "r") as file:
            d_dataset = yaml.safe_load(file)
        
        if d_dataset.get('dataset-test') == [None]:
            d_dataset['dataset-test'] = None

        return d_dataset

    def __download_datasets_from_yaml(self, ls_dataset, output_dir_section):
        for number, ds in enumerate(ls_dataset, start=1):
            if '/' in ds:
                print(f'{number}. 🌐 YAML: S3 Directory')
                # urls fetcher
                self.s3_dir_downloader.download(
                    craetion_minio_downloader=self.minio_downloader,
                    input_dataset=ds, 
                    output_dir_section=output_dir_section
                )
            else:
                # clearml id with coco fetcher
                print(f'{number}. 🇨  YAML: clearml_id')
                self.clearml_downloader.download(
                    creation_minio_downloader=self.minio_downloader,
                    dataset_input=ds,
                    output_dir=output_dir_section
                )

    def fetch(self, input_dataset, output_dir):
        print('[FETCHING DATASET]>>>>>>>>>>>>>>>>>>>>>')
        path_dir_train, path_dir_test = self.__create_output_dir(output_dir)
        
        if 's3://' in input_dataset:
            print('🔨 Working on S3-Directory-Single-Link Mode')
            print('\n[TRAINING DATASET]')
            self.s3_dir_downloader.download(
                input_dataset=input_dataset,
                output_dir_section=path_dir_train
            )
            print("🚫 NO TESTING DATASET 🚫")
        
        elif '.yaml' in input_dataset or '.yml' in input_dataset:
            print('🔨 Working on YAML Mode')
            # 1 read yaml, ouput list of url/ids
            d_dataset = self.read_dataset_yaml(input_dataset)

            if d_dataset.get('dataset-train'):
                print('\n[TRAINING DATASET]')
                self.__download_datasets_from_yaml(
                    d_dataset['dataset-train'], 
                    output_dir_section=path_dir_train
                )

            if d_dataset.get('dataset-test'):
                print('\n[TESTING DATASET]')
                self.__download_datasets_from_yaml(
                    d_dataset['dataset-test'], 
                    output_dir_section=path_dir_test
                )
            else:
                print("🚫 NO TESTING DATASET 🚫")
        else:
            raise Exception('input_dataset is not valid')

        return path_dir_train, path_dir_test

if __name__ == '__main__':
    input_dataset = 's3://10.8.0.66:9000/app-data-workflow/dataset-playground/Vegetables/test'
    input_dataset = '/mnt/hdd_2/agfian/common-project/classifier/pytorch-lighting-image-classifier/config/datasetsv2.yaml'
    output_dir = './dataset-testing'
    dir_train, dir_test = DownloaderManager().fetch(input_dataset, output_dir)    
    print(dir_train, dir_test)