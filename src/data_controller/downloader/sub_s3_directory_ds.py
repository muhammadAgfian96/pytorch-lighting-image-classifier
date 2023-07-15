
from collections import defaultdict
from urllib.parse import urljoin

from clearml import StorageManager, Task
from rich import print

from src.data_controller.utils import MinioDatasetDownloader


class S3DirectoryDownloader:
    def __init__(self):
        pass
    
    @staticmethod
    def extract_dict_format_from_s3_url(s3_url):
        if "s3://" not in s3_url:
            s3_url = urljoin("http://10.8.0.66:9000", s3_url).replace('http', 's3')
        ls_urls = StorageManager.list(s3_url, return_full_path=True)
        dict_format = defaultdict(list)
        for url in ls_urls:
            class_name = url.split('/')[-2].lower()
            dict_format[class_name].append(url)
        return dict_format

    def download(self, craetion_minio_downloader:MinioDatasetDownloader, input_dataset, output_dir_section):
        dataset_dict_format = S3DirectoryDownloader.extract_dict_format_from_s3_url(input_dataset)
        dist_dataset= {k: len(v) for k, v in dataset_dict_format.items()}
        craetion_minio_downloader.download_dataset(
            dataset_dict=dataset_dict_format, 
            output_dir=output_dir_section
        )
        msg = f"\tdistribution_final -> {dist_dataset}"
        if len(dist_dataset) == 0:
            msg+= " ⚠️ Check PATH Dataset ⚠️"
            print(f"\tpath: {input_dataset}")
            Task.current_task().add_tags(["⚠️ dataset"])
        print(msg)
