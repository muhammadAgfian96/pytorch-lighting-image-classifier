# import sys
# sys.path.append('/mnt/hdd_2/agfian/common-project/classifier/pytorch-lighting-image-classifier/')

import clearml
import json
import os
from rich import print
from clearml import Dataset, Task, StorageManager
from src.schema.coco_json import CocoFormat
from src.helper.data_helper import MinioDatasetDownloader
import random
from collections import defaultdict
random.seed(1)

dataset_input = 'f580a16f0d8e466d9ec79d12d15e071d|[*:3]'

# rules:
"""
key:value_limit

[default]
*:-1 = include all_class


[key]
* = include all_class
-* = exclude all_class
class_name = affeted only all class

- = exclude the class

[value_limit]
all = no limit
X = limit with X number

[case]
*:-1 = default
*:-1, class_1:X, -class_2:-1
"""


def read_json(filepath):
    with open(filepath) as json_file:
        data = json.load(json_file)
    return data

def __regex_input(dataset_input):
    dataset_id, limit_raw = dataset_input.replace(' ', '').split('|')
    print('dataset_id:', dataset_id)
    limit_raw = limit_raw.replace('[','').replace(']','').replace(' ','').split(',')
    print('limit_raw:', limit_raw)

    return dataset_id, limit_raw

def __query_dataset_id(limit_raw, d_urls_by_cat):
    d_limit_query = {limit.split(':')[0]:limit.split(':')[-1] for limit in limit_raw}
    print('d_limit query: ',d_limit_query)

    d_new_urls_by_cat = defaultdict(list)
    for class_lbl, list_url in d_urls_by_cat.items():
        print(class_lbl, len(list_url))
        random.shuffle(list_url)
        
        all_class_limit_include = d_limit_query.get('*', False)
        all_class_limit_exclude = d_limit_query.get('-*', False)
        is_exclude_class = False

        print('all_class_limit:', all_class_limit_include)
        if all_class_limit_include:
            if d_limit_query.get(class_lbl.lower(), False):
                num_class_limit = d_limit_query[class_lbl.lower()]
                if num_class_limit == 'all':
                    list_url = list_url
                else:
                    list_url = list_url[:int(num_class_limit)]
            elif d_limit_query.get('-'+class_lbl.lower(), False):
                num_class_limit = d_limit_query['-'+class_lbl.lower()]
                if num_class_limit == 'all':
                    list_url = []
                    is_exclude_class = True
                else:
                    list_url = list_url[int(num_class_limit):]
            else:
                # if no limit for spesific class, then use all_class_limit
                if all_class_limit_include == 'all':
                    list_url = list_url
                else:
                    list_url = list_url[:int(all_class_limit_include)]
        
        if all_class_limit_exclude:
            if d_limit_query.get(class_lbl.lower(), False):
                num_class_limit = d_limit_query[class_lbl.lower()]
                if num_class_limit == 'all':
                    list_url = list_url
                else:
                    list_url = list_url[:int(num_class_limit)]
            else:
                if all_class_limit_exclude == 'all':
                    list_url = list_url
                    is_exclude_class = True
                else:
                    list_url = list_url[int(all_class_limit_exclude):]

        if not is_exclude_class:
            d_new_urls_by_cat[class_lbl] = list_url
    return d_new_urls_by_cat


def get_local_dataset(dataset_input):

    # Extract dataset ID and limit query
    dataset_id, limit_query = __regex_input(dataset_input)

    # Get dataset
    dataset = Dataset.get(dataset_id=dataset_id)
    path_dir_ds = dataset.get_local_copy()

    # Find the first file annotations in the directory
    filename_coco = os.listdir(path_dir_ds)[0]
    path_ds_coco = os.path.join(path_dir_ds, filename_coco)
    print('path_ds:', path_ds_coco)

    # Read and process COCO format data
    coco_data = read_json(path_ds_coco)
    coco_format = CocoFormat(**coco_data)
    print('basic:', coco_format.info.summary.basic.list_names_categories)
    print('coco:', coco_format.info.summary.distribution_categories)

    # Get image URLs by category
    urls_by_category = coco_format.get_img_urls_by_category()

    # Apply query and get filtered URLs by category
    filtered_urls_by_category = __query_dataset_id(limit_query, urls_by_category)

    # Download the dataset
    minio_downloader = MinioDatasetDownloader(dataset=filtered_urls_by_category, download_dir='./datadir-debug')
    minio_downloader.download_dataset()

    # Print the count of URLs by category
    print('\nd_new_urls_by_cat:', {key: len(urls) for key, urls in filtered_urls_by_category.items()})


if __name__ == '__main__':
    print('start')
    get_local_dataset(dataset_input)
    print('done')