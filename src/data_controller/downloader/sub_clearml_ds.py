# import sys
# sys.path.append('/mnt/hdd_2/agfian/common-project/classifier/pytorch-lighting-image-classifier/')

import json
import os
import random
from collections import defaultdict

import clearml
from clearml import Dataset, StorageManager, Task
from rich import print

from src.data_controller.utils import MinioDatasetDownloader
from src.schema.coco_json import CocoFormat

dataset_input = "f580a16f0d8e466d9ec79d12d15e071d|[*:3]"

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
class ClearmlDatasetDownloader:
    def __init__(self):
        random.seed(1)
        pass

    def __read_json(self, filepath:str) -> dict:
        with open(filepath) as json_file:
            data = json.load(json_file)
        return data

    def __extract_input(self, dataset_input:str):
        result = dataset_input.replace(" ", "").split("|")
        if len(result) == 2:
            dataset_id, limit_raw = result
            print("\t🆔 dataset:", dataset_id)
            limit_raw = limit_raw.replace("[", "").replace("]", "").replace(" ", "").split(",")
            print(f"\t🛢️ limit_raw: -> {limit_raw}")
            return dataset_id, limit_raw
        
        if len(result) == 1:
            dataset_id = result[0]
            limit_raw = None 
            return dataset_id, limit_raw
        raise Exception("dataset_input is not valid")

    def __query_dataset_id(self, limit_raw:list, d_urls_by_cat):
        d_limit_query = {limit.split(":")[0]: limit.split(":")[-1] for limit in limit_raw}
        print(f"\t📖 d_limit query -> {d_limit_query}")

        d_new_urls_by_cat = defaultdict(list)
        for class_lbl, list_url in d_urls_by_cat.items():
            # print(class_lbl, len(list_url))
            random.shuffle(list_url)

            all_class_limit_include = d_limit_query.get("*", False)
            all_class_limit_exclude = d_limit_query.get("-*", False)
            is_exclude_class = False

            # print("all_class_limit:", all_class_limit_include)
            if all_class_limit_include:
                if d_limit_query.get(class_lbl.lower(), False):
                    num_class_limit = d_limit_query[class_lbl.lower()]
                    if num_class_limit == "all":
                        list_url = list_url
                    else:
                        list_url = list_url[: int(num_class_limit)]
                elif d_limit_query.get("-" + class_lbl.lower(), False):
                    num_class_limit = d_limit_query["-" + class_lbl.lower()]
                    if num_class_limit == "all":
                        list_url = []
                        is_exclude_class = True
                    else:
                        list_url = list_url[int(num_class_limit) :]
                else:
                    # if no limit for spesific class, then use all_class_limit
                    if all_class_limit_include == "all":
                        list_url = list_url
                    else:
                        list_url = list_url[: int(all_class_limit_include)]

            if all_class_limit_exclude:
                if d_limit_query.get(class_lbl.lower(), False):
                    num_class_limit = d_limit_query[class_lbl.lower()]
                    if num_class_limit == "all":
                        list_url = list_url
                    else:
                        list_url = list_url[: int(num_class_limit)]
                else:
                    if all_class_limit_exclude == "all":
                        list_url = list_url
                        is_exclude_class = True
                    else:
                        list_url = list_url[int(all_class_limit_exclude) :]

            if not is_exclude_class:
                d_new_urls_by_cat[class_lbl] = list_url
        return d_new_urls_by_cat

    def download(
            self, 
            creation_minio_downloader:MinioDatasetDownloader, 
            dataset_input:str, 
            output_dir:str="./datadir-debug",
            exclude_tags:list=[]
        ):
        """
        return outputdir
        """
        # Extract dataset ID and limit query
        dataset_id, limit_query = self.__extract_input(dataset_input)

        # Get dataset
        dataset = Dataset.get(dataset_id=dataset_id)
        path_dir_ds = dataset.get_local_copy()

        # Find the first file annotations in the directory
        filename_coco = os.listdir(path_dir_ds)[0]
        path_ds_coco = os.path.join(path_dir_ds, filename_coco)
        # print("\tpath_coco_json:", path_ds_coco)

        # Read and process COCO format data
        coco_data = self.__read_json(path_ds_coco)
        coco_format = CocoFormat(**coco_data)
        # print("list_:", coco_format.info.summary.basic.list_names_categories)
        print(f"\t distribution_raw -> {coco_format.info.summary.distribution_categories}")

        # Get image URLs by category
        urls_by_category = coco_format.get_img_urls_by_category(exclude_tags=exclude_tags)

        # Apply query and get filtered URLs by category
        if limit_query is None:
            filtered_urls_by_category = urls_by_category
        else:
            filtered_urls_by_category = self.__query_dataset_id(limit_query, urls_by_category)

        # Download the dataset
        creation_minio_downloader.download_dataset(
            dataset_dict=filtered_urls_by_category,
            output_dir=output_dir,
        )

        # Print the count of URLs by category
        dist_final = {key: len(urls) for key, urls in filtered_urls_by_category.items()}
        print(f"\tdistribution_final -> {dist_final}")
        return output_dir


if __name__ == "__main__":
    print("start")
    
    print("done")
