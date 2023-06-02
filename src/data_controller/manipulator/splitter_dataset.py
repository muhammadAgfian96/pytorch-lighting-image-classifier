from collections import defaultdict
from config.default import TrainingConfig
import os
from src.utils import check_image_health
import random
from rich import print

random.seed(1) 


def __split_dataset_based_class(images_and_cls_idx, train_ratio, val_ratio):
    """
    Split the dataset into train, validation, and test sets.
    """
    random.shuffle(images_and_cls_idx)
    num_images = len(images_and_cls_idx)
    train_count = int(train_ratio * num_images)
    val_count = int(val_ratio * num_images)

    train = images_and_cls_idx[:train_count]
    val = images_and_cls_idx[train_count : train_count + val_count] #becare ful
    test = images_and_cls_idx[train_count + val_count :]
    return train, val, test

def __splitting_and_get_metadata(data_mapped:dict, train_ratio:float, val_ratio:float, metadata:dict, is_test_dataset_exist:bool):
    ls_train_dataset, ls_val_dataset, ls_test_dataset = [], [], []

    for label_class, ls_images_file_cls_index in data_mapped.items():
        train, val, test = __split_dataset_based_class(ls_images_file_cls_index, train_ratio, val_ratio)

        if is_test_dataset_exist:
            val.extend(test)
            test = []
            
        ls_train_dataset.extend(train)
        ls_val_dataset.extend(val)
        ls_test_dataset.extend(test) # if is_test_dataset_exist `test` will be []
            
        metadata["count"]["train"][label_class] = len(train)
        metadata["count"]["val"][label_class] = len(val)
        metadata["count"]["test"][label_class] = len(test)
    return ls_train_dataset, ls_val_dataset, ls_test_dataset

def __mapping_data_dict(path_dir_section:str, class_names_train:list):
    data = defaultdict(list)
    print('🔨 Mapping data...')
    print(class_names_train)
    for single_class in class_names_train:
        single_class_dir = os.path.join(path_dir_section, single_class)

        for filename in os.listdir(single_class_dir):
            img_path = os.path.join(single_class_dir, filename)
            if check_image_health(img_path):
                data[single_class].append((img_path, class_names_train.index(single_class)))
    return data



def splitter_dataset(config: TrainingConfig, path_dir_train, path_dir_test):
    """
    Get the list of images and labels.
    return
    - data,
    - train_set, val_set, test_set,
        the_set = [
            (image_path, label),
            (image_path, label),
            (image_path, label),
        ]
    - metadata,
    - class_names_train v
    """
    # vars from config
    train_ratio = config.data.train_ratio
    val_ratio = config.data.val_ratio
    test_ratio = config.data.test_ratio

    ls_class_test = os.listdir(path_dir_test)
    ls_class_train = os.listdir(path_dir_train)

    is_test_dataset_exist = True if len(ls_class_test) > 0 else False    
    if is_test_dataset_exist:
        val_ratio += test_ratio
        test_ratio = 0.0

    class_names_train = sorted([lbl.lower() for lbl in ls_class_train])
    print(f"class_names_train -> {class_names_train}")
    # claxss_naxmes_t = sorted([lbl.lower() for lbl in ls_class_test])
    # checking datatest all class is in datatrain
    for lbl in ls_class_test:
        if lbl.lower() not in class_names_train:
            raise ValueError(
                f"⛔⛔ Class {lbl} in test dataset is not in train dataset. Please check your dataset. ⛔⛔"
            )
    print("✅✅ All class in test dataset is in train dataset. ✅✅")

    metadata = {
        "class_names": class_names_train,
        "ratio": [train_ratio, val_ratio, test_ratio],
        "count": {
            "train": {},
            "val": {},
            "test": {},
        },
        "count_section": {
            "train": 0,
            "val": 0,
            "test": 0,
        },
    }

    # check all images can be read and collect them
    data_train_mapped = __mapping_data_dict(
        path_dir_section=path_dir_train, 
        class_names_train=class_names_train
    )
    ls_train_dataset, ls_val_dataset, ls_test_dataset = __splitting_and_get_metadata(
        data_mapped=data_train_mapped,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        metadata=metadata, # ❗ be careful here metadata just updated
        is_test_dataset_exist=is_test_dataset_exist
    ) 


    if is_test_dataset_exist:
        data_test_mapped = __mapping_data_dict(path_dir_test, class_names_train)
        for label_class, ls_images_file_cls_index in data_test_mapped.items():
            metadata["count"]["test"][label_class] = len(ls_images_file_cls_index)
            ls_test_dataset.extend(ls_images_file_cls_index)

    # metadata 
    metadata["count_section"]["train"] = len(ls_train_dataset)
    metadata["count_section"]["val"] = len(ls_val_dataset)
    metadata["count_section"]["test"] = len(ls_test_dataset)

    print(metadata)
    return data_train_mapped, ls_train_dataset, ls_val_dataset, ls_test_dataset, metadata


if __name__ == "__main__":
    from config.default import TrainingConfig
    path_dir_train = "/mnt/hdd_2/agfian/common-project/classifier/pytorch-lighting-image-classifier/src/data_controller/downloader/dataset-testing/train"
    path_dir_test = "/mnt/hdd_2/agfian/common-project/classifier/pytorch-lighting-image-classifier/src/data_controller/downloader/dataset-testing/test"
    
    splitter_dataset(TrainingConfig, path_dir_train, path_dir_test)