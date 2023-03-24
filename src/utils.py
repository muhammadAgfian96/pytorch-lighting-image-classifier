from clearml import StorageManager, Task
from config.default import TrainingConfig
import yaml
import json
from clearml import Task, OutputModel, StorageManager
import os
from os.path import join
import cv2
import random
import plotly.graph_objs as go

def export_upload_model(conf, path_weights, name_upload, framework):
    try:
        print(f'Uploading {name_upload}-{framework} >>>'.upper())
        output_model = OutputModel(
            task=Task.current_task(), 
            name=name_upload, 
            framework=framework, 
        )
        extenstion = os.path.basename(path_weights).split('.')[-1]
        output_model.update_weights(
            weights_filename=path_weights,
            target_filename=f'{name_upload}-{conf.net.architecture}.{extenstion}' # it will name output
        )
        output_model.update_design(config_dict={'net': conf.net.architecture, 'input_size': conf.data.input_size})
    except Exception as e:
        print(f'Error Upload {name_upload}-{framework}'.upper(), e)


def receive_data_from_pipeline(args_from_pipeline):
    path_config_yaml = StorageManager.download_file(remote_url=args_from_pipeline['config_yaml'], local_folder='./tmp', overwrite=True)
    path_dataset_json = StorageManager.download_file(remote_url=args_from_pipeline['datasets_url'], local_folder='./tmp', overwrite=True)
    path_reports  = StorageManager.download_file(remote_url=args_from_pipeline['reports_url'])
    
    print(args_from_pipeline['datasets_url'])
    path_config_yaml = Task.current_task().connect_configuration(path_config_yaml, 'Config_YAML')
    print('path_config_yaml', path_config_yaml)
    print('path_dataset_json', path_dataset_json)
    
    d_config_yaml = read_yaml(path_config_yaml)
    d_datasets_json = read_json(path_dataset_json)
    
    print('path_config_yaml', path_config_yaml, type(d_config_yaml))
    print('path_dataset_json', path_dataset_json, type(d_datasets_json))
    print('d_config_yaml', d_config_yaml.keys())
    print('d_datasets_json', d_datasets_json.keys())

    # download all data
    print('Download all data...')
    all_dataset = d_datasets_json['dataset']
    small_dataset = d_datasets_json['small_dataset']

    d_report = d_config_yaml(path_reports)
    return d_config_yaml, d_datasets_json, d_report

def override_config(
        params:dict, 
        conf:TrainingConfig
    ):
    conf.aug.augmentor = params['aug']['augmentor']
    conf.aug.type_executions = params['aug']['type_executions']
    conf.data.batch = params['data']['batch']
    conf.data.dir = params['data']['dir']
    conf.data.dataset = params['data'].get('dataset', conf.data.dataset)
    conf.data.input_resize = params['data']['input_resize']
    conf.data.input_size = params['data']['input_size']
    conf.data.random_seed = params['data']['random_seed']
    conf.data.test_ratio = params['data']['test_ratio']
    conf.data.train_ratio = params['data']['train_ratio']
    conf.data.val_ratio = params['data']['val_ratio']
    # conf.db.bucket_dataset = params['db']['bucket_dataset']
    # conf.db.bucket_experiment = params['db']['bucket_experiment']
    conf.hyp.epoch = params['hyp']['epoch']
    conf.hyp.base_learning_rate = params['hyp']['base_learning_rate']
    conf.hyp.lr_scheduler = params['hyp']['lr_scheduler']
    conf.hyp.lr_step_size = params['hyp']['lr_step_size']
    conf.hyp.lr_decay_rate = params['hyp']['lr_decay_rate']
    # conf.hyp.lr_step_milestones = params['hyp']['lr_step_milestones']
    conf.hyp.opt_momentum = params['hyp']['opt_momentum']
    conf.hyp.opt_name = params['hyp']['opt_name']
    conf.hyp.opt_weight_decay = params['hyp']['opt_weight_decay']
    conf.hyp.precision = params['hyp']['precision']
    conf.net.architecture = params['net']['architecture']
    conf.net.dropout = params['net']['dropout']
    conf.net.checkpoint_model = params['net']['checkpoint_model']
    conf.net.pretrained = params['net']['pretrained']
    conf.net.resume = params['net']['resume']
    # if not args_pipeline['inside_pipeline']:
    #     conf.data.category = sorted(os.listdir(conf.data.dir))
    #     conf.net.num_class = len(os.listdir(conf.data.dir))
    return conf

def get_classes_from_s3_folder(dataset_url):
    ls_files = StorageManager.list(remote_url=dataset_url,
        return_full_path=True,
        with_metadata=True
    )
    ls_classe = set()
    for file in ls_files:
        classes = file.get('name').split('/')[-2]
        ls_classe.add(classes)
    return list(ls_classe)

def read_yaml(yaml_path:str) -> dict:
    with open(yaml_path, 'r') as file:
        d = yaml.safe_load(file)
    return d

def read_json(json_path:str) -> dict:
    d = {}
    with open(json_path) as file:
        file_contents = file.read()        
        d = json.loads(file_contents)
    return d

# ---------------------- Data ------------------
def check_image_health(file_path):
    """
    Check if the image is not corrupt.
    """
    try:
        img = cv2.imread(file_path)
        h, w, c = img.shape
        if h > 0 and w > 0:
            return True
    except Exception as e:
        print('[ERROR] Image Corrupt: ', file_path, e)
        return False
    
def split_dataset(images, train_ratio, val_ratio):
    """
    Split the dataset into train, validation, and test sets.
    """
    num_images = len(images)
    train_count = int(train_ratio * num_images)
    val_count = int(val_ratio * num_images)
    random.shuffle(images)
    train = images[:train_count]
    val = images[train_count:train_count + val_count]
    test = images[train_count + val_count:]
    return train, val, test

def get_list_data(config: TrainingConfig):
    test_dir = '/workspace/current_dataset_test'
    dedicated_test_dataset = os.path.exists(test_dir)

    train_ratio = config.data.train_ratio
    val_ratio = config.data.val_ratio
    test_ratio = config.data.test_ratio

    if dedicated_test_dataset:
        val_ratio += test_ratio
        test_ratio = 0.0
    print(train_ratio, val_ratio, test_ratio)

    metadata = {
        'ratio': [train_ratio, val_ratio, test_ratio],
        'counts': {
            'train': {},
            'val': {},
            'test': {},
        }
    }

    class_names = sorted(os.listdir(config.data.dir))

    if dedicated_test_dataset:
        class_names_test = sorted(os.listdir(test_dir))

    data = {label: [] for label in class_names}
    train_set, val_set, test_set = [], [], []

    for label in class_names:
        label_folder = join(config.data.dir, label)
        for file in os.listdir(label_folder):
            image_file = join(label_folder, file)
            if check_image_health(image_file):
                data[label].append((image_file, class_names.index(label)))

    if dedicated_test_dataset:
        test_data = {label: [] for label in class_names_test}
        for label in class_names_test:
            label_folder = join(test_dir, label)
            for file in os.listdir(label_folder):
                image_file = join(label_folder, file)
                if check_image_health(image_file):
                    test_data[label].append((image_file, class_names.index(label)))

    for key, images in data.items():
        train, val, test = split_dataset(images, train_ratio, val_ratio)
        train_set.extend(train)
        val_set.extend(val)
        if dedicated_test_dataset:
            val_set.extend(test)
        else:
            test_set.extend(test)

        metadata['counts']['train'][key] = len(train)
        metadata['counts']['val'][key] = len(val)
        if not dedicated_test_dataset:
            metadata['counts']['test'][key] = len(test)

    if dedicated_test_dataset:
        test_set = []
        for key, images in test_data.items():
            metadata['counts']['test'][key] = len(images)
            test_set.extend(images)

    metadata['train_count'] = len(train_set)
    metadata['val_count'] = len(val_set)
    metadata['test_count'] = len(test_set)

    return data, train_set, val_set, test_set, metadata, class_names




def get_properties_from_local_path(d_data, local_path):
    class_name = local_path.split('/')[-2]
    filename = os.path.basename(local_path)
    ext = filename.split('.')[-1]
    unique_id = filename.split(f'.{ext}')[0]
    ls_urls = d_data[class_name]
    
    url_fix = None
    for urls in ls_urls:
        if unique_id in urls:
            url_fix = urls
            break
    return class_name, unique_id, url_fix

def map_data_to_dict(d_data, local_path_dir):
    mapped = []
    list_save_local = [os.path.join(subdir, file) for subdir, _, files in os.walk(local_path_dir) for file in files]
    for i, local_path in enumerate(list_save_local):
        class_name, unique_id, url_fix = get_properties_from_local_path(d_data, local_path)
        mapped.append((url_fix, class_name, local_path))
    return mapped


def make_graph_performance(torchscript_performance, onnx_performance):
    print('Generating Graph Performance')
    def compare(ref_val, val):
        try:
            perbandingan = round( val/ref_val * 100.0, 2)
            if perbandingan > 100:
                return 100 + perbandingan
            if perbandingan < 100:
                return perbandingan
            if perbandingan == 100:
                return 100
        except Exception as e:
            print(e)
            return 100

    # Define the data
    onnx_accuracy = onnx_performance['accuracy']
    onnx_speed = onnx_performance['speed']
    onnx_vram = onnx_performance['vram']
    onnx_ram = onnx_performance['ram']
    onnx_total_time = onnx_performance['total_prediction']

    torchscript_accuracy = torchscript_performance['accuracy']
    torchscript_speed = torchscript_performance['speed']
    torchscript_vram = torchscript_performance['vram']
    torchscript_ram = torchscript_performance['ram']
    torchscript_total_time = torchscript_performance['total_prediction']

    # Convert speed values to percentage of reference
    reference = 100.0
    onnx_speed_percent = compare(torchscript_speed, onnx_speed)
    onnx_accuracy_percent = compare(torchscript_accuracy, onnx_accuracy) 
    onnx_vram_percent = compare(torchscript_vram, onnx_vram) 
    onnx_ram_percent = compare(torchscript_ram, onnx_ram)
    onnx_total_time_percent = compare(torchscript_total_time, onnx_total_time)

    # Define the data traces
    data = [
        {
            'y': ['Accuracy', 'Speed', 'VRAM', 'RAM', 'Total Time'],
            'x': [onnx_accuracy_percent, onnx_speed_percent, onnx_vram_percent, onnx_ram_percent, onnx_total_time_percent],
            'name': 'ONNX', 'type': 'bar', 'orientation': 'h',
            'marker': {
                'color': 'rgb(166,206,227)',
                'line': {
                    'color': 'rgb(54, 55, 56)',
                    'width': 1.5,
                }
            },
            'text': [
                '{:.2f}%'.format(onnx_accuracy), 
                '{:.4f} img/sec'.format(onnx_speed), 
                '{:.1f} MB'.format(onnx_vram), 
                '{:.3f} MB'.format(onnx_ram),
                '{:.3f} secs'.format(onnx_total_time)
            ],
            'textposition': 'inside',
            'insidetextanchor': 'middle',
            'textfont': {
                'color': '#ffffff'
            }
        },
        {
            'y': ['Accuracy', 'Speed', 'VRAM', 'RAM', 'Total Time'],
            'x': [reference, reference, reference, reference, reference],
            'name': 'TorchScript', 'type': 'bar', 'orientation': 'h',
            'marker': {
                'color': 'rgb(253,191,111)',
                'line': {
                    'color': 'rgb(54, 55, 56)',
                    'width': 1.5
                }
            },
            'text': [
                '{:.2f}%'.format(torchscript_accuracy), 
                '{:.4f} img/sec'.format(torchscript_speed), 
                '{:.1f} MB'.format(torchscript_vram), 
                '{:.3f} MB'.format(torchscript_ram),
                '{:.3f} secs'.format(torchscript_total_time)
            ],
            'textposition': 'inside',
            'insidetextanchor': 'middle',
            'textfont': {
                'color': '#ffffff'
            }
        }
    ]

    # Create the layout for the bar chart
    layout = go.Layout(
        title='Comparison of ONNX and TorchScript',
        xaxis=dict(title='Value'),
        # width=800, height=500,
        plot_bgcolor='rgba(240, 240, 240, 0.95)'
    )

    # Create the bar chart
    fig = go.Figure(data=data, layout=layout)

    # Update the font size of the text on the chart
    fig.update_layout(font=dict(size=12))

    # Add a reference line for accuracy
    fig.add_shape(
        type='line',
        x0=100, y0=-0.5,  x1=100, y1=3.5,
        line=dict(
            color='rgb(34, 67, 115)', width=2, dash='dash'
        )
    )

    return fig


if __name__ == '__main__':
    path_yaml = '../config/datasets.yaml'
    print(read_yaml(path_yaml))