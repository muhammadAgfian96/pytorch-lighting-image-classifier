import cv2
import numpy as np
import onnx
import onnxruntime
import torch
import albumentations as al
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningModule
from rich import print
from scipy.special import softmax
import time
import timm
import os
import psutil
import GPUtil


class ModelPredictor:
    def __init__(self, input_size, mean, std):
        self.input_size = input_size
        self.mean = mean
        self.std = std
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = al.Compose([
            al.Resize(height=self.input_size, width=self.input_size, always_apply=True),
            al.Normalize(mean=self.mean, std=self.std, always_apply=True, max_pixel_value=255.0),
            ToTensorV2(transpose_mask=True),
        ])

    def preprocess_image(self, image):
        return self.transform(image=image)["image"]

    def load_onnx_model(self, model_path):
        self.model_onnx = onnxruntime.InferenceSession(model_path, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])

    def predict_onnx(self, image):
        input_tensor = self.preprocess_image(image)
        input_name = self.model_onnx.get_inputs()[0].name
        output_name = self.model_onnx.get_outputs()[0].name
        prediction = self.model_onnx.run([output_name], {input_name: input_tensor.unsqueeze(0).numpy()})[0]
        soft_predict = softmax(np.array(prediction), axis=1)
        return [f"{p:.2f}%" for p in soft_predict[0] * 100]
    
    def load_torchscript_model(self, model_path):
        self.model_torchscript = torch.jit.load(model_path)

    def predict_torchscript(self, image):
        input_tensor = self.preprocess_image(image)
        with torch.no_grad():
            prediction = self.model_torchscript(input_tensor.unsqueeze(0))
        soft_predict = softmax(prediction.numpy(), axis=1)
        return [f"{p:.2f}%" for p in soft_predict[0] * 100]

    def load_pytorch_lightning_checkpoint(self, checkpoint_path):
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        
        # Extract the hyperparameters
        hparams = checkpoint['hyper_parameters']
        net = hparams['net']
        preprocessing = hparams['preprocessing']
        
        # Create the model using timm
        self.model_pl = timm.create_model(
            net['architecture'],
            num_classes=net['num_class'],
            drop_rate=net['dropout'],
        )
        
        # Load the model weights
        self.model_pl.load_state_dict(checkpoint['state_dict'], strict=False)
        
        # Move the model to the appropriate device
        self.model_pl = self.model_pl.to(self.device)
        
        # return self.model_pl, preprocessing

    def predict_pytorch_lightning(self, image):
        input_tensor = self.preprocess_image(image)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)  # Add batch dimension

        self.model_pl.eval()
        with torch.no_grad():
            output = self.model_pl(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1).cpu().squeeze().numpy()

        return [f"{p:.2f}%" for p in probabilities * 100]

    def benchmark(self, image, n_runs=10):
        ram_usages = []
        vram_usages = []
        # image = self.preprocess_image(image)
        # Benchmark ONNX model

        # awal_ram = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        awal_ram = psutil.virtual_memory()[3]/1048576
        gpus = GPUtil.getGPUs()
        awal_vram = gpus[0].memoryUsed if len(gpus) > 0 else 0

        start_time = time.perf_counter()
        for _ in range(n_runs):
            self.predict_onnx(image)
            # ram_usages.append(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024) - awal_ram)
            ram_usages.append(psutil.virtual_memory()[3]/1048576 - awal_ram)
            current_gpus = gpus[0].memoryUsed if len(gpus) > 0 else 0
            vram_usages.append(current_gpus - awal_vram)
        onnx_duration = time.perf_counter() - start_time
        
        # Calculate average resource usage
        avg_ram_usage_onnx = sum(ram_usages) / n_runs
        avg_vram_usage_onnx = sum(vram_usages) / n_runs

        ram_usages = []
        vram_usages = []
        # awal_ram = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        awal_ram = psutil.virtual_memory()[3]/(1048576)
        gpus = GPUtil.getGPUs()
        awal_vram = gpus[0].memoryUsed if len(gpus) > 0 else 0
        
        # Benchmark TorchScript model
        start_time = time.perf_counter()
        for _ in range(n_runs):
            self.predict_torchscript(image)
            # ram_usages.append(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024) - awal_ram)
            ram_usages.append(psutil.virtual_memory()[3]/(1048576) - awal_ram)
            gpus = GPUtil.getGPUs()
            current_gpus = gpus[0].memoryUsed if len(gpus) > 0 else 0
            vram_usages.append(current_gpus - awal_vram)
        torchscript_duration = time.perf_counter() - start_time
        
        # Calculate average resource usage
        avg_ram_usage_torchscript = sum(ram_usages) / n_runs
        avg_vram_usage_torchscript = sum(vram_usages) / n_runs


        # Benchmark PyTorch Lightning model
        ram_usages = []
        vram_usages = []
        # awal_ram = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        awal_ram = psutil.virtual_memory()[3]/(1048576)
        gpus = GPUtil.getGPUs()
        awal_vram = gpus[0].memoryUsed if len(gpus) > 0 else 0

        start_time = time.perf_counter()
        for _ in range(n_runs):
            self.predict_pytorch_lightning(image)
            # ram_usages.append(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024) - awal_ram)
            ram_usages.append(psutil.virtual_memory()[3]/(1048576) - awal_ram)
            gpus = GPUtil.getGPUs()
            current_gpus = gpus[0].memoryUsed if len(gpus) > 0 else 0
            vram_usages.append(current_gpus - awal_vram)
        pytorch_lightning_duration = time.perf_counter() - start_time
                
        # Calculate average resource usage
        avg_ram_usage_pl = sum(ram_usages) / n_runs
        avg_vram_usage_pl = sum(vram_usages) / n_runs
        
        print(f"ONNX Model: {onnx_duration / n_runs:.6f} seconds per prediction")
        print(f"TorchScript Model: {torchscript_duration / n_runs:.6f} seconds per prediction")
        print(f"PyTorch Lightning Model: {pytorch_lightning_duration / n_runs:.6f} seconds per prediction")

        return {
                'onnx': {
                    'duration': round(onnx_duration/n_runs, 5),
                    'avg_ram_usage': round(avg_ram_usage_onnx, 5),
                    'avg_vram_usage': round(avg_vram_usage_onnx, 5)
                },
                'torchscript': {
                    'duration': round(torchscript_duration/n_runs, 5),
                    'avg_ram_usage': round(avg_ram_usage_torchscript, 5),
                    'avg_vram_usage': round(avg_vram_usage_torchscript, 5)
                },
                'pytorchlightning': {
                    'duration': round(pytorch_lightning_duration/n_runs, 5),
                    'avg_ram_usage': round(avg_ram_usage_pl, 5),
                    'avg_vram_usage': round(avg_vram_usage_pl, 5)
                }
            }

if __name__ == '__main__':
    # Replace 'ModelClass' with your actual model class

    predictor = ModelPredictor(input_size=224, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    path_image = 'current_dataset/Bean/0036.jpg'
    # For ONNX model
    predictor.load_onnx_model("exports/onnx-edgenext_x_small.onnx")
    onnx_prediction = predictor.predict_onnx(cv2.imread(path_image))
    print('onnx_prediction',onnx_prediction)

    # For TorchScript model
    predictor.load_torchscript_model("exports/torchscript-edgenext_x_small.pt")
    torchscript_prediction = predictor.predict_torchscript(cv2.imread(path_image))
    print('torchscript_prediction',torchscript_prediction)

    # For PyTorch Lightning checkpoint
    predictor.load_pytorch_lightning_checkpoint("exports/best-ckpt-edgenext_x_small.ckpt")
    pytorch_lightning_prediction = predictor.predict_pytorch_lightning(cv2.imread(path_image))
    print('pytorch_lightning_prediction',pytorch_lightning_prediction)
    
    # result_bench = predictor.benchmark(
    #     image=cv2.imread(path_image),
    #     n_runs=200
    # )

    # print(result_bench)