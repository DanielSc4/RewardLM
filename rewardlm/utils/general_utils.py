import torch
import yaml
from pynvml import *

def device_selector():
    """_summary_

    Returns:
        str: device type available [cuda, mps, cpu]
    """
    # selecting device based on whats available
    device = ''
    if torch.cuda.is_available():
        print('Using Nvidia GPU')
        print(torch.cuda.get_device_name())
        device = 'cuda'
    # Only on torch night for Apple M1 GPU
    elif torch.backends.mps.is_available():
        print('Using MPS (Apple Silicon GPU)')
        device = 'mps'
    else:
        print('Using CPU, :(')
        device = 'cpu'
    return device


def load_config(path: str = './configs/', name: str = ''):
    assert name != '', 'Please use a valid config'

    with open(path + name + '.yaml', "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    
    return cfg


def print_gpu_utilization():
    if torch.cuda.is_available():
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU memory occupied: {info.used//1024**2} MB.")
    else:
        print('GPU not found')


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()