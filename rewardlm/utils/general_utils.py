import torch

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

