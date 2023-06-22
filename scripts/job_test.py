#!/bin/env python

import torch

print('This is a simple job!')

if torch.cuda.is_available():
    print(f'Cuda compatible device found: {torch.cuda.get_device_name()}')
else:
    print('No cuda device found!')

