#!/bin/env python

import torch
import test_import

print('This is a simple job!')

test_import.print_random()

if torch.cuda.is_available():
    print(f'Cuda compatible device found: {torch.cuda.get_device_name()}')
else:
    print('No cuda device found!')

from rewardlm import ToxicityMeter

print(f'class {ToxicityMeter} imported')