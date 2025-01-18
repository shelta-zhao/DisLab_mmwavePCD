"""
    Author        : Shelta Zhao(赵小棠)
    Email         : xiaotang_zhao@outlook.com
    Copyright (C) : NJU DisLab, 2025.
    Description   : Traditon pipline to generate Point Cloud Data (PCD) from raw radar data.
"""

import os
import yaml
import torch
from module.fft_process import FFTProcessor