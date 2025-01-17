"""
    Author      : Shelta Zhao(赵小棠)
    Affiliation : Nanjing University
    Email       : xiaotang_zhao@outlook.com
    Description : Perform radar process funciotns(3D-FFT, CFAR, DOA, etc).
"""

import os
import yaml
import torch
import numpy as np
import pandas as pd
from parser.param_process import get_radar_params
from parser.adc_load import get_regular_data
import matplotlib.pyplot as plt


def adc_to_pcd(regular_data, radar_params, device, save=False, load=False):
    """
    Head Function: Perform procession chain to get Point Cloud Data(pcd) from raw regular data.

    Parameters:
        ababab
    
    Returns:
        abbaababa

    """

    if load:
        print('Hello')
    else:
        # Perform range_fft
        range_fft_output = range_fft(regular_data, radar_params['rangeFFTObj'], device)
        # Perfrom doppler fft
        doppler_fft_out = doppler_fft(range_fft_output, radar_params['dopplerFFTObj'], device)



def range_fft(input, rangeFFTObj, device):
    """
    Perform Range FFT on the input data.
    
    Parameters:
        input (np.ndarray): The input data to be transformed.
        rangeFFTObj (dict): A dictionary containing the parameters for the range FFT, including:
            - 'radarPlatform' (str): The type of radar platform.
            - 'rangeFFTSize' (int): The size of the FFT.
            - 'dcOffsetCompEnable' (bool): Whether to enable DC offset compensation.
            - 'rangeWindowEnable' (bool): Whether to enable range-domain windowing.
            - 'FFTOutScaleOn' (bool): Whether to apply a scale factor to the FFT output.
            - 'scaleFactorRange' (float): The scale factor to apply if enabled.
        device (str): The device to perform the computation on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: The range FFT result. Shape: (num_frames, range_fft_size, num_chirps, num_rx, num_tx)
    """

    # Get basic fft params
    radar_type, fft_size = rangeFFTObj['radarPlatform'], rangeFFTObj['rangeFFTSize']
    dc_on, win_on = rangeFFTObj['dcOffsetCompEnable'], rangeFFTObj['rangeWindowEnable']
    scale_on, scale_factor = rangeFFTObj['FFTOutScaleOn'], rangeFFTObj['scaleFactorRange']

    # Convert input to tensorCreate output
    input = torch.tensor(input, dtype=torch.complex64).to(device)
    # Generate window coefficient
    win_coeff = torch.hann_window(input.shape[1] + 2, periodic=True).to(device)[1:-1]
    # Apply DC offset compensation
    input = input - torch.mean(input, dim=1, keepdim=True) if dc_on else input
    # Apply range-domain windowing
    input = input * win_coeff.view(-1, 1, 1, 1) if win_on else input
    # Perform FFT for each TX/RX chain
    fft_output = torch.fft.fft(input, n=fft_size, dim=1)
    # Apply scale factor
    fft_output = fft_output * scale_factor if scale_on else fft_output
    # Phase compensation for IWR6843ISK-ODS
    if radar_type == 'IWR6843ISK-ODS':
        fft_output[:, :, :, 1:3, :] *= torch.exp(-1j * torch.pi)

    # Return range fft result
    return fft_output

def doppler_fft(input, dopplerFFTObj, device):
    """
    Perform Doppler FFT on the input data (result of range fft).
    
    Parameters:
        input (torch.Tensor): The input data to be transformed.
        dopplerFFTObj (dict): A dictionary containing the parameters for the doppler FFT, including:
            - 'dopplerFFTSize' (int): The size of the FFT.
            - 'dopplerWindowEnable' (bool): Whether to enable doppler-domain windowing.
            - 'FFTOutScaleOn' (bool): Whether to apply a scale factor to the FFT output.
            - 'scaleFactorDoppler' (float): The scale factor to apply if enabled.
        device (str): The device to perform the computation on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: The doppler FFT result. Shape: (num_frames, range_fft_size, doppler_fft_size, num_rx, num_tx)
    """

    # Get the basic fft parmas
    fft_size, win_on  = dopplerFFTObj['dopplerFFTSize'], dopplerFFTObj['dopplerWindowEnable']
    scale_on, scale_factor = dopplerFFTObj['FFTOutScaleOn'], dopplerFFTObj['scaleFactorDoppler']

    # Generate window coefficient
    win_coeff = torch.hann_window(input.shape[1] + 2, periodic=True).to(device)[1:-1]
    # Apply doppler-domain windowing
    input = input * win_coeff.view(-1, 1, 1, 1) if win_on else input
    # Perform FFT for each TX/RX chain
    fft_output = torch.fft.fftshift(torch.fft.fft(input, n=fft_size, dim=2), dim=2)
    # Apply scale factor
    fft_output = fft_output * scale_factor if scale_on else fft_output

    # Return doppler fft result
    return fft_output.cpu() if device == 'cuda' else fft_output 

def angle_fft(rangeFFT_output, DOAObj):
    """
    Angle FFT
    :param rangeFFT_output: range FFT output
    :param angleFFTObj: angle FFT object
    :return: angle FFT data
    """
    angle_fft_data = angle_fft.angle_fft(rangeFFT_output)
    return angle_fft_data

def CFAR(range_fft_data, CFARObj):
    """
    CFAR
    :param range_fft_data: range FFT data
    :param CFARObj: CFAR object
    :return: CFAR data
    """
    CFAR_data = CFARObj.CFAR(range_fft_data)
    return CFAR_data


if __name__ == "__main__":
    
    # Parse data config & Get radar params
    with open("data2parse.yaml", "r") as file:
        data = yaml.safe_load(file)
    data_path = os.path.join("datas/adcDatas", f"{data['prefix']}/{data['index']}")
    config_path = os.path.join("datas/configs", data["config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get radar params
    radar_params = get_radar_params(config_path, data['radar'], load=True)

    # Get regular raw radar data
    regular_data, timestamp = get_regular_data(data_path, radar_params['readObj'], '1', timestamp=True)

    # Test Range FFT
    range_fft_output = range_fft(regular_data, radar_params['rangeFFTObj'], device)

    # Test Doppler FFT
    doppler_fft_out = doppler_fft(range_fft_output, radar_params['dopplerFFTObj'], device)


