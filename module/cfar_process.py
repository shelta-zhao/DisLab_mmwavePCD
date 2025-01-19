"""
    Author        : Shelta Zhao(赵小棠)
    Email         : xiaotang_zhao@outlook.com
    Copyright (C) : NJU DisLab, 2025.
    Description   : Define CFAR Processor, including CFAR-CASO.
"""

import os
import sys
import yaml
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from handler.param_process import get_radar_params
from handler.adc_load import get_regular_data
from module.fft_process import FFTProcessor

class CFARProcessor:
    def __init__(self, detectObj, device='cpu'):
        """
        Initialize the CFAR Processor with detect configurations.

        Parameters:
            detectObj (dict): Configuration for CFAR-CASO.
            device (str): The device to perform the computation on ('cpu' or 'cuda').
        """

        self.detectObj = detectObj
        self.device = device

    def run(self, input):
        """
        Perform CFAR-CASO on the 2D FFT result.

        Parameters:
            input(torch.Tensor): FFT result. Shape: (range_fft_size, doppler_fft_size, num_rx, num_tx)

        Returns:
            torch.Tensor: CFAR result.        
        """

        # Get 
        sig = 1

        # 
        self.CFAR_CASO_Range(sig)

    def CFAR_CASO_Range(self, sig):
        """
        CFAR detection for range using the CASO method. 

        Parameters:
            sig: The integrated signal tensor (sum of squared magnitudes).

        Returns:
            N_obj: Number of detected objects.
            Ind_obj: Indices of detected objects.
            noise_obj: Noise variances for detected objects.
            CFAR_SNR: Signal-to-noise ratio for each detected object.
        """

        # Retrieve parameters for range CFAR
        range_fft_size, doppler_fft_size = sig.shape[0], sig.shape[1]
        cellNum, gapNum, K0 = self.detectObj['refWinSize'][0], self.detectObj['guardWinSize'][0], self.detectObj['K0'][0]
        discardCellLeft, discardCellRight = self.detectObj['discardCellLeft'], self.detectObj['discardCellRight']
        gaptot = gapNum + cellNum

        # Get return features for detected objects
        N_obj, Ind_obj, noise_obj, snr_obj = 0, [], [], []
        
        # Perform CFAR detection for each doppler bin
        for k in range(doppler_fft_size):
            # Extract the relevant section of the signal
            vec = sig[:, k].T[discardCellLeft: range_fft_size - discardCellRight]  
            vec = torch.cat((vec[:gaptot], vec, vec[-gaptot:]))

            # Detect objects by comparing each cell to the CFAR threshold
            for j in range(1, range_fft_size - discardCellLeft - discardCellRight + 1):
                # Get the indices of the reference cells
                cellInda = torch.arange(j - gaptot, j - gapNum)
                cellIndb = torch.arange(j + gapNum + 1, j + gaptot + 1)
                
                # Shift indices by gaptot
                cellInda = cellInda + gaptot
                cellIndb = cellIndb + gaptot
                
                # Compute the mean of the reference cells for both parts
                cellave1a = torch.mean(vec[cellInda.int() - 1])
                cellave1b = torch.mean(vec[cellIndb.int() - 1])
                
                # Take the minimum of both averages
                cellave1 = torch.min(cellave1a, cellave1b)

                # Check if the signal exceeds the threshold
                if vec[j + gaptot - 1] > K0 * cellave1:
                    N_obj += 1                                       # Increment object count
                    Ind_obj.append([j + discardCellLeft - 1, k])     # Store object indices
                    noise_obj.append(cellave1)                       # Store noise variance
                    snr_obj.append(vec[j + gaptot - 1] / cellave1)  # Store SNR

        # Convert object indices to tensor for consistency
        Ind_obj = torch.tensor(Ind_obj, dtype=torch.long)

        # Return the detected objects
        return N_obj, Ind_obj, noise_obj, snr_obj

    def CFAR_CASO_Doppler(self, sig, N_obj, Ind_obj, noise_obj, snr_obj):

        # Retrieve parameters for range CFAR
        range_fft_size, doppler_fft_size = sig.shape[0], sig.shape[1]
        cellNum, gapNum, K0 = self.detectObj['refWinSize'][1], self.detectObj['guardWinSize'][1], self.detectObj['K0'][1]
        discardCellLeft, discardCellRight = self.detectObj['discardCellLeft'], self.detectObj['discardCellRight']
        gaptot = gapNum + cellNum


if __name__ == "__main__":
    
    # Parse data config & Get radar params
    with open("adc_list.yaml", "r") as file:
        data = yaml.safe_load(file)
    data_path = os.path.join("data/adc_data", f"{data['prefix']}/{data['index']}")
    config_path = os.path.join("data/radar_config", data["config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get radar params
    radar_params = get_radar_params(config_path, data['radar'], load=True)

    # Get regular raw radar data
    regular_data, timestamp = get_regular_data(data_path, radar_params['readObj'], '1', timestamp=True)

    # Get FFT Output
    fft_processor = FFTProcessor(radar_params['rangeFFTObj'], radar_params['dopplerFFTObj'], device)
    fft_output = fft_processor.run(regular_data)
    
    # Test CFAR-CASO
    cfar_processor = CFARProcessor(radar_params['detectObj'], device)
    cfar_output = cfar_processor.run(fft_output)
    




    

