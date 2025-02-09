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
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from handler.param_process import get_radar_params
from handler.adc_load import get_regular_data
from module.fft_process import FFTProcessor
from utility.tool_box import reshape_fortran


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

    def run(self, input, frameIdx):
        """
        Perform CFAR-CASO on the 2D FFT result.

        Parameters:
            input(torch.Tensor): FFT result. Shape: (range_fft_size, doppler_fft_size, num_rx, num_tx).
            frameIdx(int): The frame index.

        Returns:
            torch.Tensor: detection result.
                - 0 frameIdx: The frame index.
                - 1 rangeInd: The range index.
                - 2 range: The range value.
                - 3 dopplerInd: The Doppler index.
                - 4 doppler: The Doppler value.
                - 5 noise_var: The noise variance.
                - 6 signalPower: The signal power.
                - 7 estSNR: The estimated SNR.   
                - 8~end: The real and imaginary parts of the signal bin.
            torch.Tensor: The non-coherent signal combination.
        """

        # Get non-coherent signal combination along the antenna array
        input = reshape_fortran(input, (input.shape[0], input.shape[1], input.shape[2] * input.shape[3]))
        sig_integrate = torch.sum(torch.abs(input)**2, dim=2) + 1
        
        # Switch on the dectect method
        if self.detectObj['detectMethod'] == 1:

            # Perform CFAR-CASO Range
            N_obj_Rag, Ind_obj_Rag, noise_obj_Rag, _ = self.CFAR_CASO_Range(input)

            if N_obj_Rag > 0:

                # Perform CFAR-CASO Doppler
                N_obj_valid, Ind_obj_valid, _ = self.CFAR_CASO_Doppler(input, Ind_obj_Rag)

                # Use aggregate noise estimation for each obj
                noise_obj_agg = torch.tensor([], device=self.device)
                for i_obj in range(N_obj_valid):

                    # Extract range and Doppler indices for the current object
                    indx1R = Ind_obj_valid[i_obj][0]
                    indx1D = Ind_obj_valid[i_obj][1]
 
                    # Find matching indices in Ind_obj_Rag
                    mask = (Ind_obj_Rag[:, 0] == indx1R) & (Ind_obj_Rag[:, 1] == indx1D)
                    if mask.any():
                        noiseInd = torch.nonzero(mask, as_tuple=False).squeeze(1)
                        noise_obj_agg = torch.cat((noise_obj_agg, noise_obj_Rag[noiseInd].unsqueeze(0)))
            
                # Generate detection results
                detection_results = torch.zeros((N_obj_valid, 8 + 2 * self.detectObj['numRxAnt'] * self.detectObj['TDM_MIMO_numTX']), device=self.device, dtype=torch.float32)
                for i_obj in range(N_obj_valid):

                    # Extract range and Doppler indices
                    range_ind = Ind_obj_valid[i_obj][0].item()
                    doppler_ind = Ind_obj_valid[i_obj][1].item()

                    # Compute range and Doppler values
                    range_val = range_ind * self.detectObj['rangeBinSize']
                    doppler_val = (doppler_ind - self.detectObj['dopplerFFTSize'] / 2) * self.detectObj['velocityBinSize']

                    # Extract bin_val and noise variance
                    bin_val = input[range_ind, doppler_ind, :].squeeze()
                    noise_var = noise_obj_agg[i_obj]

                    # Compute signal power and noise power
                    signal_power = torch.sum(torch.abs(bin_val) ** 2).item()
                    noise_power = torch.sum(noise_var).item()
                    est_snr = signal_power / noise_power if noise_power > 0 else float('inf')

                    # Perform phase correction for TDM MIMO
                    deltaPhi = 2 * torch.pi * (doppler_ind - self.detectObj['dopplerFFTSize'] / 2) / (self.detectObj['TDM_MIMO_numTX'] * self.detectObj['dopplerFFTSize'])
                    sig_bin = torch.zeros_like(bin_val, dtype=torch.complex64)
                    for i_TX in range(self.detectObj['TDM_MIMO_numTX']):
                        RX_ID = slice(i_TX * self.detectObj['numRxAnt'], (i_TX + 1) * self.detectObj['numRxAnt'])
                        deltaPhi_iTX = torch.tensor(i_TX * deltaPhi, device=self.device)
                        sig_bin[RX_ID] = bin_val[RX_ID] * (torch.cos(deltaPhi_iTX) - 1j * torch.sin(deltaPhi_iTX))

                    # Fill the result tensor
                    detection_results[i_obj, 0] = frameIdx          # frameIdx
                    detection_results[i_obj, 1] = range_ind         # rangeInd
                    detection_results[i_obj, 2] = range_val         # range
                    detection_results[i_obj, 3] = doppler_ind       # dopplerInd
                    detection_results[i_obj, 4] = doppler_val       # doppler
                    detection_results[i_obj, 5] = noise_power       # noise_var
                    detection_results[i_obj, 6] = signal_power      # signalPower
                    detection_results[i_obj, 7] = est_snr           # estSNR
                    
                    # Store real and imaginary parts of sig_bin in separate columns
                    detection_results[i_obj, 8:8 + sig_bin.numel()] = sig_bin.real.view(-1)
                    detection_results[i_obj, 8 + sig_bin.numel():] = sig_bin.imag.view(-1)

                # Return the detection results
                return detection_results, sig_integrate
        else:
            raise ValueError("Unknown Detect Method!")

    def CFAR_CASO_Range(self, input):
        """
        CFAR detection for range using the CASO method. 

        Parameters:
            input (torch.Tensor): The input tensor for range CFAR.

        Returns:
            N_obj: Number of detected objects.
            Ind_obj: Indices of detected objects.
            noise_obj: Noise variances for detected objects.
            snr_obj: Signal-to-noise ratio for each detected object.
        """

        # Retrieve parameters for range CFAR
        sig_integrate = torch.sum(torch.abs(input)**2, dim=2) + 1
        M_Samp, N_Pul = sig_integrate.shape[0], sig_integrate.shape[1]
        cellNum, gapNum, K0 = self.detectObj['refWinSize'][0], self.detectObj['guardWinSize'][0], self.detectObj['K0'][0]
        discardCellLeft, discardCellRight = self.detectObj['discardCellLeft'][0], self.detectObj['discardCellRight'][0]
        gaptot = gapNum + cellNum

        # Get return features for detected objects
        N_obj, Ind_obj, noise_obj, snr_obj = 0, torch.tensor([], dtype=torch.long, device=self.device), torch.tensor([], device=self.device), torch.tensor([], device=self.device)

        # Perform CFAR detection for each doppler bin
        for k in range(N_Pul):
            # Extract the relevant section of the signal
            vec = sig_integrate[:, k][discardCellLeft: M_Samp - discardCellRight]  
            vec = torch.cat((vec[:gaptot], vec, vec[-gaptot:]))

            # Detect objects by comparing each cell to the CFAR threshold
            for j in range(M_Samp - discardCellLeft - discardCellRight):

                # Get the indices of the reference cells
                cellInda = torch.arange(j - gaptot, j - gapNum) + gaptot
                cellIndb = torch.arange(j + gapNum + 1, j + gaptot + 1) + gaptot
                
                # Compute the mean of the reference cells for both parts
                cellave1a = torch.mean(vec[cellInda])
                cellave1b = torch.mean(vec[cellIndb])
                
                # Take the minimum of both averages
                cellave1 = torch.min(cellave1a, cellave1b)

                # Check if the signal exceeds the threshold
                if vec[j + gaptot] > K0 * cellave1:
                    if self.detectObj['maxEnable'] == 0 or vec[j + gaptot] > torch.max(vec[cellInda[0]:cellIndb[-1] + 1]):
                        N_obj += 1                                       # Increment object count
                        Ind_obj = torch.cat((Ind_obj, torch.tensor([j + discardCellLeft, k], device = self.device).unsqueeze(0)))
                        noise_obj = torch.cat((noise_obj, cellave1.unsqueeze(0)))
                        snr_obj = torch.cat((snr_obj, torch.tensor([vec[j + gaptot] / cellave1], device = self.device)))

        # Return the detected objects
        return N_obj, Ind_obj, noise_obj, snr_obj

    def CFAR_CASO_Doppler(self, input, Ind_obj_Rag):
        """
        CFAR detection for Doppler using the CASO method.

        Parameters:
            input (torch.Tensor): The input tensor for Doppler CFAR.
            Ind_obj_Rag (torch.Tensor): The indices of detected objects in range.
        
        Returns:
            N_obj_valid: Number of valid detected objects.
            Ind_obj_valid: Indices of valid detected objects.
            noise_obj_valid: Noise variances for valid detected objects.
        """

        # Retrieve parameters for doppler CFAR
        sig_integrate = torch.sum(torch.abs(input)**2, dim=2) + 1
        cellNum, gapNum, K0 = self.detectObj['refWinSize'][1], self.detectObj['guardWinSize'][1], self.detectObj['K0'][1]
        discardCellLeft, discardCellRight = self.detectObj['discardCellLeft'][1], self.detectObj['discardCellRight'][1]
        gaptot = gapNum + cellNum

        # Get detected range cells
        detected_Rag_Cell = torch.unique(Ind_obj_Rag[:, 0])
        sig_integrate = sig_integrate[detected_Rag_Cell, :]
        M_Samp, N_Pul = sig_integrate.shape[0], sig_integrate.shape[1]

        # Get return features for detected objects
        Ind_obj = torch.tensor([], dtype=torch.long, device=self.device)
        
        # Perform CFAR detection for each range bin
        for k in range(M_Samp):
            # Get the range index at current loop
            range_bin_index = int(detected_Rag_Cell[k])
            ind1 = torch.nonzero(Ind_obj_Rag[:, 0] == range_bin_index).squeeze()
            indR = Ind_obj_Rag[ind1, 1]

            # Extend the left vector by copying the leftmost and rightmost gaptot samples
            vec = sig_integrate[k, :][discardCellLeft: N_Pul - discardCellRight]  
            vec = torch.cat((vec[-gaptot:], vec, vec[:gaptot]))
            # vec = torch.cat((vec[:gaptot], vec, vec[-gaptot:])) # This is the correct one

            # Start to process
            ind_loc_all = torch.tensor([], dtype=torch.long, device=self.device)
            ind_loc_Dop = torch.tensor([], dtype=torch.long, device=self.device)

            for j in range(N_Pul - discardCellLeft - discardCellRight):
                # Get the indices of the reference cells
                cellInda = torch.arange(j - gaptot, j - gapNum) + gaptot
                cellIndb = torch.arange(j + gapNum + 1, j + gaptot + 1) + gaptot
                
                # Compute the mean of the reference cells for both parts
                cellave1a = torch.mean(vec[cellInda])
                cellave1b = torch.mean(vec[cellIndb])

                # Take the minimum of both averages
                cellave1 = torch.min(cellave1a, cellave1b)
                
                # Check if the signal exceeds the threshold
                if vec[j + gaptot] > K0 * cellave1:
                    if self.detectObj['maxEnable'] == 0 or vec[j + gaptot] > torch.max(vec[cellInda[0]:cellIndb[-1] + 1]):
                        if torch.isin(indR, j).any():
                            ind_loc_all = torch.cat((ind_loc_all, torch.tensor([range_bin_index], device=self.device)))
                            ind_loc_Dop = torch.cat((ind_loc_Dop, torch.tensor([j + discardCellLeft], device=self.device)))

            # Check if ind_loc_all is not empty
            if ind_loc_all.numel() > 0:
                # Create ind_obj_0 with 2 columns
                ind_obj_0 = torch.stack((ind_loc_all, ind_loc_Dop), dim=1)
                
                # If Ind_obj is empty, initialize it with ind_obj_0
                if Ind_obj.numel() == 0:
                    Ind_obj = ind_obj_0
                else:
                    # Avoid duplicated detection points by checking ind_obj_0_sum
                    ind_obj_0_sum = {loc + 10000 * dop for loc, dop in zip(ind_loc_all, ind_loc_Dop)}
                    Ind_obj_sum = {loc + 10000 * dop for loc, dop in Ind_obj}

                    # Add only non-duplicate elements
                    Ind_obj = torch.cat((Ind_obj, torch.stack([ind for ind, ind_sum in zip(ind_obj_0, ind_obj_0_sum) if ind_sum not in Ind_obj_sum]).to(self.device)))

        # Initialize variables for valid objects
        cellNum, gapNum = self.detectObj['refWinSize'][0], self.detectObj['guardWinSize'][0]
        N_obj_valid, Ind_obj_valid, noise_obj_valid = 0, torch.tensor([], dtype=torch.long, device=self.device), torch.tensor([], device=self.device)
        gaptot = gapNum + cellNum

        # Process each detected object
        for i_obj in range(len(Ind_obj)):

            ind_range = Ind_obj[i_obj][0]      # Range index of the object
            ind_doppler = Ind_obj[i_obj][1]    # Doppler index of the object

            # Skip the object if the signal power is below the threshold
            if torch.min(torch.abs(input[ind_range, ind_doppler, :])**2) < self.detectObj['powerThre']:
                continue
            
            # Handle edge cases (objects near boundaries)
            if ind_range <= gaptot:
                # On the left boundary, use the right side samples twice
                cellInd = torch.cat([torch.arange(ind_range + gapNum + 1, ind_range + gaptot + 1)] * 2)
            elif ind_range >= input.shape[0] - gaptot:
                # On the right boundary, use the left side samples twice
                cellInd = torch.cat([torch.arange(ind_range - gaptot, ind_range - gapNum)] * 2)
            else:
                # In the middle, use both left and right side samples
                cellInd = torch.cat([torch.arange(ind_range - gaptot, ind_range - gapNum), torch.arange(ind_range + gapNum + 1, ind_range + gaptot + 1)])
            
            # Add valid object features
            N_obj_valid += 1
            Ind_obj_valid = torch.cat((Ind_obj_valid, Ind_obj[i_obj].clone().detach().unsqueeze(0).to(self.device)))
            noise_obj_valid = torch.cat((noise_obj_valid, torch.mean(torch.abs(input[ind_range, ind_doppler, :])**2, dim=0).unsqueeze(0)))

        # Return the detected objects
        return N_obj_valid, Ind_obj_valid, noise_obj_valid


if __name__ == "__main__":
    
    # Parse data config & Get radar params
    with open("adc_list.yaml", "r") as file:
        data = yaml.safe_load(file)[0]
    data_path = os.path.join("data/adc_data", f"{data['prefix']}/{data['index']}")
    config_path = os.path.join("data/radar_config", data["config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get radar params
    radar_params = get_radar_params(config_path, data['radar'], load=False)

    # Get regular raw radar data
    regular_data, timestamp = get_regular_data(data_path, radar_params['readObj'], '1', timestamp=True)

    # Get FFT Output
    fft_processor = FFTProcessor(radar_params['rangeFFTObj'], radar_params['dopplerFFTObj'], device)
    fft_output = fft_processor.run(regular_data)
    
    # Test CFAR-CASO
    cfar_processor = CFARProcessor(radar_params['detectObj'], device)
    cfar_output = cfar_processor.run(fft_output[0,:256,:,:,:], 0)
    print(cfar_output.shape)