"""
    Author        : Shelta Zhao(赵小棠)
    Email         : xiaotang_zhao@outlook.com
    Copyright (C) : NJU DisLab, 2025.
    Description   : Define DOA Processor, including Angle FFT.
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
from module.cfar_process import CFARProcessor
from utility.tool_box import peak_detect
from utility.visualizer_box import PCD_display


class DOAProcessor:
    def __init__(self, DOAObj, device):
        """
        Initialize the DOAProcessor with DOA configurations.

        Parameters:
            DOAObj (dict): Configuration for DOA Estimation.
            device (str): The device to perform the computation on ('cpu' or 'cuda').
        """

        self.DOAObj = DOAObj
        self.device = device

        # Get the DOA radar parameters
        self.d = DOAObj['antDis']
        self.D = torch.tensor(DOAObj['D'])
        self.gamma = DOAObj['gamma']
        self.sidelobeLevel_dB = DOAObj['sidelobeLevel_dB']
        self.doa_fft_size = DOAObj['DOAFFTSize']
        self.angles_DOA_azi = DOAObj['angles_DOA_azi']
        self.angles_DOA_ele = DOAObj['angles_DOA_ele']
        self.antenna_azimuthonly = DOAObj['antenna_azimuthonly']
        self.wx_vec = torch.linspace(-torch.pi, torch.pi, self.doa_fft_size + 1)[:-1]
        self.wz_vec = torch.linspace(-torch.pi, torch.pi, self.doa_fft_size + 1)[:-1]
        self.antenna_azimuthonly = DOAObj['antenna_azimuthonly']

    def run(self, detection_results, save=False, load=False):
        """
        Run the DOA Estimation on the given detection results.

        Parameters:
            detection_results (list): The detection results from CFAR Processor.
            save (bool): Whether to save the results to a file.
            load (bool): Whether to load the results from a file.

        Returns:
            point cloud data (list): The DOA estimation results.
            1. frameIdx: Frame index
            2. objectIdx: Object index
            3. x: X coordinate
            4. y: Y coordinate
            5. z: Z coordinate
            6. distance: Distance
            7. velocity: Velocity
            8. azimuth: Azimuth angle
            9. elevation: Elevation angle
            10. signalPower: Signal power
            11. noise_var: Noise variance
            12. estSNR: Estimated SNR
            13. rangeInd: Range index
            14. dopplerInd: Doppler index
        """

        # Initialize the list to store the output objects
        doa_estimate_result = []  
        for detection_point in detection_results:
            
            # Perform DOA estimation
            num_elements = detection_point[8:].numel() // 2
            current_bin_val = torch.complex(detection_point[8:8 + num_elements], detection_point[8 + num_elements:])
            DOAObj_est, doa_fft_result = self.DOA_beamformingFFT(current_bin_val)

            if len(DOAObj_est) > 0:
                for angle_obj in range(len(DOAObj_est)):  # Iterate through the columns of DOA_angles
                    doa_point = {}
                    
                    doa_point['frameIdx'] = detection_point[0]
                    doa_point['rangeInd'] = detection_point[1]
                    doa_point['range'] = detection_point[2]
                    doa_point['dopplerInd'] = detection_point[3]
                    doa_point['doppler'] = detection_point[4]
                    doa_point['noise_var'] = detection_point[5]
                    doa_point['signalPower'] = detection_point[6]
                    doa_point['estSNR'] = detection_point[7]
                    doa_point['bin_val'] = current_bin_val
                    
                    # Set the DOA angle and spectrum for the current object
                    doa_point['angles'] = DOAObj_est[angle_obj]
                    doa_point['spectrum'] = doa_fft_result 
                    
                    # Append the current object to the out list
                    doa_estimate_result.append(doa_point)

        # Perform coordinate transformation
        if not doa_estimate_result:
            return np.array([])
        
        # Initialize the output array
        point_cloud_data = np.zeros((len(doa_estimate_result), 14))

        for iobj, result in enumerate(doa_estimate_result):
            # Coordinate transformation (x, y, z)
            azimuth, elevation = np.deg2rad(result['angles'][0].item()), np.deg2rad(result['angles'][1].item())
            point_cloud_data[iobj, 0] = result['frameIdx']                                    # Frame index
            point_cloud_data[iobj, 1] = iobj + 1                                              # Object index (start from 1)
            point_cloud_data[iobj, 2] = result['range'] * np.sin(azimuth) * np.cos(elevation) # X
            point_cloud_data[iobj, 3] = result['range'] * np.cos(azimuth) * np.cos(elevation) # Y
            point_cloud_data[iobj, 4] = result['range'] * np.sin(elevation)                   # Z

            # Distance, velocity, azimuth, elevation
            point_cloud_data[iobj, 5] = result['range']
            point_cloud_data[iobj, 6] = result['doppler']
            point_cloud_data[iobj, 7] = result['angles'][0]                                   # Azimuth
            point_cloud_data[iobj, 8] = result['angles'][1]                                   # Elevation

            # Noise, SNR
            point_cloud_data[iobj, 9] = result['signalPower']
            point_cloud_data[iobj, 10] = result['noise_var']
            point_cloud_data[iobj, 11] = result['estSNR']

            # RDM index
            point_cloud_data[iobj, 12] = result['rangeInd']
            point_cloud_data[iobj, 13] = result['dopplerInd']

        return point_cloud_data
    
    def DOA_beamformingFFT(self, bin_val):
        
        # Get the DOA radar parameters
        apertureLen_azi = torch.max(self.D[:, 0]).item() + 1  
        apertureLen_ele = torch.max(self.D[:, 1]).item() + 1
        sig_2D = torch.zeros(apertureLen_azi, apertureLen_ele, dtype=torch.complex64)
        sig_2D[self.D[:, 0], self.D[:, 1]] = bin_val

        # Perform 2D Angle FFT
        angle_sepc_1D_fft = torch.fft.fftshift(torch.fft.fft(sig_2D, n=self.doa_fft_size, dim=0), dim = 0)
        angle_sepc_2D_fft = torch.fft.fftshift(torch.fft.fft(angle_sepc_1D_fft, n=self.doa_fft_size, dim=1), dim=1)

        # Generate the doa estimation results
        obj_cnt, DOAObj_est = 0, torch.tensor([], device=self.device)
        spec_azi = torch.abs(angle_sepc_1D_fft[:, self.antenna_azimuthonly])
        _, peakLoc_azi = peak_detect(torch.abs(spec_azi), self.gamma, self.sidelobeLevel_dB[0])

        if apertureLen_ele == 1:
            # Azimuth array only, no elevation antennas
            for ind in peakLoc_azi:
                azi_est = torch.asin(self.wx_vec[ind] / (2 * torch.pi * self.d)) * 180 / torch.pi
                if self.angles_DOA_azi[0] <= azi_est <= self.angles_DOA_azi[1]:
                    DOAObj_est = torch.cat((DOAObj_est, torch.tensor([azi_est, 0, ind, 0], device=self.device).unsqueeze(0)))
                    obj_cnt += 1
        else:
            # Azimuth and elevation angle estimation
            for ind in peakLoc_azi:
                spec_ele = torch.abs(angle_sepc_2D_fft[ind, :])
                _, peakLoc_ele = peak_detect(spec_ele, self.gamma, self.sidelobeLevel_dB[1])

                for peak in peakLoc_ele:
                    # Calculate angles
                    azi_est = torch.arcsin(self.wx_vec[ind] / (2 * torch.pi * self.d)) * 180 / torch.pi
                    ele_est = torch.arcsin(self.wz_vec[peak] / (2 * torch.pi * self.d)) * 180 / torch.pi

                    if (self.angles_DOA_azi[0] <= azi_est <= self.angles_DOA_azi[1] and self.angles_DOA_ele[0] <= ele_est <= self.angles_DOA_ele[1]):
                        DOAObj_est = torch.cat((DOAObj_est, torch.tensor([azi_est, ele_est, ind, peak], device=self.device).unsqueeze(0)))
                        obj_cnt += 1
            
        return DOAObj_est.cpu(), angle_sepc_2D_fft.cpu()

        
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
    regular_data, timestamp = get_regular_data(data_path, radar_params['readObj'], 'all', load=False, timestamp=True)

    # Perform Range & Doppler FFT
    fft_processor = FFTProcessor(radar_params['rangeFFTObj'], radar_params['dopplerFFTObj'], device)
    fft_output = fft_processor.run(regular_data)
    
    # Perform CFAR-CASO detection
    cfar_processor = CFARProcessor(radar_params['detectObj'], device)
    detection_results = cfar_processor.run(fft_output[0,:256,:,:,:], 0)

    # Test DOA Estimation
    doa_processor = DOAProcessor(radar_params['DOAObj'], device)
    doa_output = doa_processor.run(detection_results)
    PCD_display(doa_output)
    print(len(doa_output))