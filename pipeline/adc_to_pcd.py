"""
    Author        : Shelta Zhao(赵小棠)
    Email         : xiaotang_zhao@outlook.com
    Copyright (C) : NJU DisLab, 2025.
    Description   : Traditon pipeline to generate Point Cloud Data (PCD) from raw radar data.
"""

import os
import sys
import yaml
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from handler.param_process import get_radar_params
from handler.adc_load import get_regular_data
from module.fft_process import FFTProcessor
from module.cfar_process import CFARProcessor
from module.doa_process import DOAProcessor
from utility.visualizer_box import PCD_display


def adc_to_pcd(adc_list, device, save=False, display=False):
    """
    Generate Point Cloud Data (PCD) from raw radar data.

    Parameters:
        adc_list (str): The list of ADC data to be processed.
        device (str): The device to perform the computation on ('cpu' or 'cuda').
        save (bool): Whether to save the results to a file.
        display (bool): Whether to display the results.

    Returns:
        point_cloud_data (np.ndarray): The generated Point Cloud Data (PCD) from the raw radar data.
    """

    # Parse data config & Get radar params
    with open(f"{adc_list}.yaml", "r") as file:
        adc_list = yaml.safe_load(file)
    
    # Process each data in the list
    for adc_data in adc_list:
        
        # Print the current data info
        print(f"Processing data: {adc_data['prefix']} | {adc_data['config']} | {adc_data['radar']}")

        # Generate regular data & radar params
        data_path = os.path.join("data/adc_data", f"{adc_data['prefix']}/{adc_data['index']}")
        config_path = os.path.join("data/radar_config", adc_data["config"])
        
        radar_params = get_radar_params(config_path, adc_data['radar'])
        regular_data, _ = get_regular_data(data_path, radar_params['readObj'], 'all', timestamp=True)

        # Generate all module instances
        fft_processor = FFTProcessor(radar_params['rangeFFTObj'], radar_params['dopplerFFTObj'], device)
        cfar_processor = CFARProcessor(radar_params['detectObj'], device)
        doa_processor = DOAProcessor(radar_params['DOAObj'], device)

        # Perform Range & Doppler FFT
        fft_output = fft_processor.run(regular_data)
    
        # Perform CFAR-CASO detection
        for frameIdx in tqdm(range(fft_output.shape[0]), desc="Processing frames"):
            detection_results, _ = cfar_processor.run(fft_output[frameIdx,:radar_params['detectObj']['rangeFFTSize'] // 2,:,:,:], frameIdx)

            # Perform DOA Estimation
            doa_results = doa_processor.run(detection_results)

            # Merge the DOA results
            if frameIdx == 0:
                point_cloud_data = doa_results
            else:
                point_cloud_data = np.concatenate((point_cloud_data, doa_results), axis=0)
        
        # Save the PCD data if required
        if save:
            file_path = f"{data_path}/PCD_data.npy"
            if os.path.exists(file_path):
                os.remove(file_path)
            np.save(file_path, point_cloud_data)
            print(f"PCD data has been saved to {file_path}.")

        # Display the PCD data if required
        if display:
            PCD_display(point_cloud_data)

        # ====================================================================================
        # Add your own code here to process the PCD data
        # ====================================================================================

        # Add a line break
        print("------------------------------------------------------------------------------------------")
            

if __name__ == "__main__":

    # Test the ADC to PCD pipeline
    adc_to_pcd("adc_list", "cpu", save=True, display=False)