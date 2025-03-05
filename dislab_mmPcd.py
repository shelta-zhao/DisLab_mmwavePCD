"""
    Author        : Shelta Zhao(赵小棠)
    Email         : xiaotang_zhao@outlook.com
    Copyright (C) : NJU DisLab, 2025.
    Description   : This script converts radar data to Point Cloud Data (PCD) format.
"""

import os
import sys
import yaml
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utility.tool_box import parse_arguments
from pipeline.adc_to_pcd import adc_to_pcd


if __name__ == "__main__":
    
    # Parse options in the command line
    args = parse_arguments()

    # Perform the processing pipeline
    if args.pipeline == 1:

        # Traditional pipeline to generate PCD from raw radar data
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        point_cloud_data = adc_to_pcd(args.yaml_path, device, save=args.save, display=args.display)
    elif args.pipeline == 2:
        
        # ====================================================================================
        # Add your own code here to process the PCD data
        # ====================================================================================
        pass
    else:
        
        print("Invalid pipeline option. Please choose 1 for the traditional pipeline.")
        sys.exit(1) 