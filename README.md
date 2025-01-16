# Overview

This repository processes raw mmWave radar data to generate point cloud data. The project aims to provide an efficient pipeline for handling radar data, including parsing, preprocessing, and visualization. This is particularly useful in applications like robotics, autonomous driving, and smart sensing.

## Features

- **Raw Data Parsing**: Converts raw mmWave radar data into a usable format.  
- **Point Cloud Generation**: Produces 3D point cloud data for visualization and analysis.  
- **Cross-Platform Support**: Works on Windows, macOS, and Linux.  
- **Python Compatibility**: Supports Python 3.8, 3.9, and 3.10.  

## Prerequisites

Ensure you have the following installed:

- Python 3.8 or above  
- `conda` (for managing environments)

## Installation

1. Clone the repository:

   ```bash
   git clone git@github.com:shelta-zhao/DisLab_mmwavePCD.git
   cd DisLab_mmwavePCD
   ```

2. Create the environment:
   
   ```bash
   conda env create -f environments.yaml
   conda activate <your_environment_name>
   ```

## Usage

1. Prepare your raw radar data in the required format.  
2. Run the script to process the data:  

   ```bash
   python process_mmwave_data.py --input <path_to_raw_data> --output <path_to_point_cloud>
   ```

3. Visualize the point cloud (optional):  

   ```bash
   python visualize_point_cloud.py --input <path_to_point_cloud>
   ```

## Workflow

This project uses GitHub Actions to ensure code quality.

- Each push triggers a Pylint check for Python files.  
- Compatibility is tested on Ubuntu, macOS, and Windows using Python 3.8, 3.9, and 3.10.  


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## Contact

For questions or feedback, feel free to contact [Shelta Zhao](mailto:xiaotang_zhao@outlook.com).
