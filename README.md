# Overview

This repository processes raw mmWave radar data to generate point cloud data. The project aims to provide an efficient pipeline for handling radar data, including parsing, preprocessing, and visualization. This is particularly useful in applications like robotics, autonomous driving, and smart sensing.

## Features

- **Raw Data Parsing**: Converts raw mmWave radar data into a usable format.  
- **Point Cloud Generation**: Produces 3D point cloud data for visualization and analysis.  
- **Cross-Platform Support**: Works on Windows, macOS, and Linux.  

## Prerequisites

Ensure you have the following installed:

- Python 3.8 or above  
- `conda` (for managing environments)

## Installation

1. Clone the repository:

   ```bash
   git clone git@github.com:shelta-zhao/dislab-mmPcd.git
   cd dislab-mmPcd
   ```

2. Create the environment:
   
   ```bash
   conda env create -f requirements.yaml
   conda activate <your_environment_name>
   ```

## Usage

1. Run the script to process the data:  

   ```bash
   python dislab_mmPcd.py
   ```

2. Visualize the point cloud (optional):  

   ```bash
   python dislab_mmPcd.py --display
   ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## Contact

For questions or feedback, feel free to contact [Shelta Zhao](mailto:xiaotang_zhao@outlook.com).
