"""
    Author        : Shelta Zhao(赵小棠)
    Email         : xiaotang_zhao@outlook.com
    Copyright (C) : NJU DisLab, 2025.
    Description   : Parse mmWave Studio config JSON files.
"""

import os
import yaml
import json
import numpy as np
from datetime import datetime, timezone, timedelta


def get_radar_params(config_path, radar_type, save=False, load=False):
    """
    Head Function: Get radar params.

    Parameters:
        config_path (str): Path to the folder containing JSON files from mmWave Studio.
        radar_type (str): The type of radar device used in the experiment.
        save (bool): Option to save the radar params in yaml file.
        load (bool): Option to load the radar params from config path.

    Returns:
        dict: A dictionary containing the configuration parameters for the mmWave devices.
        - readObj: Parameters for reading the data from the binary files.
        - rangeFFTObj: Parameters for range processing.
        - dopplerFFTObj: Parameters for Doppler processing.
        - detectObj: Parameters for CFAR-CASO.
        - DOAObj: Parameters for Direction of Arrival (DOA) estimation.
    """

    # Load params if required
    if load:
        try:
            with open(f"{config_path}/radar_params.yaml", "r") as file:
                radar_params = yaml.safe_load(file)
            print(f"Radar params have been loaded from {config_path}/radar_params.yaml.")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {config_path}/radar_params.yaml")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {config_path}/radar_params.yaml. Details: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while loading {config_path}/radar_params.yaml. Details: {e}")
    else:
        # Generate radar params
        radar_params = generate_params(config_path, radar_type)

        # Save params if required
        if save:
            save_params(radar_params, config_path)
            print(f"Radar params has been saved to {config_path}/radar_params.yaml.")
    
    # Return radar params
    return radar_params


def generate_params(config_path, radar_type):
    """
    Generates the configuration parameters for the mmWave devices based on the JSON files in the config folder.
    
    Parameters:
        config_path (str): Path to the folder containing JSON files from mmWave Studio.
        radar_type (str): The type of radar device used in the experiment.
    
    Returns:
        dict: A dictionary containing the configuration parameters for the mmWave devices.
    """
    
    # Validate the JSON files
    json_params, json_valid = validate_json(config_path)

    # Return {} if the JSON files are not valid
    if not json_valid:
        return {}
    
    # Basic configuration parameters
    params = {}
    params['dataPath'] = config_path
    params['radarType'] = radar_type
    params['generateDate'] = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
    
    # Radar configuration parameters    
    TxChannelEnabled = [
        np.argmax([
            json_params['DevConfig'][i + 1]['Chirp'][iconfig + 1]['Tx0Enable'],
            json_params['DevConfig'][i + 1]['Chirp'][iconfig + 1]['Tx1Enable'],
            json_params['DevConfig'][i + 1]['Chirp'][iconfig + 1]['Tx2Enable']
        ]) for i in range(json_params['NumDevices'])
        for iconfig in range(json_params['DevConfig'][i + 1]['NumChirps'])
    ]
    params['iqSwap'] = json_params['DevConfig'][1]['DataFormat']['IQSwap']
    params['chInterleave'] = json_params['DevConfig'][1]['DataFormat']['chInterleave']
    params['numLane'] = json_params['DevConfig'][1]['DataFormat']['numLane']
    params['numADCSample'] = json_params['DevConfig'][1]['Profile'][0]['NumSamples']
    params['AdcOneSampleSize'] = json_params['DevConfig'][1]['DataFormat']['gAdcOneSampleSize']
    params['adcSampleRate'] = json_params['DevConfig'][1]['Profile'][0]['SamplingRate'] * 1e3
    params['startFreqConst'] = json_params['DevConfig'][1]['Profile'][0]['StartFreq'] * 1e9
    params['chirpSlope'] = json_params['DevConfig'][1]['Profile'][0]['FreqSlope'] * 1e12
    params['chirpIdleTime'] = json_params['DevConfig'][1]['Profile'][0]['IdleTime'] * 1e-6
    params['chirpRampEndTime'] = json_params['DevConfig'][1]['Profile'][0]['RampEndTime'] * 1e-6
    params['adcStartTimeConst'] = json_params['DevConfig'][1]['Profile'][0]['AdcStartTime'] * 1e-6
    params['frameCount'] = json_params['DevConfig'][1]['FrameConfig']['NumFrames']
    params['numChirpsInLoop'] = json_params['DevConfig'][1]['NumChirps']
    params['nchirp_loops'] = json_params['DevConfig'][1]['FrameConfig']['NumChirpLoops']
    params['framePeriodicty'] = json_params['DevConfig'][1]['FrameConfig']['Periodicity'] * 1e-3
    params['NumDevices'] = json_params['NumDevices']
    params['numTxAnt'] = len(TxChannelEnabled)
    params['TxToEnable'] = TxChannelEnabled
    params['numRxToEnable'] = len(json_params['RxToEnable'])

    # Radar TX/RX parameters
    if radar_type == "IWR6843ISK-ODS":
        TX_position_azi = [0, 2, 2]
        TX_position_ele = [2, 2, 0]
        RX_position_azi = [0, 0, 1, 1]
        RX_position_ele = [1, 0, 0, 1]
    else:
        TX_position_azi = [0, 2, 4]
        TX_position_ele = [0, 1, 0]
        RX_position_azi = [0, 1, 2, 3]
        RX_position_ele = [0, 0, 0, 0]
    D = np.array([[rx_azi + tx_azi, rx_ele + tx_ele] for tx_azi, tx_ele in zip(TX_position_azi, TX_position_ele) for rx_azi, rx_ele in zip(RX_position_azi, RX_position_ele)])

    # Derived parameters
    speed_of_light = 3e8
    scale_factor = [x * 4 for x in [0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125, 0.0009765625, 0.00048828125]]

    # Chirp
    chirp_ramp_time = params['numADCSample'] / params['adcSampleRate']
    chirp_bandwidth = params['chirpSlope'] * chirp_ramp_time
    chirp_interval = params['chirpRampEndTime'] + params['chirpIdleTime']
    carrier_frequency = params['startFreqConst'] + (params['adcStartTimeConst'] + chirp_ramp_time / 2) * params['chirpSlope']
    lambda_ = speed_of_light / carrier_frequency
    data_size_one_chirp = params['AdcOneSampleSize'] * params['numADCSample'] * params['numRxToEnable']
    data_size_one_frame = data_size_one_chirp * params['numChirpsInLoop'] * params['nchirp_loops']

    # MIMO
    num_sample_per_chirp = round(chirp_ramp_time * params['adcSampleRate'])
    num_chirps_per_frame = params['nchirp_loops'] * params['numChirpsInLoop']
    num_chirps_per_vir_ant = params['nchirp_loops']
    num_virtual_rx_ant = len(params['TxToEnable']) * params['numRxToEnable']

    # FFT
    range_fft_size = 2 ** (int(np.ceil(np.log2(num_sample_per_chirp))))
    range_resolution = speed_of_light / 2 / chirp_bandwidth
    range_bin_size = range_resolution * num_sample_per_chirp / range_fft_size
    max_range = speed_of_light * params['adcSampleRate'] * chirp_ramp_time / (2 * 2 * chirp_bandwidth)
    doppler_fft_size = 2 ** (int(np.ceil(np.log2(params['nchirp_loops']))))
    velocity_resolution = lambda_ / (2 * params['nchirp_loops'] * chirp_interval * params['numTxAnt'])
    velocity_bin_size = velocity_resolution * num_chirps_per_vir_ant / doppler_fft_size
    maximum_velocity = lambda_ / (chirp_interval * 4)

    # Read data parameters
    read_data_params = {
        'iqSwap': params['iqSwap'],
        'numLane': params['numLane'],
        'chInterleave': params['chInterleave'],
        'dataSizeOneFrame': data_size_one_frame,
        'numAdcSamplePerChirp': num_sample_per_chirp,
        'numChirpsPerFrame': num_chirps_per_frame,
        'numTxForMIMO': len(params['TxToEnable']),
        'numRxForMIMO': params['numRxToEnable']
    }

    # Range FFT parameters
    range_proc_params = {
        'radarPlatform': radar_type,
        'rangeFFTSize': range_fft_size,
        'dcOffsetCompEnable': 1,
        'rangeWindowEnable': 1,
        'FFTOutScaleOn': 0,
        'scaleFactorRange': scale_factor[int(np.log2(range_fft_size)) - 4],
        'discardEnable': 1,
        'discardCellLeft': 0.05,
        'discardCellRight': 0.1,
        'rangeResolution': range_resolution,
        'maxRange': max_range
    }

    # Doppler FFT parameters
    doppler_proc_params = {
        'dopplerFFTSize': doppler_fft_size,
        'dopplerWindowEnable': 0,
        'FFTOutScaleOn': 0,
        'scaleFactorDoppler': scale_factor[int(np.log2(doppler_fft_size)) - 4],
        'velocityResolution': velocity_resolution,
        'maximumVelocity': maximum_velocity
    }

    # CFAR-CASO parameters
    cfar_caso_params = {
        'detectMethod': 1,
        'numAntenna': num_virtual_rx_ant,
        'refWinSize': [5, 3],
        'guardWinSize': [3, 2],
        'K0': [5, 4],
        'discardCellLeft': [10, 0],
        'discardCellRight': [20, 0],
        'maxEnable': 0,
        'rangeBinSize': range_bin_size,
        'velocityBinSize': velocity_bin_size,
        'rangeFFTSize': range_fft_size,
        'dopplerFFTSize': doppler_fft_size,
        'powerThre': 0,
        'numRxAnt': params['numRxToEnable'],
        'TDM_MIMO_numTX': params['numTxAnt'],
        'minDisApplyVmaxExtend': 10,
        'applyVmaxExtend': 0
    }

    # DOA parameters
    doa_params = {
        'D': D.tolist(),
        'DOAFFTSize': 180,
        'antenna_DesignFreq': params['startFreqConst'],
        'antPos': np.arange(num_virtual_rx_ant).tolist(),
        'antenna_azimuthonly': int(max(np.unique(D[:, 1]), key=lambda y: (np.sum(D[:, 1] == y), y))),
        'antDis': 0.5 * carrier_frequency / params['startFreqConst'],
        'method': 1,
        'angles_DOA_azi': [-80, 80],
        'angles_DOA_ele': [-80, 80] if radar_type == "IWR6843ISK-ODS" else [-20, 20],
        'gamma': 10 ** (0.2 / 10),
        'sidelobeLevel_dB': [1, 0]
    }
    
    # Combine all params
    radar_params = {
        'readObj': read_data_params,
        'rangeFFTObj': range_proc_params,
        'dopplerFFTObj': doppler_proc_params,
        'detectObj': cfar_caso_params,
        'DOAObj': doa_params
    }

    # Return the parameters
    return radar_params


def validate_json(config_path):
    """
    Validates and parses JSON files in the config folder.
    
    Parameters:
        config_path (str): Path to the folder containing JSON files from mmWave Studio.
    
    Returns:
        tuple: A tuple containing the parsed parameters (Params) and the validation status (jsonValid).
    """

    # Get file path for the JSON files
    filenames = [file for file in os.listdir(config_path) if file.endswith('.json')]

    # Load the JSON files
    if len(filenames) != 2:
        raise ValueError("Too many or missing JSON files in the data folder")

    # Parse the JSON files
    setup_json = None
    mmwave_json = None
    for filename in filenames:
        file_path = os.path.join(config_path, filename)
        if 'setup.json' in filename:
            with open(file_path, 'r') as file:
                setup_json = json.load(file)
        elif 'mmwave.json' in filename:
            with open(file_path, 'r') as file:
                mmwave_json = json.load(file)

    if setup_json is None or mmwave_json is None:
        raise ValueError("Required JSON files (setup.json and mmwave.json) are missing or misnamed")

    # Check the validity of the JSON files
    json_valid = checkout_json(setup_json, mmwave_json)

    # Parse the JSON files
    json_params = parse_json(mmwave_json)
    json_params['mmWaveDevice'] = setup_json.get('mmWaveDevice')

    return json_params, json_valid


def checkout_json(setup_json, mmwave_json):
    """
    Validates the setupJSON and mmwaveJSON configurations.

    Parameters:
        setup_json (dict): The parsed setup JSON object.
        mmwave_json (dict): The parsed mmWave JSON object.

    Returns:
        bool: True if the JSON configurations are valid, False otherwise.
    """

    # Supported platform list
    supported_platforms = ["awr1843", "iwr1843", "iwr6843"]
    json_valid = True

    # Validate if the device is supported
    mmwave_device = setup_json.get("mmWaveDevice")
    if mmwave_device not in supported_platforms:
        print(f"Platform not supported: {mmwave_device}")
        json_valid = False

    # Validate the capture hardware
    if setup_json.get("captureHardware") != "DCA1000":
        print(f"Capture hardware is not supported: {setup_json.get('captureHardware')}")
        json_valid = False

    # Validate ADC_ONLY capture
    transfer_fmt_pkt0 = mmwave_json.get("mmWaveDevices", {})[0].get("rawDataCaptureConfig", {}).get("rlDevDataPathCfg_t", {}).get("transferFmtPkt0")
    if transfer_fmt_pkt0 != "0x1":
        print(f"Capture data format is not supported: {transfer_fmt_pkt0}")
        json_valid = False

    # Validate the data logging mode
    data_logging_mode = setup_json.get("DCA1000Config", {}).get("dataLoggingMode")
    if data_logging_mode != "raw":
        print(f"Capture data logging mode is not supported: {data_logging_mode}")
        json_valid = False

    # Validate the capture configuration
    lane_enable = mmwave_json.get("mmWaveDevices", {})[0].get("rawDataCaptureConfig", {}).get("rlDevLaneEnable_t", {}).get("laneEn")
    num_lane = bin(int(lane_enable, 16)).count("1") if lane_enable else 0
    ch_interleave = mmwave_json.get("mmWaveDevices", {})[0].get("rawDataCaptureConfig", {}).get("rlDevDataFmtCfg_t", {}).get("chInterleave")

    if mmwave_device in {"awr1443", "iwr1443", "awr1243", "iwr1243"}:
        if num_lane != 4:
            print(f"{num_lane} LVDS Lane is not supported for device: {mmwave_device}")
            json_valid = False
        if ch_interleave != 0:
            print(f"Interleave mode {ch_interleave} is not supported for device: {mmwave_device}")
            json_valid = False
    else:
        if num_lane != 2:
            print(f"{num_lane} LVDS Lane is not supported for device: {mmwave_device}")
            json_valid = False
        if ch_interleave != 1:
            print(f"Interleave mode {ch_interleave} is not supported for device: {mmwave_device}")
            json_valid = False

    return json_valid


def parse_json(mmwave_json):
    """
    Configures the mmWave devices based on the given JSON data, including enabling channels, setting up profiles, and chirp configurations.
    
    Parameters:
        mmwave_json (dict): The parsed mmWave JSON object.
    
    Returns:
        dict: A dictionary containing the configured parameters for each mmWave device.
    """

    Params = {}
    MmWaveDevicesConfig = mmwave_json["mmWaveDevices"]

    # Number of devices
    Params["NumDevices"] = len(MmWaveDevicesConfig)

    # TX/RX Channels enabled
    TxToEnable = []
    RxToEnable = []
    DevIdMap = []
    Params["DevConfig"] = {}

    for count, deviceConfig in enumerate(MmWaveDevicesConfig):
        DevIdMap.append(deviceConfig["mmWaveDeviceId"] + 1)
        DevId = DevIdMap[count]

        # Initialize device configuration
        Params["DevConfig"][DevId] = {}

        # TX Channel Enable
        TxChannelEn = int(deviceConfig["rfConfig"]["rlChanCfg_t"]["txChannelEn"][2:], 16)
        for txChannel in range(1, 4):
            if TxChannelEn & (1 << (txChannel - 1)):
                TxToEnable.append((3 * (DevId - 1)) + txChannel)

        # RX Channel Enable
        RxChannelEn = int(deviceConfig["rfConfig"]["rlChanCfg_t"]["rxChannelEn"][2:], 16)
        for rxChannel in range(1, 5):
            if RxChannelEn & (1 << (rxChannel - 1)):
                RxToEnable.append((4 * (DevId - 1)) + rxChannel)

        # Profile Configuration
        Params.setdefault('DevConfig', {})[DevId] = {}
        Params['DevConfig'][DevId]['NumProfiles'] = len(MmWaveDevicesConfig[count]['rfConfig']['rlProfiles'])
        NumChirpBlocks = len(MmWaveDevicesConfig[count]['rfConfig']['rlChirps'])
        Profiles = MmWaveDevicesConfig[count]['rfConfig']['rlProfiles']
        Chirps = MmWaveDevicesConfig[count]['rfConfig']['rlChirps']

        for ProfileCount in range(Params['DevConfig'][DevId]['NumProfiles']):
            profile = Profiles[ProfileCount]['rlProfileCfg_t']
            Params['DevConfig'][DevId].setdefault('Profile', {})[ProfileCount] = {
                'ProfileId': profile['profileId'],
                'StartFreq': profile['startFreqConst_GHz'],
                'FreqSlope': profile['freqSlopeConst_MHz_usec'],
                'IdleTime': profile['idleTimeConst_usec'],
                'AdcStartTime': profile['adcStartTimeConst_usec'],
                'RampEndTime': profile['rampEndTime_usec'],
                'TxStartTime': profile['txStartTime_usec'],
                'RxGain': int(profile['rxGain_dB'][2:], 16),
                'NumSamples': profile['numAdcSamples'],
                'SamplingRate': profile['digOutSampleRate'],
                'HpfCornerFreq1': profile['hpfCornerFreq1'],
                'HpfCornerFreq2': profile['hpfCornerFreq2'],
                'Tx0PhaseShift': (int(profile['txPhaseShifter'][2:], 16) & 255) / 4 * 5.625,
                'Tx1PhaseShift': ((int(profile['txPhaseShifter'][2:], 16) >> 8) & 255) / 4 * 5.625,
                'Tx2PhaseShift': ((int(profile['txPhaseShifter'][2:], 16) >> 16) & 255) / 4 * 5.625,
                'Tx0OutPowerBackOff': int(profile['txOutPowerBackoffCode'][2:], 16) & 255,
                'Tx1OutPowerBackOff': (int(profile['txOutPowerBackoffCode'][2:], 16) >> 8) & 255,
                'Tx2OutPowerBackOff': (int(profile['txOutPowerBackoffCode'][2:], 16) >> 16) & 255
            }

        # Chirp Configuration
        Params['DevConfig'][DevId]['NumChirps'] = 0
        for ChirpCount in range(NumChirpBlocks):
            Params['DevConfig'][DevId]['NumChirps'] += (Chirps[ChirpCount]['rlChirpCfg_t']['chirpEndIdx'] - Chirps[ChirpCount]['rlChirpCfg_t']['chirpStartIdx']) + 1
            for ChirpId in range(Chirps[ChirpCount]['rlChirpCfg_t']['chirpStartIdx'] + 1, Chirps[ChirpCount]['rlChirpCfg_t']['chirpEndIdx'] + 2):
                chirp = Chirps[ChirpCount]['rlChirpCfg_t']
                Params['DevConfig'][DevId].setdefault('Chirp', {})[ChirpId] = {
                    'ChirpIdx': ChirpId,
                    'ProfileId': chirp['profileId'],
                    'StartFreqVar': chirp['startFreqVar_MHz'],
                    'FreqSlopeVar': chirp['freqSlopeVar_KHz_usec'],
                    'IdleTimeVar': chirp['idleTimeVar_usec'],
                    'AdcStartTime': chirp['adcStartTimeVar_usec'],
                    'Tx0Enable': bool(int(chirp['txEnable'][2:], 16) & 1),
                    'Tx1Enable': bool(int(chirp['txEnable'][2:], 16) & 2),
                    'Tx2Enable': bool(int(chirp['txEnable'][2:], 16) & 4)
                }

        # Configure the burst profile mappings (BPM) for the device
        NumBpmBlocks = len(MmWaveDevicesConfig[count]['rfConfig']['rlBpmChirps'])
        BpmConfig = MmWaveDevicesConfig[count]['rfConfig']['rlBpmChirps']
        for BpmCount in range(NumBpmBlocks):
            for ChirpId in range(BpmConfig[BpmCount]['rlBpmChirpCfg_t']['chirpStartIdx'] + 1, BpmConfig[BpmCount]['rlBpmChirpCfg_t']['chirpEndIdx'] + 2):
                bpm = BpmConfig[BpmCount]['rlBpmChirpCfg_t']
                Params['DevConfig'][DevId]['Chirp'][ChirpId].update({
                    'Tx0OffBpmVal': bool(int(bpm['constBpmVal'][2:], 16) & 1),
                    'Tx0OnBpmVal': bool(int(bpm['constBpmVal'][2:], 16) & 2),
                    'Tx1OffBpmVal': bool(int(bpm['constBpmVal'][2:], 16) & 4),
                    'Tx1OnBpmVal': bool(int(bpm['constBpmVal'][2:], 16) & 8),
                    'Tx2OffBpmVal': bool(int(bpm['constBpmVal'][2:], 16) & 16),
                    'Tx2OnBpmVal': bool(int(bpm['constBpmVal'][2:], 16) & 32)
                })

        # Phase shift configuration
        NumPhaseShiftBlocks = len(MmWaveDevicesConfig[count]['rfConfig']['rlRfPhaseShiftCfgs'])
        PhaseShiftConfig = MmWaveDevicesConfig[count]['rfConfig']['rlRfPhaseShiftCfgs']
        for PhaseShiftCount in range(NumPhaseShiftBlocks):
            for ChirpId in range(PhaseShiftConfig[PhaseShiftCount]['rlRfPhaseShiftCfg_t']['chirpStartIdx'] + 1, PhaseShiftConfig[PhaseShiftCount]['rlRfPhaseShiftCfg_t']['chirpEndIdx'] + 2):
                phase_shift = PhaseShiftConfig[PhaseShiftCount]['rlRfPhaseShiftCfg_t']
                Params['DevConfig'][DevId]['Chirp'][ChirpId].update({
                    'Tx0PhaseShift': phase_shift['tx0PhaseShift'] * 5.625,
                    'Tx1PhaseShift': phase_shift['tx1PhaseShift'] * 5.625,
                    'Tx2PhaseShift': phase_shift['tx2PhaseShift'] * 5.625
                })
        
        # Data format configuration
        AdcFormatConfig = MmWaveDevicesConfig[count]['rfConfig']['rlAdcOutCfg_t']['fmt']
        Params['DevConfig'][DevId]['DataFormat'] = {
            'NumAdcBits': AdcFormatConfig['b2AdcBits'],
            'Format': AdcFormatConfig['b2AdcOutFmt'],
            'ReductionFactor': AdcFormatConfig['b8FullScaleReducFctr'],
            'IQSwap': MmWaveDevicesConfig[count]['rawDataCaptureConfig']['rlDevDataFmtCfg_t']['iqSwapSel'],
            'chInterleave': MmWaveDevicesConfig[count]['rawDataCaptureConfig']['rlDevDataFmtCfg_t']['chInterleave'],
            'numLane': bin(int(MmWaveDevicesConfig[count]['rawDataCaptureConfig']['rlDevLaneEnable_t']['laneEn'][2:], 16)).count('1')
        }

        if Params["DevConfig"][DevId]["DataFormat"]["NumAdcBits"] == 2:
            if Params["DevConfig"][DevId]["DataFormat"]["Format"] == 0:
                gAdcOneSampleSize = 2  # real data, one sample is 16bits = 2 bytes
            elif Params["DevConfig"][DevId]["DataFormat"]["Format"] in [1, 2]:
                gAdcOneSampleSize = 4  # complex data, one sample is 32bits = 4 bytes
            else:
                print("Error: unsupported ADC dataFmt")
        else:
            print(f"Error: unsupported ADC bits ({Params['DevConfig'][DevId]['DataFormat']['NumAdcBits']})")
        Params["DevConfig"][DevId]["DataFormat"]["gAdcOneSampleSize"] = gAdcOneSampleSize

        # Data path configuration
        DataPathConfig = MmWaveDevicesConfig[count]['rawDataCaptureConfig']['rlDevDataPathCfg_t']
        Params['DevConfig'][DevId]['DataPath'] = {
            'Interface': DataPathConfig['intfSel'],
            'Packet0': int(DataPathConfig['transferFmtPkt0'][2:], 16),
            'Packet1': int(DataPathConfig['transferFmtPkt1'][2:], 16),
            'CqConfig': DataPathConfig['cqConfig'],
            'Cq0TransSize': DataPathConfig['cq0TransSize'],
            'Cq1TransSize': DataPathConfig['cq1TransSize'],
            'Cq2TransSize': DataPathConfig['cq2TransSize']
        }

        if Params['DevConfig'][DevId]['DataPath']['Interface'] == 1:
            Params['DevConfig'][DevId]['DataPath'].update({
                'LaneMap': int(MmWaveDevicesConfig[count]['rawDataCaptureConfig']['rlDevLaneEnable_t']['laneEn'][2:], 16),
                'LvdsFormat': int(MmWaveDevicesConfig[count]['rawDataCaptureConfig']['rlDevLvdsLaneCfg_t']['laneFmtMap']),
                'LvdsMsbFirst': bool(int(MmWaveDevicesConfig[count]['rawDataCaptureConfig']['rlDevLvdsLaneCfg_t']['laneParamCfg'][2:], 16) & 1),
                'LvdsMsbCrcPresent': bool(int(MmWaveDevicesConfig[count]['rawDataCaptureConfig']['rlDevLvdsLaneCfg_t']['laneParamCfg'][2:], 16) & 2),
                'LvdsPktEndPulse': bool(int(MmWaveDevicesConfig[count]['rawDataCaptureConfig']['rlDevLvdsLaneCfg_t']['laneParamCfg'][2:], 16) & 4)
            })
        
        if Params['DevConfig'][DevId]['DataPath']['Interface'] == 0:
            Params['DevConfig'][DevId]['DataPath'].update({
                'CsiLane0Pos': int(MmWaveDevicesConfig[count]['rawDataCaptureConfig']['rlDevCsi2Cfg_t']['lanePosPolSel'][2:], 16) & 15,
                'CsiLane1Pos': (int(MmWaveDevicesConfig[count]['rawDataCaptureConfig']['rlDevCsi2Cfg_t']['lanePosPolSel'][2:], 16) >> 4) & 15,
                'CsiLane2Pos': (int(MmWaveDevicesConfig[count]['rawDataCaptureConfig']['rlDevCsi2Cfg_t']['lanePosPolSel'][2:], 16) >> 8) & 15,
                'CsiLane3Pos': (int(MmWaveDevicesConfig[count]['rawDataCaptureConfig']['rlDevCsi2Cfg_t']['lanePosPolSel'][2:], 16) >> 12) & 15,
                'CsiLaneClkPos': (int(MmWaveDevicesConfig[count]['rawDataCaptureConfig']['rlDevCsi2Cfg_t']['lanePosPolSel'][2:], 16) >> 16) & 15
            })

        # Frame configuration
        Params['FrameType'] = 0
        if Params['FrameType'] == 0:
            FrameConfig = MmWaveDevicesConfig[count]['rfConfig']['rlFrameCfg_t']
            Params['DevConfig'][DevId]['FrameConfig'] = {
                'ChirpIdx': FrameConfig['chirpStartIdx'],
                'ChirpEndIdx': FrameConfig['chirpEndIdx'],
                'NumChirpLoops': FrameConfig['numLoops'],
                'NumFrames': FrameConfig['numFrames'],
                'Periodicity': FrameConfig['framePeriodicity_msec'],
                'FrameTriggerDelay': FrameConfig['frameTriggerDelay']
            }
        
        if Params['FrameType'] == 1:
            # Advanced frame configuration
            AdvancedFrameSequence = MmWaveDevicesConfig[count]['rfConfig']['rlAdvFrameCfg_t']['frameSeq']
            AdvancedFrameData = MmWaveDevicesConfig[count]['rfConfig']['rlAdvFrameCfg_t']['frameData']
            Params['DevConfig'][DevId]['AdvFrame'] = {
                'NumFrames': AdvancedFrameSequence['numFrames'],
                'NumSubFrames': AdvancedFrameSequence['numOfSubFrames'],
                'FrameTriggerDelay': AdvancedFrameSequence['frameTrigDelay_usec']
            }
            # Subframe configuration
            for SubFrameId in range(len(AdvancedFrameSequence['subFrameCfg'])):
                sub_frame = AdvancedFrameSequence['subFrameCfg'][SubFrameId]['rlSubFrameCfg_t']
                Params['DevConfig'][DevId]['AdvFrame'].setdefault('SubFrame', {})[SubFrameId] = {
                    'ForceProfileIdx': sub_frame['forceProfileIdx'],
                    'ChirpStartIdx': sub_frame['chirpStartIdx'],
                    'NumChirp': sub_frame['numOfChirps'],
                    'NumChirpLoops': sub_frame['numLoops'],
                    'BurstPeriod': sub_frame['burstPeriodicity_msec'],
                    'ChirpStartIdOffset': sub_frame['chirpStartIdxOffset'],
                    'NumBurst': sub_frame['numOfBurst'],
                    'NumBurstLoops': sub_frame['numOfBurstLoops'],
                    'SubFramePeriod': sub_frame['subFramePeriodicity_msec'],
                    'ChirpsPerDataPkt': AdvancedFrameData['subframeDataCfg'][SubFrameId]['rlSubFrameDataCfg_t']['numChirpsInDataPacket']
                }
        
        if Params['FrameType'] == 2:
            # Continuous mode configuration
            ContinuousModeConfig = MmWaveDevicesConfig[count]['rfConfig']['rlContModeCfg_t']
            Params['DevConfig'][DevId]['ContFrame'] = {
                'StartFreq': ContinuousModeConfig['startFreqConst_GHz'],
                'SamplingRate': ContinuousModeConfig['digOutSampleRate'],
                'Tx0OutPowerBackoffCode': int(ContinuousModeConfig['txOutPowerBackoffCode'][2:], 16) & 255,
                'Tx1OutPowerBackoffCode': (int(ContinuousModeConfig['txOutPowerBackoffCode'][2:], 16) >> 8) & 255,
                'Tx2OutPowerBackoffCode': (int(ContinuousModeConfig['txOutPowerBackoffCode'][2:], 16) >> 16) & 255,
                'Tx0PhaseShifter': int(ContinuousModeConfig['txPhaseShifter'][2:], 16) & 255,
                'Tx1PhaseShifter': (int(ContinuousModeConfig['txPhaseShifter'][2:], 16) >> 8) & 255,
                'Tx2PhaseShifter': (int(ContinuousModeConfig['txPhaseShifter'][2:], 16) >> 16) & 255,
                'RxGain': ContinuousModeConfig['rxGain_dB'],
                'HpfCornerFreq1': ContinuousModeConfig['hpfCornerFreq1'],
                'HpfCornerFreq2': ContinuousModeConfig['hpfCornerFreq2']
            }
    Params['TxToEnable'] = TxToEnable
    Params['RxToEnable'] = RxToEnable
        
    # Return the configuration parameters
    return Params


def save_params(radar_params, config_path):
    """
    Save all the params related to the config file to yaml file.

    Parameters:
        radar_params (dict): Configuration parameters for the mmWave devices.
    
    Returns:
        None
    """

    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return obj
        return obj
    
    # Save radar parameters
    radar_params = convert_numpy(radar_params)
    file_path = f"{config_path}/radar_params.yaml"
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, 'w') as file:
        yaml.dump(radar_params, file, default_flow_style=False, sort_keys=False, width=-1)


if __name__ == "__main__":
    
    # Parse radar config
    with open("adc_list.yaml", "r") as file:
        data = yaml.safe_load(file)[0]
    config_path = os.path.join("data/radar_config", data["config"])
    
    # Test generate params
    radar_params = generate_params(config_path, data['radar'])
    if not radar_params:
        print("Invalid JSON files")
    else:
        print(radar_params)