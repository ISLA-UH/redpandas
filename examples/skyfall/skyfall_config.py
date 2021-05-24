import os
"""
Vegas Skyfall Configuration file
"""
# TODO: Build load gui
EVENT_NAME = "Skyfall"
# Absolute path to the skyfall data
# INPUT_DIR = "/Users/mgarces/Documents/DATA/SDK_DATA/api900_Skyfall_20201027/"
# INPUT_DIR = '/Users/jmtobin/Desktop/skyfall/api900'
# INPUT_DIR = '/Users/spopen/redvox/data/spacex_data/falcon9/api900'
# INPUT_DIR = "/Users/tyler/Documents/api900"
# INPUT_DIR = "/Users/tokyok/Desktop/skyfall/api900"
INPUT_DIR = "/Users/meritxell/Documents/api900"

OUTPUT_DIR = os.path.join(INPUT_DIR, "rpd_files")  # Absolute path for output pickle and parquet files
DW_FILE = EVENT_NAME + ".pickle"
PD_PQT_FILE = EVENT_NAME + "_df.parquet"
STATIONS = {"1637610021"}

# Timestamps in seconds since UTC epoch
# EVENT_ORIGIN_EPOCH_S = 1603806300  # 2020-10-27T13:45
EVENT_ORIGIN_EPOCH_S = 1603806314  # 2020-10-27T13:45:13.132 start time of first data packet
# minutes = 2
# duration = minutes*60  # ALWAYS start small
duration = 30*60  # 30 minutes
# From start
EPISODE_START_EPOCH_S = EVENT_ORIGIN_EPOCH_S
EPISODE_END_EPOCH_S = EVENT_ORIGIN_EPOCH_S + duration

# # From impact SOI
# time_edge_s = 120
# duration = 30*60  # 30 minutes
# EPISODE_CENTER_EPOCH_S = EVENT_ORIGIN_EPOCH_S + duration - 150
# EPISODE_START_EPOCH_S = EPISODE_CENTER_EPOCH_S - time_edge_s
# EPISODE_END_EPOCH_S = EPISODE_CENTER_EPOCH_S + time_edge_s

# Settings for skyfall_template_basic_rpd.py (Step 1)
# Available sensors: 'audio', 'barometer', 'accelerometer', 'magnetometer', 'gyroscope', 'location', 'health', 'image',
# 'clock' and 'synchronization'
SENSOR_LABEL = ['audio', 'barometer', 'accelerometer', 'magnetometer', 'gyroscope',
                'health', 'location', 'clock', 'synchronization']

build_dw_pickle: bool = True  # Handling of RDVX DataWindow structure
print_datawindow_dq: bool = False  # Print basic DQ/DA to screen
plot_mic_waveforms: bool = False  # Show raw RDVX DataWindow waveforms
build_df_parquet: bool = True  # Export pandas data frame as parquet

# Settings for skyfall_tdr_rpd.py (Step 2)
use_datawindow: bool = False  # Load data using RDVX DataWindow
use_pickle: bool = False  # Load data using a pickle with RDVX DataWindow, either serialized or not
use_parquet: bool = True  # Load data using parquet with RedPandas dataframe

PIPELINE_LABEL = ['TBD']

