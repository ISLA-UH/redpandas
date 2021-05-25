import os
"""
Skyfall 2020 Configuration file
"""

EVENT_NAME = "Skyfall"

# Step 0: I/O files
# Absolute path to the skyfall data
INPUT_DIR = "/Users/mgarces/Documents/DATA/SDK_DATA/api900_Skyfall_20201027/"
# INPUT_DIR = '/Users/jmtobin/Desktop/skyfall/api900'
# INPUT_DIR = '/Users/spopen/redvox/data/spacex_data/falcon9/api900'
# INPUT_DIR = "/Users/tyler/Documents/api900"
# INPUT_DIR = "/Users/tokyok/Desktop/skyfall/api900"
# INPUT_DIR = "/Users/meritxell/Documents/api900"

OUTPUT_DIR = os.path.join(INPUT_DIR, "rpd_files")  # Absolute path for output pickle and parquet files
# Data Window Pickle
DW_FILE = EVENT_NAME + ".pickle"
# RedPandas Parquets
PD_PQT_FILE = EVENT_NAME + "_df.parquet"
PD_PQT_FILE_LOC = EVENT_NAME + "loc_df.parquet"

# Step 1: Station ID and Event time
STATIONS = ["1637610021"]
# Timestamps in seconds since UTC epoch
EVENT_ORIGIN_EPOCH_S = 1603806314  # 2020-10-27T13:45:14
duration_s = 30*60  # 30 minutes

# From start
EPISODE_START_EPOCH_S = EVENT_ORIGIN_EPOCH_S
EPISODE_END_EPOCH_S = EVENT_ORIGIN_EPOCH_S + duration_s

# Step 2: Sensor selection
SENSOR_LABEL = ['audio', 'barometer', 'accelerometer', 'magnetometer', 'gyroscope',
                'health', 'location', 'clock', 'synchronization']

# Step 3: Pipeline actions
build_dw_pickle: bool = True  # Handling of RDVX DataWindow structure
print_datawindow_dq: bool = False  # Print basic DQ/DA to screen
plot_mic_waveforms: bool = False  # Show raw RDVX DataWindow waveforms
build_df_parquet: bool = True  # Export pandas data frame as parquet

# Build TDR Data Products: Settings for skyfall_tdr_rpd.py
use_datawindow: bool = False  # Load data using RDVX DataWindow
use_pickle: bool = False  # Load data using a pickle with RDVX DataWindow, either serialized or not
use_parquet: bool = True  # Load data using parquet with RedPandas dataframe

# Build TFR Data Products: Settings for skyfall_tfr_rpd.py
PIPELINE_LABEL = ['TBD']

