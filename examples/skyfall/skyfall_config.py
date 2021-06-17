import os

from redpandas.redpd_config import RedpdConfig

"""
Skyfall 2020 Configuration file
"""

# Step 0: I/O files
# Absolute path to the skyfall data
INPUT_DIR = "/Users/mgarces/Documents/DATA/SDK_DATA/api900_Skyfall_20201027"

# See above for example of a well named directory
# Avoid specifying /api900 directories if possible.
# INPUT_DIR = '/Users/jmtobin/Desktop/skyfall'
# INPUT_DIR = "/Users/spopen/redvox/data/skyfall_data"
# INPUT_DIR = "/Users/tyler/Documents/skyfall_pipeline"
# INPUT_DIR = "/Users/tokyok/Desktop/skyfall"
# INPUT_DIR = "/Users/meritxell/Documents/skyfall"

RPD_DIR = "rpd_files"
EVENT_NAME = "Skyfall"

# Absolute path for output pickle and parquet files
OUTPUT_DIR = os.path.join(INPUT_DIR, RPD_DIR)

# Data Window Pickle
DW_FILE = EVENT_NAME  # + ".pickle"

# RedPandas Parquets
PD_PQT_FILE = EVENT_NAME + "_df.parquet"

# Absolute path to bounder data; could be a list
OTHER_INPUT_PATH = os.path.join(INPUT_DIR, "bounder")
OTHER_INPUT_FILE = "skyfall_bounder.csv"
OTHER_PD_PQT_FILE = EVENT_NAME + "_df_bounder.parquet"

# Step 1: Station ID and Event time
STATIONS = ["1637610021"]
# Timestamps in seconds since UTC epoch
EVENT_ORIGIN_EPOCH_S = 1603806314  # 2020-10-27T13:45:14
duration_s = 30*60  # 30 minutes

# From start
EPISODE_START_EPOCH_S = EVENT_ORIGIN_EPOCH_S
EPISODE_END_EPOCH_S = EVENT_ORIGIN_EPOCH_S + duration_s

# Reference lat lon altitude and time at terminus, from Bounder
ref_latitude_deg = 35.83728
ref_longitude_deg = -115.57234
ref_altitude_m = 1028.2
ref_epoch_s = 1603808160

# Step 2: Sensor selection
SENSOR_LABEL = ['audio', 'barometer', 'accelerometer', 'magnetometer', 'gyroscope',
                'health', 'location', 'clock', 'synchronization']

# Step 3: Pipeline actions, needed parquet for geospatial
build_dw_pickle: bool = False  # Handling of RDVX DataWindow structure
print_datawindow_dq: bool = False  # Print basic DQ/DA to screen
plot_mic_waveforms: bool = False  # Show raw RDVX DataWindow waveforms
build_df_parquet: bool = True  # Export pandas data frame as parquet

# Build TDR Data Products: Settings for skyfall_tdr_rpd.py
# Choose one method to load the Skyfall data
# todo: do not use boolean if this is an enum
use_datawindow_tdr: bool = False  # Load data using RDVX DataWindow
use_pickle_tdr: bool = False  # Load data using a pickle with RDVX DataWindow, either serialized or not
use_parquet_tdr: bool = True  # Load data using parquet with RedPandas dataframe

# Build TFR Data Products: Settings for skyfall_tfr_rpd.py
# Choose one method to load the Skyfall data
# todo: do not use boolean if this is an enum
use_datawindow: bool = False  # Load data using RDVX DataWindow
use_pickle: bool = False  # Load data using a pickle with RDVX DataWindow, either serialized or not
use_parquet: bool = True  # Load data using parquet with RedPandas dataframe
# band_order_Nth = 12
# axes = ["X", "Y", "Z"]
# verbosity = 1  # verbosity > 1, plots extra raw and highpass plots

# Build Bounder Data Products: Settings for skyfall_loc_rpd.py
is_rerun_bounder: bool = True  # If true, rerun and save as parquet

PIPELINE_LABEL = ['TBD']

skyfall_config = RedpdConfig(input_dir=INPUT_DIR,
                             event_name=EVENT_NAME,
                             sensor_labels=SENSOR_LABEL,
                             output_dir=OUTPUT_DIR,
                             )
