import os

"""
Skyfall 2020 Configuration file
"""

EVENT_NAME = "Skyfall"

# Step 0: I/O files
# Absolute path to the skyfall data
INPUT_DIR = "/Users/mgarces/Documents/DATA/SDK_DATA/api900_Skyfall_20201027"

# TODO: /api900 only necessary to isolate 900, not good if both 900 and 1000 co-exist. Remove.
# INPUT_DIR = '/Users/jmtobin/Desktop/skyfall'
# TODO: Sarah, please separate Skyfall event into a new directory
# INPUT_DIR = "/Users/spopen/redvox/data/skyfall_data"
# INPUT_DIR = "/Users/tyler/Documents"
# INPUT_DIR = "/Users/tokyok/Desktop/skyfall"
# INPUT_DIR = "/Users/meritxell/Documents"

# Absolute path to bounder input data, could be a list
OTHER_INPUT_PATH = os.path.join(INPUT_DIR, "bounder")
OTHER_INPUT_FILE = "skyfall_bounder.csv"

# Absolute path for output pickle and parquet files
RPD_DIR = "rpd_files"
OUTPUT_DIR = os.path.join(INPUT_DIR, RPD_DIR)

# Check
if not os.path.exists(INPUT_DIR):
    print("Input directory does not exist, check path:")
    print(INPUT_DIR)
    exit()

if not os.path.exists(OUTPUT_DIR):
    print("Creating output directory")
    os.mkdir(OUTPUT_DIR)

# Data Window Pickle
DW_FILE = EVENT_NAME + ".pickle"

# RedPandas Parquets
PD_PQT_FILE = EVENT_NAME + "_df.parquet"
OTHER_PD_PQT_FILE = EVENT_NAME + "_df_bounder.parquet"

# Step 1: Station ID and Event time
STATIONS = ["1637610021"]
# Timestamps in seconds since UTC epoch
EVENT_ORIGIN_EPOCH_S = 1603806314  # 2020-10-27T13:45:14
duration_s = 30*60  # 30 minutes

# From start
EPISODE_START_EPOCH_S = EVENT_ORIGIN_EPOCH_S
EPISODE_END_EPOCH_S = EVENT_ORIGIN_EPOCH_S + duration_s

# # Reference lat lon altitude and time at terminus
event_ref_latitude_deg = 35.83728
event_ref_longitude_deg = -115.57234
event_ref_altitude_m = 1028.2
event_ref_epoch_s = EPISODE_END_EPOCH_S

# Step 2: Sensor selection
SENSOR_LABEL = ['audio', 'barometer', 'accelerometer', 'magnetometer', 'gyroscope',
                'health', 'location', 'clock', 'synchronization']

# Step 3: Pipeline actions, needed parquet for geospatial
build_dw_pickle: bool = False  # Handling of RDVX DataWindow structure
print_datawindow_dq: bool = False  # Print basic DQ/DA to screen
plot_mic_waveforms: bool = False  # Show raw RDVX DataWindow waveforms
build_df_parquet: bool = True  # Export pandas data frame as parquet

# Build TDR Data Products: Settings for skyfall_tdr_rpd.py
use_datawindow: bool = False  # Load data using RDVX DataWindow
use_pickle: bool = False  # Load data using a pickle with RDVX DataWindow, either serialized or not
use_parquet: bool = True  # Load data using parquet with RedPandas dataframe

# Build TFR Data Products: Settings for skyfall_tfr_rpd.py
PIPELINE_LABEL = ['TBD']

