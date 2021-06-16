import os
import enum
from typing import List, MutableMapping
from dataclasses import dataclass
from dataclasses_json import dataclass_json

import pprint

"""
Skyfall 2020 Configuration file
"""


class DataLoadMethod(enum.Enum):
    UNKNOWN = 0
    DATAWINDOW = 1
    PICKLE = 2
    PARQUET = 3

    @staticmethod
    def method_from_str(method_str: str) -> "DataLoadMethod":
        if method_str.lower() == "datawindow":
            return DataLoadMethod.DATAWINDOW
        elif method_str.lower() == "pickle":
            return DataLoadMethod.PICKLE
        elif method_str.lower() == "parquet":
            return DataLoadMethod.PARQUET
        else:
            return DataLoadMethod.UNKNOWN

@dataclass_json()
@dataclass
class SkyfallConfig:
    event_name: str

    # I/O files
    # Absolute path to the data directory
    input_dir: str
    # Absolute path for output directories
    rpd_dir: str
    output_dir: str

    # Station ID and Event time
    # Timestamps in seconds since UTC epoch
    stations: List[str]
    episode_start_epoch_s: float
    duration_s: int

    # Reference lat lon altitude and time at terminus, from Bounder
    ref_latitude_deg: float
    ref_longitude_deg: float
    ref_altitude_m: float
    ref_epoch_s: int

    # Sensor selection
    sensor_labels: List[str]

    # Pipeline actions
    compress_dw: bool           # Handling of RDVX DataWindow structure
    print_dw_quality: bool      # Print basic DQ/DA to screen
    plot_mic_waveforms: bool    # Show raw RDVX DataWindow waveforms
    build_df_parquet: bool      # Export pandas data frame as parquet

    # Settings for skyfall_tdr_rpd.py
    tdr_load_method: DataLoadMethod

    # Settings for skyfall_tfr_rpd.py
    tfr_load_method: DataLoadMethod
    band_order_Nth: int
    verbosity: int = 1          # verbosity > 1, plots extra raw and highpass plots

    # Build Bounder Data Products: Settings for skyfall_loc_rpd.py
    is_rerun_bounder: bool = True  # If true, rerun and save as parquet

    def __post_init__(self):
        # Check if input and output dir exists
        if not os.path.exists(self.input_dir):
            print(f"Input directory does not exist, check path: {self.input_dir}")
            exit()

        if not os.path.exists(self.output_dir):
            print(f"Creating output directory: {self.output_dir}")
            os.mkdir(self.output_dir)

        self.dw_file: str = self.event_name + ".pkl"
        self.pd_pqt_file: str = self.event_name + "_df.parquet"
        self.bounder_input_path = os.path.join(self.input_dir, "bounder")
        self.bounder_input_csv_file = "skyfall_bounder.csv"
        self.bounder_pd_pqt_file = self.event_name + "_df_bounder.parquet"

        self.episode_end_epoch_s: float = self.episode_start_epoch_s + self.duration_s

        # todo: what is this?
        self.pipeline_label: List[str] = ['TBD']

    # @staticmethod
    # def from_path(config_path: str) -> "SkyfallConfig":
    #     try:
    #         with open(config_path, "r") as config_in:
    #             config_dict: MutableMapping = toml.load(config_in)
    #             # noinspection Mypy
    #             return SkyfallConfig.from_dict(config_dict)
    #     except Exception as e:
    #         print(f"Error loading configuration at: {config_path}")
    #         raise e

    def pretty(self) -> str:
        # noinspection Mypy
        return pprint.pformat(self.to_dict())


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

# TODO: Minimum reqs are event_name, input_dir, sensor_label = ['audio']
# Next step: start time, duration, stations, clock, location

skyfall_config = SkyfallConfig(
    event_name="Skyfall",
    input_dir=INPUT_DIR,
    sensor_labels=['audio', 'barometer', 'accelerometer', 'magnetometer', 'gyroscope',
                   'health', 'location', 'clock', 'synchronization'],
    rpd_dir="rpd_files",
    output_dir=os.path.join(INPUT_DIR, RPD_DIR),
    stations=["1637610021"],
    episode_start_epoch_s=1603806314,  # 2020-10-27T13:45:14
    duration_s=30*60,                  # 30 minutes
    ref_latitude_deg=35.83728,
    ref_longitude_deg=-115.57234,
    ref_altitude_m=1028.2,
    ref_epoch_s=1603808160,
    compress_dw=True,
    print_dw_quality=False,
    plot_mic_waveforms=False,
    build_df_parquet=True,
    tdr_load_method=DataLoadMethod.PARQUET,
    tfr_load_method=DataLoadMethod.PARQUET,
    band_order_Nth=12,
    verbosity=1,
    is_rerun_bounder=True
)

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
