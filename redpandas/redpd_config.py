import os
import enum
from typing import List, Optional

import pprint


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


# todo: finish defining required and optional/not required fields with defaults
# Minimum reqs are event_name, input_directory, sensor_label = ['audio']
class RedpdConfig:

    def __init__(self, input_directory: str,
                 event_name: str = "Redvox",
                 output_directory: Optional[str] = None,
                 output_filename_pkl_pqt: Optional[str] = None,
                 station_ids: Optional[List[str]] = None,
                 sensor_labels: Optional[List[str]] = None,
                 event_start_epoch_s: Optional[float] = None,
                 duration_s: Optional[int] = None,
                 start_buffer_minutes: Optional[int] = 3,
                 end_buffer_minutes: Optional[int] = 3,
                 debug_dw: bool = False,
                 tdr_load_method: Optional[str] = "datawindow" ):

        """
        Configuration paramters for RedPandas

        :param input_directory: string, directory that contains the files to read data from.  REQUIRED
        :param event_name: optional string, name of event. Default is "Redvox"
        :param output_directory: optional string, directory to created save pickle/JSON/parquet
        :param output_filename_pkl_pqt: optional string, name of created parquet and pickle files
        :param station_ids: optional list of strings, list of station ids to filter on
        :param sensor_labels: optional list of strings, list of sensors. Default is "audio"
        :param event_start_epoch_s: optional float, start time in epoch s. Default is None
        :param duration_s: optional int, durtion of event in minutes. Default is None
        :param start_buffer_minutes: float representing the amount of minutes to include before the start datetime
         when filtering data. Default is 3
        :param end_buffer_minutes: float representing the amount of minutes to include before the end datetime
         when filtering data. Default is 3
        :param debug_dw: bool, if True, output additional information when processing data window.  Default False
        :param tdr_load_method: optional string, chose loading data method: "datawindow", "pickle", or "parquet".
        Default is "datawindow"
        """

        self.input_dir = input_directory
        self.event_name = event_name

        # Check if input and output dir exists
        if not os.path.exists(self.input_dir):
            print(f"Input directory does not exist, check path: {self.input_dir}")
            exit()

        if output_directory is not None:
            self.output_dir = output_directory
            if not os.path.exists(self.output_dir):
                print(f"Creating output directory: {self.output_dir}")
                os.mkdir(self.output_dir)
        else:
            self.output_dir = os.path.join(self.input_dir, "rpd_files")

        if output_filename_pkl_pqt is None:
            self.output_filename_base = event_name

        self.dw_file: str = self.event_name + ".pkl"
        self.pd_pqt_file: str = self.event_name + "_df.parquet"

        self.bounder_input_path = os.path.join(self.input_dir, "bounder")
        self.bounder_input_csv_file = "skyfall_bounder.csv"
        self.bounder_pd_pqt_file = self.event_name + "_df_bounder.parquet"

        self.station_ids = station_ids

        if sensor_labels is not None:
            self.sensor_labels = sensor_labels
        else:
            self.sensor_labels = ["audio"]

        self.event_start_epoch_s = event_start_epoch_s
        self.duration_s = duration_s

        if duration_s is not None:
            self.event_end_epoch_s: float = self.event_start_epoch_s + self.duration_s
        else:
            self.event_end_epoch_s = None

        self.start_buffer_minutes = start_buffer_minutes
        self.end_buffer_minutes = end_buffer_minutes
        self.debug = debug_dw

        self.tdr_load_method = DataLoadMethod.method_from_str(tdr_load_method)

        # todo: what is this?
        # self.pipeline_label: List[str] = ['TBD']



    def pretty(self) -> str:
        # noinspection Mypy
        return pprint.pformat(vars(self))


if __name__ == "__main__":

    RedpdConfig(input_directory="/Users/meritxell/Desktop/skyfall_dummy_test",
                 tdr_load_method="parquet")

    print(DataLoadMethod.method_from_str("parquet"))

# TODO: Do not instantiate an object here; put it in an example or demo file
# Next step: start time, duration, stations, clock, location

# example_config = RedpdConfig(
#     event_name="Skyfall",
#     input_directory=INPUT_DIR,
#     sensor_labels=['audio', 'barometer', 'accelerometer', 'magnetometer', 'gyroscope',
#                    'health', 'location', 'clock', 'synchronization'],
#     rpd_dir="rpd_files",
#     output_directory=os.path.join(INPUT_DIR, RPD_DIR),
#     stations=["1637610021"],
#     episode_start_epoch_s=1603806314,  # 2020-10-27T13:45:14
#     duration_s=30*60,                  # 30 minutes
#     ref_latitude_deg=35.83728,
#     ref_longitude_deg=-115.57234,
#     ref_altitude_m=1028.2,
#     ref_epoch_s=1603808160,
#     compress_dw=True,
#     print_dw_quality=False,
#     plot_mic_waveforms=False,
#     build_df_parquet=True,
#     tdr_load_method=DataLoadMethod.PARQUET,
#     tfr_load_method=DataLoadMethod.PARQUET,
#     band_order_Nth=12,
#     verbosity=1,
#     is_rerun_bounder=True
# )


# TODO MC: think about what can be added in the future projects

# # Absolute path for output directories
# rpd_dir: str
#
# # Station ID and Event time
# # Timestamps in seconds since UTC epoch
# stations: List[str]
# episode_start_epoch_s: float
# duration_s: int
#
# # Reference lat lon altitude and time at terminus, from Bounder
# ref_latitude_deg: float
# ref_longitude_deg: float
# ref_altitude_m: float
# ref_epoch_s: int
#
# # Pipeline actions
# compress_dw: bool           # Handling of RDVX DataWindow structure
# print_dw_quality: bool      # Print basic DQ/DA to screen
# plot_mic_waveforms: bool    # Show raw RDVX DataWindow waveforms
# build_df_parquet: bool      # Export pandas data frame as parquet
#
# # Settings for skyfall_tdr_rpd.py
# tdr_load_method: DataLoadMethod
#
# # Settings for skyfall_tfr_rpd.py
# tfr_load_method: DataLoadMethod
# band_order_Nth: int
# verbosity: int = 1          # verbosity > 1, plots extra raw and highpass plots
#
# # Build Bounder Data Products: Settings for skyfall_loc_rpd.py
# is_rerun_bounder: bool = True  # If true, rerun and save as parquet

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