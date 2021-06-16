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


# Minimum reqs are event_name, input_dir, sensor_label = ['audio']
class RedpdConfig:

    def __init__(self, input_dir: str, event_name: str = "Redvox", sensor_labels: Optional[List[str]] = None,
                 output_dir: Optional[str] = None):
        self.input_dir = input_dir
        self.event_name = event_name
        if sensor_labels is not None:
            self.sensor_labels = sensor_labels
        else:
            self.sensor_labels = ["audio"]

        # Check if input and output dir exists
        if not os.path.exists(self.input_dir):
            print(f"Input directory does not exist, check path: {self.input_dir}")
            exit()

        if output_dir is not None:
            self.output_dir = output_dir
            if not os.path.exists(self.output_dir):
                print(f"Creating output directory: {self.output_dir}")
                os.mkdir(self.output_dir)
        else:
            self.output_dir = os.path.join(self.input_dir, "rpd_files")

        self.dw_file: str = self.event_name + ".pkl"
        self.pd_pqt_file: str = self.event_name + "_df.parquet"
        self.bounder_input_path = os.path.join(self.input_dir, "bounder")
        self.bounder_input_csv_file = "skyfall_bounder.csv"
        self.bounder_pd_pqt_file = self.event_name + "_df_bounder.parquet"

        self.episode_end_epoch_s: float = self.episode_start_epoch_s + self.duration_s

        # todo: what is this?
        self.pipeline_label: List[str] = ['TBD']

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

    def pretty(self) -> str:
        # noinspection Mypy
        return pprint.pformat(self.to_dict())