"""
Configuration class for RedPandas.
"""
import os
import enum
from typing import List, Optional, Dict, Union

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
                 tdr_load_method: Optional[str] = "datawindow"):
        """
        Configuration parameters for RedPandas

        :param input_directory: string, directory that contains the files to read data from.  REQUIRED
        :param event_name: optional string, name of event. Default is "Redvox"
        :param output_directory: optional string, directory to created save pickle/JSON/parquet
        :param output_filename_pkl_pqt: optional string, name of created parquet and pickle files
        :param station_ids: optional list of strings, list of station ids to filter on
        :param sensor_labels: optional list of strings, list of sensors. Default is "audio"
        :param event_start_epoch_s: optional float, start time in epoch s. Default is None
        :param duration_s: optional int, duration of event in minutes. Default is None
        :param start_buffer_minutes: float representing the amount of minutes to include before the start datetime
            when filtering data. Default is 3
        :param end_buffer_minutes: float representing the amount of minutes to include before the end datetime
            when filtering data. Default is 3
        :param tdr_load_method: optional string, chose loading data method: "datawindow", "pickle", or "parquet".
            Default is "datawindow"
        """
        self.input_dir = input_directory
        self.event_name = event_name

        # Check if input and output dir exists
        if not os.path.exists(self.input_dir):
            raise FileNotFoundError(f"Input directory does not exist, check path: {self.input_dir}")

        if output_directory is not None:
            self.output_dir = output_directory
            if not os.path.exists(self.output_dir):
                print(f"Creating output directory: {self.output_dir}")
                os.makedirs(self.output_dir)
        else:
            self.output_dir = os.path.join(self.input_dir, "rpd_files")

        self.output_filename_pkl_pqt = event_name if output_filename_pkl_pqt is None else output_filename_pkl_pqt
        self.dw_file: str = self.output_filename_pkl_pqt + ".pkl"
        self.pd_pqt_file: str = self.output_filename_pkl_pqt + "_df.parquet"

        self.station_ids = station_ids
        self.sensor_labels = ["audio"] if sensor_labels is None else sensor_labels

        self.event_start_epoch_s = event_start_epoch_s
        self.duration_s = duration_s
        self.event_end_epoch_s: Optional[float] = \
            None if duration_s is None else self.event_start_epoch_s + self.duration_s
        self.start_buffer_minutes = start_buffer_minutes
        self.end_buffer_minutes = end_buffer_minutes

        self.tdr_load_method = DataLoadMethod.method_from_str(tdr_load_method)

    def pretty(self) -> str:
        # noinspection Mypy
        return pprint.pformat(vars(self))


class TFRConfig:
    def __init__(self, tfr_type: str,
                 tfr_order_number_n: int,
                 show_fig_titles: bool,
                 mesh_color_scale: Optional[Union[Dict[str, str] or str]] = 'range',
                 mesh_color_range: Optional[Union[Dict[str, float] or float]] = 18.,
                 sensor_highpass: Optional[Union[Dict[str, bool] or bool]] = True,
                 tfr_load_method: Optional[str] = "datawindow"):
        """
        Configuration parameters for skyfall_tfr_rpd

        :param tfr_type: string, 'stft' or 'cwt'
        :param tfr_order_number_n: int, order number of the transform
        :param show_fig_titles: bool, display or hide figure titles
        :param mesh_color_scale: string or dictionary of strings, color scale mode for spectrograms
        :param mesh_color_range: float or dictionary of floats, color range for spectrograms
        :param sensor_highpass: boolean or dictionary of booleans, use highpass of data if available
        :param tfr_load_method: optional string, choose loading data method: "datawindow", "pickle", or "parquet"
        """
        self.tfr_type = tfr_type
        self.tfr_order_number_n = tfr_order_number_n
        self.show_fig_titles = show_fig_titles
        self.tfr_load_method = DataLoadMethod.method_from_str(tfr_load_method)
        self.sensor_labels = ['Audio', 'Bar', 'Acc', 'Gyr', 'Mag']
        n = len(self.sensor_labels)

        if type(mesh_color_scale) == str:
            self.mc_scale = dict(zip(self.sensor_labels, n*[mesh_color_scale]))
        else:
            self.mc_scale = dict(zip(self.sensor_labels, n*['range']))
            for label in mesh_color_scale.keys():
                self.mc_scale[label] = mesh_color_scale[label]

        if type(mesh_color_range) == float:
            self.mc_range = dict(zip(self.sensor_labels, n*[mesh_color_range]))
        else:
            self.mc_range = dict(zip(self.sensor_labels, n*[18.]))
            for label in mesh_color_range.keys():
                self.mc_range[label] = mesh_color_range[label]

        if type(sensor_highpass) == bool:
            self.sensor_hp = dict(zip(self.sensor_labels[1:], (n-1)*[sensor_highpass]))
        else:
            self.sensor_hp = dict(zip(self.sensor_labels[1:], (n-1)*[True]))
            for label in sensor_highpass.keys():
                self.sensor_hp[label] = sensor_highpass[label]

        self.sensor_3d = dict(zip(['Audio', 'Bar', 'Acc', 'Gyr', 'Mag'], [False, False, True, True, True]))
