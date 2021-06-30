import os

from redpandas.redpd_config import RedpdConfig, DataLoadMethod
from typing import Optional, Union, Dict

# Absolute path to the skyfall data.
INPUT_DIR = "/Users/mgarces/Documents/DATA/SDK_DATA/api900_Skyfall_20201027"

# See above for example of a well named directory
# Avoid specifying /api900 directories if possible.
# INPUT_DIR = '/Users/jmtobin/Desktop/skyfall'
# INPUT_DIR = "/Users/spopen/redvox/data/skyfall_data"
# INPUT_DIR = "/Users/tyler/Documents/skyfall_pipeline"
# INPUT_DIR = "/Users/tokyok/Desktop/skyfall"
# INPUT_DIR = "/Users/meritxell/Desktop/skyfall_dummy_test"

# TODO MC: one slash vs two slash for windows, only edit input_dir, make sure inpur_dir is correct

# Build Bounder Data Products: Settings for skyfall_loc_rpd.py
is_rerun_bounder: bool = True  # If true, rerun and save as parquet
OTHER_INPUT_PATH = os.path.join(INPUT_DIR, "bounder")
OTHER_INPUT_FILE = "skyfall_bounder.csv"
OTHER_PD_PQT_FILE = "Skyfall_df_bounder.parquet"

# Propagated to tdr
ref_latitude_deg = 35.83728
ref_longitude_deg = -115.57234
ref_altitude_m = 1028.2
ref_epoch_s = 1603808160


# TFR configuration
# TODO MC/SP: add TFR parameters to RedpdConfig, eliminate TFRConfig


class TFRConfig:

    def __init__(self, tfr_type: str,
                 tfr_order_number_N: int,
                 show_fig_titles: bool,
                 mesh_color_scale: Optional[Union[Dict[str, str] or str]] = 'range',
                 mesh_color_range: Optional[Union[Dict[str, float] or float]] = 18.,
                 sensor_highpass: Optional[Union[Dict[str, bool] or bool]] = True,
                 tfr_load_method: Optional[str] = "datawindow"):
        """
        Configuration parameters for skyfall_tfr_rpd

        :param tfr_type: string, 'stft' or 'cwt'
        :param tfr_order_number_N: int, order number of the transform
        :param show_fig_titles: bool, display or hide figure titles
        :param mesh_color_scale: string or dictionary of strings, color scale mode for spectrograms
        :param mesh_color_range: float or dictionary of floats, color range for spectrograms
        :param sensor_highpass: boolean or dictionary of booleans, use highpass of data if available
        :param tfr_load_method: optional string, chose loading data method: "datawindow", "pickle", or "parquet"
        """
        self.tfr_type = tfr_type
        self.tfr_order_number_N = tfr_order_number_N
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


tfr_config = TFRConfig(tfr_type='stft',
                       tfr_order_number_N=12,
                       show_fig_titles=False,
                       mesh_color_scale='range',
                       mesh_color_range={'Audio': 21.,
                                         'Bar': 18.,
                                         'Acc': 18.,
                                         'Gyr': 18.,
                                         'Mag': 18.},
                       sensor_highpass=True)

# Redpd Configuration
skyfall_config = RedpdConfig(input_directory=INPUT_DIR,
                             event_name="Skyfall",
                             output_directory=os.path.join(INPUT_DIR, "rpd_files"),
                             station_ids=["1637610021"],
                             sensor_labels=['audio', 'barometer', 'accelerometer', 'magnetometer', 'gyroscope',
                                            'health', 'location', 'clock', 'synchronization'],
                             event_start_epoch_s=1603806314,
                             duration_s=30 * 60,
                             start_buffer_minutes=3,
                             end_buffer_minutes=3,
                             tdr_load_method="parquet")
