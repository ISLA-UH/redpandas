import os
from redpandas.redpd_config import RedpdConfig, TFRConfig

# Absolute path to the skyfall data.
INPUT_DIR = "/Users/mgarces/Documents/DATA/SDK_DATA/api900_Skyfall_20201027"

# See above for example of a well named directory
# Avoid specifying /api900 directories if possible.
# INPUT_DIR = "/Users/spopen/redvox/data/skyfall_data"
# INPUT_DIR = "/Users/tyler/Documents/skyfall_pipeline"
# INPUT_DIR = "/Users/tokyok/Desktop/skyfall"
# INPUT_DIR = "/Users/meritxell/Desktop/skyfall_dummy_test"
INPUT_DIR = "/Users/meritxell/Desktop/skyfall_dummy_test_2"


# Build Bounder Data Products: Settings for skyfall_loc_rpd.py
is_rerun_bounder: bool = True  # If true, rerun and save as parquet
BOUNDER_DIR = "."
OTHER_INPUT_PATH = os.path.join(BOUNDER_DIR, "bounder")
OTHER_INPUT_FILE = "skyfall_bounder.csv"
OTHER_PD_PQT_FILE = "Skyfall_df_bounder.parquet"

# For lines 24 and below, no need to change as they contain Skyfall example set parameters.

# Reference lat, long, altitude and epoch s. Skyfall specific
ref_latitude_deg = 35.83728
ref_longitude_deg = -115.57234
ref_altitude_m = 1028.2
ref_epoch_s = 1603808160

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
                             tdr_load_method="datawindow")

# TFR configuration
tfr_config = TFRConfig(tfr_type='stft',
                       tfr_order_number_N=12,
                       show_fig_titles=False,
                       mesh_color_scale='range',
                       mesh_color_range={'Audio': 21.,
                                         'Bar': 18.,
                                         'Acc': 18.,
                                         'Gyr': 18.,
                                         'Mag': 18.},
                       sensor_highpass=True,
                       tfr_load_method="datawindow")
