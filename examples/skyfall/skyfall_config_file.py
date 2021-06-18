import os

from redpandas.redpd_config import RedpdConfig

# Absolute path to the skyfall data.
# INPUT_DIR = "/Users/mgarces/Documents/DATA/SDK_DATA/api900_Skyfall_20201027"

# See above for example of a well named directory
# Avoid specifying /api900 directories if possible.
# INPUT_DIR = '/Users/jmtobin/Desktop/skyfall'
# INPUT_DIR = "/Users/spopen/redvox/data/skyfall_data"
# INPUT_DIR = "/Users/tyler/Documents/skyfall_pipeline"
# INPUT_DIR = "/Users/tokyok/Desktop/skyfall"
INPUT_DIR = "/Users/meritxell/Desktop/skyfall_dummy_test"

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

# Build Bounder Data Products: Settings for skyfall_loc_rpd.py
is_rerun_bounder: bool = True  # If true, rerun and save as parquet
OTHER_INPUT_PATH = os.path.join(INPUT_DIR, "bounder")
OTHER_INPUT_FILE = "skyfall_bounder.csv"
OTHER_PD_PQT_FILE = "Skyfall_df_bounder.parquet"
