from redpandas.redpd_dw_to_parquet import redpd_dw_to_parquet
from redpandas.redpd_ensonify import ensonify_sensors_pandas
import pandas as pd
import redpandas.redpd_plot as rpd_plot
import matplotlib.pyplot as plt

# redpd_dw_to_parquet(input_dir='/Users/meritxell/Documents/api_m_pipeline_tests/20210617_Sweep_test_students',
#                     sensor_labels=["audio", "barometer", "accelerometer", "gyroscope", "magnetometer"])


INPUT_DIR = '/Users/meritxell/Documents/api_m_pipeline_tests/20210617_Sweep_test_students'
df = pd.read_parquet(INPUT_DIR + "/rpd_files/Redvox_df.parquet")


# sensor_column_label_list = ["audio_wf"]
# sensor_fs_label_list = ["audio_sample_rate_nominal_hz"]
# sensor_name_key_list = ["aud"]
#
# ensonify_sensors_pandas(df=df,
#                         sig_id_label='station_id',
#                         sig_sample_rate_label_list=sensor_fs_label_list,
#                         sensor_column_label_list=sensor_column_label_list,
#                         wav_sample_rate_hz=192000.,
#                         output_wav_directory=INPUT_DIR,
#                         output_wav_filename='A_cool_example',
#                         sensor_name_list=sensor_name_key_list)


rpd_plot.plot_wiggles_pandas(df=df,
                             sig_wf_label="audio_wf",
                             sig_sample_rate_label="audio_sample_rate_nominal_hz",
                             sig_id_label="station_id",
                             x_label="Time (s)",
                             y_label="Stations",
                             fig_title_show=True,
                             fig_title='Audio',
                             sig_timestamps_label='audio_epoch_s',
                             custom_yticks=["Station 1", "Station 2", "Station 3"])


rpd_plot.plot_wiggles_pandas(df=df,
                             sig_wf_label=["barometer_wf_raw", "audio_wf"],
                             sig_sample_rate_label=["barometer_sample_rate_hz", "audio_sample_rate_nominal_hz"],
                             sig_id_label="station_id",
                             station_id_str="1637610012",
                             x_label="Time (s)",
                             y_label="Sensors",
                             fig_title_show=True,
                             fig_title='Signals for Station 2',
                             sig_timestamps_label=['barometer_epoch_s', 'audio_epoch_s'],
                             custom_yticks=["Bar", "Aud"])

# plt.show()

import redpandas.redpd_filter as rpd_filter

rpd_filter.signal_zero_mean_pandas(df=df,
                                   sig_wf_label="audio_wf")