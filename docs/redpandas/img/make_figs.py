from redpandas.redpd_dw_to_parquet import redpd_df
import pandas as pd
import redpandas.redpd_plot.redpd_plot as rpd_plot
import matplotlib.pyplot as plt

redpd_df(input_dir='/Users/meritxell/Documents/api_m_pipeline_tests/20210617_Sweep_test_students',
         sensor_labels=["audio", "barometer", "accelerometer", "gyroscope", "magnetometer"],
         start_epoch_s=1623962734, end_epoch_s=1623962730 + 15)


# INPUT_DIR = '/Users/meritxell/Documents/api_m_pipeline_tests/20210617_Sweep_test_students'
df = pd.read_parquet('/Users/meritxell/Documents/api_m_pipeline_tests/20210617_Sweep_test_students' + "/rpd_files/Redvox_df.parquet")


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

# rpd_filter.decimate_signal_pandas(df=df,
#                                   downsample_frequency_hz=20,
#                                   sig_id_label="station_id",
#                                   sig_wf_label="audio_wf",
#                                   sig_timestamps_label="audio_epoch_s",
#                                   sample_rate_hz_label="audio_sample_rate_nominal_hz")

import redpandas.redpd_tfr as rpd_tfr

rpd_tfr.tfr_bits_panda(df=df,
                       sig_wf_label="audio_wf",
                       sig_sample_rate_label="audio_sample_rate_nominal_hz",
                       order_number_input=12,  # Optional, default=3
                       tfr_type='stft')  # Optional, 'stft' or 'cwt, default='stft'

df = df.drop(index=2)
rpd_plot.plot_mesh_pandas(df=df,
                          mesh_time_label="tfr_time_s",
                          mesh_frequency_label="tfr_frequency_hz",
                          mesh_tfr_label="tfr_bits",
                          t0_sig_epoch_s=df["audio_epoch_s"][0][0],
                          frequency_hz_ymin=20,
                          sig_id_label=["Station 1", "Station 2"],
                          fig_title="STFT for Station 1 and Station 2")
plt.show()
