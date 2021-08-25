import pandas as pd
import numpy as np
from redvox.common.data_window import DataWindow
from redpandas.redpd_plot.wiggles import plot_wiggles_pandas
from redpandas.redpd_df import redpd_dataframe, export_df_to_parquet
from redpandas.redpd_tfr import tfr_bits_panda


if __name__ == '__main__':

    rdvx_data: DataWindow = DataWindow.from_json_file(base_dir="./test_data",
                                                      file_name="aud_bar_acc_mag_gyr_loc_soh_clock_sync")

    # rdvx_data: DataWindow = DataWindow(input_dir="/Users/meritxell/Desktop/skyfall_dummy_test")
    #
    # df = redpd_dataframe(input_dw=rdvx_data, sensor_labels=['audio',
    #                                                         # 'barometer',
    #                                                         'accelerometer',
    #                                                         # 'gyroscope',
    #                                                         # 'magnetometer',
    #                                                         # 'health',
    #                                                         # 'location',
    #                                                         # 'clock',
    #                                                         # 'synchronization']
    #                                                         ])
    #
    # print(df.columns)
    # tfr_bits_panda(df=df,
    #                sig_wf_label="audio_wf",
    #                sig_sample_rate_label="audio_sample_rate_nominal_hz",
    #                tfr_type="stft",
    #                new_column_tfr_bits="audio_tfr_bits",
    #                new_column_tfr_time_s="audio_tfr_time_s",
    #                new_column_tfr_frequency_hz="audio_tfr_frequency_hz")
    # tfr_bits_panda(df=df,
    #                sig_wf_label="accelerometer_wf_highpass",
    #                sig_sample_rate_label="accelerometer_sample_rate_hz",
    #                tfr_type="stft",
    #                new_column_tfr_bits="accelerometer_tfr_bits",
    #                new_column_tfr_time_s="accelerometer_tfr_time_s",
    #                new_column_tfr_frequency_hz="accelerometer_tfr_frequency_hz")
    # print(df.columns)
    # export_df_to_parquet(df, "/Users/meritxell/Desktop", "test_tfr_aud_acc")

    # fig = plot_wiggles_pandas(df=df,
    #                           sig_wf_label=['audio_wf', 'accelerometer_wf_raw'],
    #                           sig_timestamps_label=['audio_epoch_s', 'accelerometer_epoch_s'])

    # start_time = 0
    # end_time = 10
    # sample_rate_audio = 100
    # signal_time_audio = np.arange(start_time, end_time, 1/sample_rate_audio)
    # frequency = 3
    # amplitude = 1
    # sinewave_audio = amplitude * np.sin(2 * np.pi * frequency * signal_time_audio)
    #
    # # Create barometer
    # sample_rate_barometer = 31
    # signal_time_barometer = np.arange(start_time, end_time, 1/sample_rate_barometer)
    # sinewave_barometer_base = amplitude * np.sin(2 * np.pi * frequency * signal_time_barometer)
    # sinewave_barometer = sinewave_barometer_base.reshape((1, len(sinewave_barometer_base)))
    #
    # # Create accelerometer
    # sample_rate_acc = 30*2
    # signal_time_acc = np.arange(start_time, end_time, 1/(sample_rate_acc/3))
    #
    # length_for_signal = np.arange(start_time, end_time, 1/(sample_rate_acc))
    # sinewave_acc_base = amplitude * np.sin(2 * np.pi * frequency * length_for_signal)
    #
    # points_per_row = int(len(sinewave_acc_base)/3)
    #
    # sinewave_acc = sinewave_acc_base.reshape((3, points_per_row))
    #
    # # Create irregular df with one station with audio and barometer, other only audio
    # dict_to_df_irregular = {0: {"station_id": "1234567890",
    #                             "audio_sensor_name": "synch_audio",
    #                             "audio_sample_rate_nominal_hz": sample_rate_audio,
    #                             "audio_epoch_s": signal_time_audio,
    #                             "audio_wf": sinewave_audio,
    #                             "barometer_sensor_name": "synch_barometer",
    #                             "barometer_sample_rate_nominal_hz": sample_rate_barometer,
    #                             "barometer_epoch_s": signal_time_barometer,
    #                             "barometer_wf_raw": sinewave_barometer,
    #                             "accelerometer_epoch_s": signal_time_acc,
    #                             "accelerometer_wf_raw": sinewave_acc},
    #                         1: {"station_id": "2345678901",   # Add another station
    #                             "audio_sensor_name": "synch_audio",
    #                             "audio_sample_rate_nominal_hz": sample_rate_audio,
    #                             "audio_epoch_s": signal_time_audio,
    #                             "audio_wf": sinewave_audio,
    #                             "accelerometer_epoch_s": signal_time_acc,
    #                             "accelerometer_wf_raw": sinewave_acc},
    #                         2: {"station_id": "3456789012",
    #                             "audio_epoch_s": signal_time_audio,
    #                             "audio_wf": sinewave_audio}}
    #
    # df_data_irregular = pd.DataFrame(dict_to_df_irregular).T
    #
    # fig = plot_wiggles_pandas(df=df_data_irregular,
    #                           sig_wf_label=['audio_wf', 'accelerometer_wf_raw'],
    #                           sig_timestamps_label=['audio_epoch_s', 'accelerometer_epoch_s'],
    #                           station_id_str="1234567890")




