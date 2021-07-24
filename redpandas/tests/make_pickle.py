import pandas as pd
from redpandas.redpd_plot.redpd_plot import plot_wiggles_pandas
import numpy as np


if __name__ == '__main__':

    # rdvx_data: DataWindow = DataWindow.from_json_file(base_dir="./test_data",
    #                                                   file_name="aud_bar_acc_mag_gyr_loc_soh_clock_sync")
    #
    # df, _, _ = redpd_dataframe(input_dw_or_path=rdvx_data, show_raw_waveform_plots=False, sensor_labels=['audio',
    #                                                                                                      'barometer',
    #                                                                                                      'accelerometer',
    #                                                                                                      'gyroscope',
    #                                                                                                      'magnetometer',
    #                                                                                                      'health',
    #                                                                                                      'location',
    #                                                                                                      'clock',
    #                                                                                                      'synchronization'],
    #                            output_dir="./test_data", output_filename_pqt="aud_bar_acc_mag_gyr_loc_soh_clock_sync_df")
    #
    # fig = plot_wiggles_pandas(df=df,
    #                           sig_wf_label=['audio_wf', 'gyroscope_wf_raw'],
    #                           sig_timestamps_label=['audio_epoch_s', 'gyroscope_epoch_s'])

    start_time = 0
    end_time = 10
    sample_rate_audio = 100
    signal_time_audio = np.arange(start_time, end_time, 1/sample_rate_audio)
    frequency = 3
    amplitude = 1
    sinewave_audio = amplitude * np.sin(2 * np.pi * frequency * signal_time_audio)

    # Create barometer
    sample_rate_barometer = 31
    signal_time_barometer = np.arange(start_time, end_time, 1/sample_rate_barometer)
    sinewave_barometer_base = amplitude * np.sin(2 * np.pi * frequency * signal_time_barometer)
    sinewave_barometer = sinewave_barometer_base.reshape((1, len(sinewave_barometer_base)))

    # Create accelerometer
    sample_rate_acc = 30*2
    signal_time_acc = np.arange(start_time, end_time, 1/(sample_rate_acc/3))
    print(len(signal_time_acc))
    length_for_signal = np.arange(start_time, end_time, 1/(sample_rate_acc))
    sinewave_acc_base = amplitude * np.sin(2 * np.pi * frequency * length_for_signal)
    print(len(sinewave_acc_base))
    points_per_row = int(len(sinewave_acc_base)/3)
    print(points_per_row)
    sinewave_acc = sinewave_acc_base.reshape((3, points_per_row))

    # Create irregular df with one station with audio and barometer, other only audio
    dict_to_df_irregular = {0: {"station_id": "1234567890",
                                "audio_sensor_name": "synch_audio",
                                "audio_sample_rate_nominal_hz": sample_rate_audio,
                                "audio_epoch_s": signal_time_audio,
                                "audio_wf": sinewave_audio,
                                "barometer_sensor_name": "synch_barometer",
                                "barometer_sample_rate_nominal_hz": sample_rate_barometer,
                                "barometer_epoch_s": signal_time_barometer,
                                "barometer_wf_raw": sinewave_barometer},
                            1: {"station_id": "2345678901",   # Add another station
                                "audio_sensor_name": "synch_audio",
                                "audio_sample_rate_nominal_hz": sample_rate_audio,
                                "audio_epoch_s": signal_time_audio,
                                "audio_wf": sinewave_audio,
                                "accelerometer_epoch_s": signal_time_acc,
                                "accelerometer_wf_raw": sinewave_acc}}

    df_data_irregular = pd.DataFrame(dict_to_df_irregular).T
    print(len(signal_time_acc), len(sinewave_acc[0]))

    fig = plot_wiggles_pandas(df=df_data_irregular,
                              sig_wf_label=['audio_wf', 'accelerometer_wf_raw'],
                              sig_timestamps_label=['audio_epoch_s', 'accelerometer_epoch_s'],
                              custom_yticks=['Station 1 aud', 'Station 2 aud', 'Station 2 acc x', 'Station 2 acc y',
                                             'Station 2 acc z'])







