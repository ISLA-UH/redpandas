# todo: address possible invalid values in building plots section
# Python libraries
import os.path
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dtime

# RedVox RedPandas and related RedVox modules
from redvox.common.data_window import DataWindow
import redpandas.redpd_datawin as rpd_dw
import redpandas.redpd_preprocess as rpd_prep
import redpandas.redpd_build_station as rpd_build_sta
import redpandas.redpd_plot as rpd_plot
import redpandas.redpd_geospatial as rpd_geo
from redpandas.redpd_scales import METERS_TO_KM
from libquantum.plot_templates import plot_time_frequency_reps as pnl

# Configuration files
from redpandas.redpd_config import DataLoadMethod
from examples.skyfall.skyfall_config_file import skyfall_config, OTHER_INPUT_PATH, OTHER_INPUT_FILE, OTHER_PD_PQT_FILE

if __name__ == "__main__":
    """
    Station sonification
    Load from parquet
    """

    redvox_sdk_version_label: str = 'redvox_sdk_version'
    print("Loading existing RedPandas Parquet...", end=" ")
    df_skyfall_data = pd.read_parquet(os.path.join(skyfall_config.output_dir, skyfall_config.pd_pqt_file))

    print(f"Done. RedVox SDK version: {df_skyfall_data[redvox_sdk_version_label][0]}")

    print(df_skyfall_data.columns)

    # Label columns in dataframe
    station_label: str = "station_id"

    # Audio columns
    audio_data_label: str = "audio_wf"
    audio_epoch_s_label: str = "audio_epoch_s"
    audio_fs_label: str = "audio_sample_rate_nominal_hz"

    # All the sensor channels to be ensonified have 100 Hz highpass

    # Barometer columns
    barometer_data_highpass_label: str = "barometer_wf_highpass"
    barometer_epoch_s_label: str = "barometer_epoch_s"
    barometer_fs_label: str = "barometer_sample_rate_hz"

    # Accelerometer columns
    accelerometer_data_highpass_label: str = "accelerometer_wf_highpass"
    accelerometer_epoch_s_label: str = "accelerometer_epoch_s"
    accelerometer_fs_label: str = "accelerometer_sample_rate_hz"

    # Gyroscope columns
    gyroscope_data_highpass_label: str = "gyroscope_wf_highpass"
    gyroscope_epoch_s_label: str = "gyroscope_epoch_s"
    gyroscope_fs_label: str = "gyroscope_sample_rate_hz"

    # Magnetometer columns
    magnetometer_data_highpass_label: str = "magnetometer_wf_highpass"
    magnetometer_epoch_s_label: str = "magnetometer_epoch_s"
    magnetometer_fs_label: str = "magnetometer_sample_rate_hz"

    # Must unflatten barometer and 3C from parquet
    # Unflattened numpy arrays inherit the same column label
    rpd_prep.df_column_unflatten(df=df_skyfall_data,
                                 col_wf_label=barometer_data_highpass_label,
                                 col_ndim_label=barometer_data_highpass_label + "_ndim")
    rpd_prep.df_column_unflatten(df=df_skyfall_data,
                                 col_wf_label=accelerometer_data_highpass_label,
                                 col_ndim_label=accelerometer_data_highpass_label + "_ndim")
    rpd_prep.df_column_unflatten(df=df_skyfall_data,
                                 col_wf_label=gyroscope_data_highpass_label,
                                 col_ndim_label=gyroscope_data_highpass_label + "_ndim")
    rpd_prep.df_column_unflatten(df=df_skyfall_data,
                                 col_wf_label=magnetometer_data_highpass_label,
                                 col_ndim_label=magnetometer_data_highpass_label + "_ndim")

    # Loop over a numbered list
    sensor_column_label_list = [audio_data_label, barometer_data_highpass_label,
                                accelerometer_data_highpass_label, gyroscope_data_highpass_label,
                                magnetometer_data_highpass_label]
    sensor_channels = [1, 1, 3, 3, 3]
    sensor_fs_label_list = [audio_fs_label, barometer_fs_label,
                            accelerometer_fs_label, gyroscope_fs_label,
                            magnetometer_fs_label]

    sensor_number_total = len(sensor_column_label_list)
    sensor_channels_total = sum(sensor_channels)
    print("Number of sensors to ensonify = ", sensor_number_total)
    print("Number of channels to ensonify = ", sensor_channels_total)

    # TODO: loop over number of sensors, then number of channels

    # Plot sensor wiggles
    sensor_epoch_column_label_list = [audio_epoch_s_label, barometer_epoch_s_label,
                                      accelerometer_epoch_s_label, gyroscope_epoch_s_label,
                                      magnetometer_epoch_s_label]

    sensor_ticklabels_list = ['Audio', 'Bar hp', 'Acc X hp', 'Acc Y hp',
                              'Acc Z hp', 'Gyr X hp', 'Gyr Y hp', 'Gyr Z hp',
                              'Mag X hp', 'Mag Y hp', 'Mag Z hp']

    rpd_plot.plot_sensor_wiggles_pandas(df=df_skyfall_data,
                                        station_id_str='1637610021',
                                        sensor_wf_label_list=sensor_column_label_list,
                                        sensor_timestamps_label_list=sensor_epoch_column_label_list,
                                        sig_id_label='station_id',
                                        x_label='Time (s)',
                                        y_label='Sensor',
                                        fig_title_show=True,
                                        fig_title='sensor waveforms',
                                        wf_color='midnightblue',
                                        sensor_yticks_label_list=sensor_ticklabels_list)

    plt.show()