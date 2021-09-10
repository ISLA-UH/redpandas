# RedVox RedPandas and related RedVox modules
import redpandas.redpd_ensonify as rpd_sound
import redpandas.redpd_plot.wiggles as rpd_plot

# Configuration files
import examples.skyfall.lib.skyfall_dw as sf_dw
from examples.skyfall.skyfall_config_file import skyfall_config


def main():
    """
    Station sonification
    Load from datawindow
    """
    # Refine loading checks; need hp and strings
    redvox_sdk_version_label: str = 'redvox_sdk_version'
    print("Create RedVox DataWindow")
    df_skyfall_data = sf_dw.dw_main(skyfall_config.tdr_load_method)
    print(f"Done. RedVox SDK version: {df_skyfall_data[redvox_sdk_version_label][0]}")

    # print(df_skyfall_data.columns)

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

    # Loop over a numbered list
    sensor_column_label_list = [audio_data_label, barometer_data_highpass_label,
                                accelerometer_data_highpass_label, gyroscope_data_highpass_label,
                                magnetometer_data_highpass_label]

    sensor_fs_label_list = [audio_fs_label, barometer_fs_label,
                            accelerometer_fs_label, gyroscope_fs_label,
                            magnetometer_fs_label]

    sensor_name_key_list = ['Aud', 'Bar',
                            'AccX', 'AccY', 'AccZ',
                            'GyrX', 'GyrY', 'GyrZ',
                            'MagX', 'MagY', 'MagZ']

    rpd_sound.ensonify_sensors_pandas(df=df_skyfall_data,
                                      sig_id_label='station_id',
                                      sig_sample_rate_label_list=sensor_fs_label_list,
                                      sensor_column_label_list=sensor_column_label_list,
                                      wav_sample_rate_hz=192000.,
                                      output_wav_directory=skyfall_config.input_dir,
                                      output_wav_filename='skyfall',
                                      sensor_name_list=sensor_name_key_list)

    # Plot sensor wiggles, need epoch time
    sensor_epoch_column_label_list = [audio_epoch_s_label, barometer_epoch_s_label,
                                      accelerometer_epoch_s_label, gyroscope_epoch_s_label,
                                      magnetometer_epoch_s_label]

    rpd_plot.plot_wiggles_pandas(df=df_skyfall_data,
                                 station_id_str='1637610021',
                                 sig_wf_label=sensor_column_label_list,
                                 sig_timestamps_label=sensor_epoch_column_label_list,
                                 sig_id_label='station_id',
                                 fig_title_show=True,
                                 fig_title='sensor waveforms')


if __name__ == "__main__":
    main()
