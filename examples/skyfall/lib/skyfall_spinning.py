# todo: address possible invalid values in building plots section
# Python libraries
import os.path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# RedVox RedPandas and related RedVox modules
from redvox.common.data_window import DataWindow
import redpandas.redpd_datawin as rpd_dw
import redpandas.redpd_preprocess as rpd_prep
import redpandas.redpd_build_station as rpd_build_sta
from libquantum.plot_templates import plot_time_frequency_reps as pnl

# Configuration files
from redpandas.redpd_config import DataLoadMethod
from examples.skyfall.skyfall_config_file import skyfall_config, OTHER_INPUT_PATH, OTHER_PD_PQT_FILE


def main():
    """
    RedVox RedPandas time-domain representation of API900 data. Example: Skyfall.
    Last updated: 18 June 2021
    """
    print('Let the sky fall')

    # Label columns in dataframe
    station_label: str = "station_id"
    redvox_sdk_version_label: str = 'redvox_sdk_version'

    # Audio columns
    audio_data_label: str = "audio_wf"
    audio_epoch_s_label: str = "audio_epoch_s"
    audio_fs_label: str = "audio_sample_rate_nominal_hz"

    # Accelerometer columns
    accelerometer_data_raw_label: str = "accelerometer_wf_raw"
    accelerometer_data_highpass_label: str = "accelerometer_wf_highpass"
    accelerometer_epoch_s_label: str = "accelerometer_epoch_s"
    accelerometer_fs_label: str = "accelerometer_sample_rate_hz"

    # Gyroscope columns
    gyroscope_data_raw_label: str = "gyroscope_wf_raw"
    gyroscope_data_highpass_label: str = "gyroscope_wf_highpass"
    gyroscope_epoch_s_label: str = "gyroscope_epoch_s"
    gyroscope_fs_label: str = "gyroscope_sample_rate_hz"

    # Magnetometer columns
    magnetometer_data_raw_label: str = "magnetometer_wf_raw"
    magnetometer_data_highpass_label: str = "magnetometer_wf_highpass"
    magnetometer_epoch_s_label: str = "magnetometer_epoch_s"
    magnetometer_fs_label: str = "magnetometer_sample_rate_hz"

    # Load parquet with bounder data fields
    # TODO: clean up
    bounder_loc = pd.read_parquet(os.path.join(OTHER_INPUT_PATH, OTHER_PD_PQT_FILE))

    # Load data options
    # if use_datawindow_tdr is True or use_pickle_tdr is True:
    if skyfall_config.tdr_load_method == DataLoadMethod.DATAWINDOW or \
            skyfall_config.tdr_load_method == DataLoadMethod.PICKLE:
        print("Initiating Conversion from RedVox DataWindow to RedVox RedPandas:")
        if skyfall_config.tdr_load_method == DataLoadMethod.DATAWINDOW:  # Option A: Create DataWindow object
            print("Constructing RedVox DataWindow ...", end=" ")
            rdvx_data = rpd_dw.dw_from_redpd_config(config=skyfall_config)

        else:  # Option B: Load pickle with DataWindow object. Assume compressed
            print("Unpickling existing compressed RedVox DataWindow with JSON...", end=" ")
            rdvx_data: DataWindow = DataWindow.from_json_file(base_dir=skyfall_config.output_dir,
                                                              file_name=skyfall_config.output_filename_pkl_pqt)
        print(f"Done. RedVox SDK version: {rdvx_data.sdk_version}")

        # For option A or B, begin RedPandas
        print("\nInitiating RedVox Redpandas:")
        df_skyfall_data = pd.DataFrame([rpd_build_sta.station_to_dict_from_dw(station=station,
                                                                              sdk_version=rdvx_data.sdk_version,
                                                                              sensor_labels=skyfall_config.sensor_labels)
                                        for station in rdvx_data.stations])
        df_skyfall_data.sort_values(by="station_id", ignore_index=True, inplace=True)

    elif skyfall_config.tdr_load_method == DataLoadMethod.PARQUET:  # Option C: Open dataframe from parquet file
        print("Loading existing RedPandas Parquet...", end=" ")
        df_skyfall_data = pd.read_parquet(os.path.join(skyfall_config.output_dir, skyfall_config.pd_pqt_file))
        print(f"Done. RedVox SDK version: {df_skyfall_data[redvox_sdk_version_label][0]}")

    else:
        print('\nNo data loading method selected.')
        exit()

    # Start of building plots
    print("\nInitiating time-domain representation of Skyfall:")
    for station in df_skyfall_data.index:
        station_id_str = df_skyfall_data[station_label][station]  # Get the station id

        if audio_data_label and audio_fs_label in df_skyfall_data.columns:

            print('mic_sample_rate_hz: ', df_skyfall_data[audio_fs_label][station])
            print('mic_epoch_s_0: ', df_skyfall_data[audio_epoch_s_label][station][0])

            # Frame to mic start and end and plot
            event_reference_time_epoch_s = df_skyfall_data[audio_epoch_s_label][station][0]

        if gyroscope_data_raw_label and gyroscope_fs_label and gyroscope_data_highpass_label \
                in df_skyfall_data.columns:

            if skyfall_config.tdr_load_method == DataLoadMethod.PARQUET:
                # Reshape wf columns
                rpd_prep.df_column_unflatten(df=df_skyfall_data,
                                             col_wf_label=gyroscope_data_raw_label,
                                             col_ndim_label=gyroscope_data_raw_label + "_ndim")

                rpd_prep.df_column_unflatten(df=df_skyfall_data,
                                             col_wf_label=gyroscope_data_highpass_label,
                                             col_ndim_label=gyroscope_data_highpass_label + "_ndim")

            print('gyroscope_sample_rate_hz:', df_skyfall_data[gyroscope_fs_label][station])
            print('gyroscope_epoch_s_0:', df_skyfall_data[gyroscope_epoch_s_label][station][0],
                  df_skyfall_data[gyroscope_epoch_s_label][station][-1])
            print('gyroscope max rotation rate, rad/s:',
                  np.max(df_skyfall_data[gyroscope_data_raw_label][station][2]))
            print('gyroscope max rotation rate, Hz:',
                  np.max(df_skyfall_data[gyroscope_data_raw_label][station][2] / (2*np.pi)))
            # Plot 3c raw gyroscope waveforms
            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=df_skyfall_data[gyroscope_data_raw_label][station][2] / (2*np.pi),
                                   wf_panel_2_time=df_skyfall_data[gyroscope_epoch_s_label][station],
                                   wf_panel_1_sig=df_skyfall_data[gyroscope_data_raw_label][station][1] / (2*np.pi),
                                   wf_panel_1_time=df_skyfall_data[gyroscope_epoch_s_label][station],
                                   wf_panel_0_sig=df_skyfall_data[gyroscope_data_raw_label][station][0] / (2*np.pi),
                                   wf_panel_0_time=df_skyfall_data[gyroscope_epoch_s_label][station],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Gyr Z, rotation/s",
                                   wf_panel_1_units="Gyr Y, rotation/s",
                                   wf_panel_0_units="Gyr X, rotation/s",
                                   figure_title=skyfall_config.event_name + ": Gyroscope raw",
                                   figure_title_show=False,
                                   label_panel_show=True,  # for press
                                   labels_fontweight='bold')

        # TODO: Add magnetometer spectrogram to illustrate match of frequency information

        plt.show()


if __name__ == "__main__":
    main()
