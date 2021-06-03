# Python libraries
import os.path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dtime

# RedVox RedPandas and related RedVox modules
from redvox.common.data_window import DataWindow
import redvox.common.date_time_utils as dt
import redpandas.redpd_preprocess as rpd_prep
import redpandas.redpd_build_station as rpd_build_sta
import redpandas.redpd_plot as rpd_plot
import redpandas.redpd_geospatial as rpd_geo
from redpandas.redpd_scales import METERS_TO_KM
from libquantum.plot_templates import plot_time_frequency_reps as pnl
from libquantum.spectra import stft_from_sig

# Configuration file
from examples.skyfall.skyfall_config import EVENT_NAME, INPUT_DIR, OUTPUT_DIR, EPISODE_START_EPOCH_S, \
    EPISODE_END_EPOCH_S, STATIONS, DW_FILE, use_datawindow, use_pickle, use_parquet, PD_PQT_FILE, SENSOR_LABEL, \
    ref_latitude_deg, ref_longitude_deg, ref_altitude_m, ref_epoch_s


band_order_Nth = 12
axes = ["X", "Y", "Z"]
verbosity = 1


if __name__ == "__main__":
    """
    RedVox RedPandas time-frequency representation of API900 data. Example: Skyfall.
    Last updated: 2 June 2021
    """

    print('Let the sky fall')

    # Label columns in dataframe
    station_label: str = "station_id"
    redvox_sdk_version_label: str = 'redvox_sdk_version'

    # Audio columns
    audio_data_label: str = "audio_wf"
    audio_epoch_s_label: str = "audio_epoch_s"
    audio_fs_label: str = "audio_sample_rate_nominal_hz"

    # Barometer columns
    barometer_data_raw_label: str = "barometer_wf_raw"
    barometer_data_highpass_label: str = "barometer_wf_highpass"
    barometer_epoch_s_label: str = "barometer_epoch_s"
    barometer_fs_label: str = "barometer_sample_rate_hz"

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

    # Load data options
    if use_datawindow is True or use_pickle is True:
        print("Initiating Conversion from RedVox DataWindow to RedVox RedPandas:")
        if use_datawindow:  # Option A: Create DataWindow object
            print("Constructing RedVox DataWindow Fast...", end=" ")
            rdvx_data = DataWindow(input_dir=INPUT_DIR,
                                   station_ids=STATIONS,
                                   start_datetime=dt.datetime_from_epoch_seconds_utc(EPISODE_START_EPOCH_S),
                                   end_datetime=dt.datetime_from_epoch_seconds_utc(EPISODE_END_EPOCH_S),
                                   apply_correction=True,
                                   structured_layout=True)

        else:  # Option B: Load pickle with DataWindow object. Assume compressed
            print("Unpickling existing compressed RedVox DataWindow with JSON...", end=" ")
            rdvx_data: DataWindow = DataWindow.from_json_file(base_dir=OUTPUT_DIR, file_name=DW_FILE)
        print(f"Done. RedVox SDK version: {rdvx_data.sdk_version}")

        # For option A or B, begin RedPandas
        print("\nInitiating RedVox Redpandas:")
        df_skyfall_data = pd.DataFrame([rpd_build_sta.station_to_dict_from_dw(station=station,
                                                                              sdk_version=rdvx_data.sdk_version,
                                                                              sensor_labels=SENSOR_LABEL)
                                        for station in rdvx_data.stations])
        df_skyfall_data.sort_values(by="station_id", ignore_index=True, inplace=True)

    elif use_parquet:  # Option C: Open dataframe from parquet file
        print("Loading existing RedPandas Parquet...", end=" ")
        df_skyfall_data = pd.read_parquet(os.path.join(OUTPUT_DIR, PD_PQT_FILE))
        print(f"Done. RedVox SDK version: {df_skyfall_data[redvox_sdk_version_label][0]}")

    else:
        print('\nNo data loading method selected. '
              'Check that use_datawindow, use_pickle, or use_parquet in the Skyfall configuration file are set to True.')
        exit()

    # PLOTTING
    print("\nInitiating time-frequency representation of Skyfall:")
    for station in df_skyfall_data.index:
        station_id_str = df_skyfall_data[station_label][station]  # Get the station id

        if audio_data_label and audio_fs_label in df_skyfall_data.columns:
            print('mic_sample_rate_hz: ', df_skyfall_data[audio_fs_label][station])
            print('mic_epoch_s_0: ', df_skyfall_data[audio_epoch_s_label][station][0])

            # Frame to mic start and end and plot
            event_reference_time_epoch_s = df_skyfall_data[audio_epoch_s_label][station][0]
            print(event_reference_time_epoch_s)

        if barometer_data_raw_label and barometer_data_highpass_label and barometer_fs_label in df_skyfall_data.columns:
            if use_parquet is True and use_datawindow is False and use_pickle is False:
                # Reshape wf columns
                rpd_prep.df_column_unflatten(df=df_skyfall_data,
                                             col_wf_label=barometer_data_raw_label,
                                             col_ndim_label=barometer_data_raw_label + "_ndim")

                rpd_prep.df_column_unflatten(df=df_skyfall_data,
                                             col_wf_label=barometer_data_highpass_label,
                                             col_ndim_label=barometer_data_highpass_label + "_ndim")

            print('barometer_sample_rate_hz:', df_skyfall_data[barometer_fs_label][station])
            print('barometer_epoch_s_0:', df_skyfall_data[barometer_epoch_s_label][station][0])

            barometer_height_m = \
                rpd_geo.bounder_model_height_from_pressure(df_skyfall_data[barometer_data_raw_label][station][0])
            baro_height_from_bounder_km = barometer_height_m * METERS_TO_KM

            baro_raw_wf = df_skyfall_data[barometer_data_raw_label][station][0]
            baro_raw_stft, baro_raw_stft_bits, baro_raw_time_stft_s, baro_raw_frequency_stft_hz = \
                stft_from_sig(baro_raw_wf,
                              df_skyfall_data[barometer_fs_label][station],
                              band_order_Nth)
            pnl.plot_wf_mesh_vert(redvox_id=station_id_str,
                                  wf_panel_2_sig=baro_raw_wf,
                                  wf_panel_2_time=df_skyfall_data[barometer_epoch_s_label][station],
                                  mesh_time=baro_raw_time_stft_s,
                                  mesh_frequency=baro_raw_frequency_stft_hz,
                                  mesh_panel_0_tfr=baro_raw_stft_bits,
                                  figure_title=EVENT_NAME + ": Baro raw, STFT and waveform",
                                  start_time_epoch=event_reference_time_epoch_s)

            baro_hp_wf = df_skyfall_data[barometer_data_highpass_label][station][0]
            baro_hp_stft, baro_hp_stft_bits, baro_hp_time_stft_s, baro_hp_frequency_stft_hz = \
                stft_from_sig(baro_hp_wf,
                              df_skyfall_data[barometer_fs_label][station],
                              band_order_Nth)
            pnl.plot_wf_mesh_vert(redvox_id=station_id_str,
                                  wf_panel_2_sig=baro_hp_wf,
                                  wf_panel_2_time=df_skyfall_data[barometer_epoch_s_label][station],
                                  mesh_time=baro_hp_time_stft_s,
                                  mesh_frequency=baro_hp_frequency_stft_hz,
                                  mesh_panel_0_tfr=baro_hp_stft_bits,
                                  figure_title=EVENT_NAME + ": Baro highpass, STFT and waveform",
                                  start_time_epoch=event_reference_time_epoch_s)

        # Repeat here
        if accelerometer_data_raw_label and accelerometer_fs_label and accelerometer_data_highpass_label \
                in df_skyfall_data.columns:
            if use_parquet is True and use_datawindow is False and use_pickle is False:
                # Reshape wf columns
                rpd_prep.df_column_unflatten(df=df_skyfall_data,
                                             col_wf_label=accelerometer_data_raw_label,
                                             col_ndim_label=accelerometer_data_raw_label + "_ndim")
                rpd_prep.df_column_unflatten(df=df_skyfall_data,
                                             col_wf_label=accelerometer_data_highpass_label,
                                             col_ndim_label=accelerometer_data_highpass_label + "_ndim")

            print('accelerometer_sample_rate_hz:', df_skyfall_data[accelerometer_fs_label][station])
            print('accelerometer_epoch_s_0:', df_skyfall_data[accelerometer_epoch_s_label][station][0],
                  df_skyfall_data[accelerometer_epoch_s_label][station][-1])

            if verbosity > 1:
                # Plot 3c acceleration raw waveforms
                pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                       wf_panel_2_sig=df_skyfall_data[accelerometer_data_raw_label][station][2],
                                       wf_panel_2_time=df_skyfall_data[accelerometer_epoch_s_label][station],
                                       wf_panel_1_sig=df_skyfall_data[accelerometer_data_raw_label][station][1],
                                       wf_panel_1_time=df_skyfall_data[accelerometer_epoch_s_label][station],
                                       wf_panel_0_sig=df_skyfall_data[accelerometer_data_raw_label][station][0],
                                       wf_panel_0_time=df_skyfall_data[accelerometer_epoch_s_label][station],
                                       start_time_epoch=event_reference_time_epoch_s,
                                       wf_panel_2_units="Acc Z, m/$s^2$",
                                       wf_panel_1_units="Acc Y, m/$s^2$",
                                       wf_panel_0_units="Acc X, m/$s^2$",
                                       figure_title=EVENT_NAME + ": Accelerometer raw",
                                       figure_title_show=True,
                                       label_panel_show=False,
                                       labels_fontweight='bold')

                # Plot 3c acceleration highpass waveforms
                pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                       wf_panel_2_sig=df_skyfall_data[accelerometer_data_highpass_label][station][2],
                                       wf_panel_2_time=df_skyfall_data[accelerometer_epoch_s_label][station],
                                       wf_panel_1_sig=df_skyfall_data[accelerometer_data_highpass_label][station][1],
                                       wf_panel_1_time=df_skyfall_data[accelerometer_epoch_s_label][station],
                                       wf_panel_0_sig=df_skyfall_data[accelerometer_data_highpass_label][station][0],
                                       wf_panel_0_time=df_skyfall_data[accelerometer_epoch_s_label][station],
                                       start_time_epoch=event_reference_time_epoch_s,
                                       wf_panel_2_units="Acc Z, m/$s^2$",
                                       wf_panel_1_units="Acc Y, m/$s^2$",
                                       wf_panel_0_units="Acc X, m/$s^2$",
                                       figure_title=EVENT_NAME + ": Accelerometer highpass",
                                       figure_title_show=True,
                                       label_panel_show=False,
                                       labels_fontweight='bold')

            for ax_n in range(3):
                acc_raw_wf = df_skyfall_data[accelerometer_data_raw_label][station][ax_n]
                acc_raw_stft, acc_raw_stft_bits, acc_raw_time_stft_s, acc_raw_frequency_stft_hz = \
                    stft_from_sig(acc_raw_wf,
                                  df_skyfall_data[accelerometer_fs_label][station],
                                  band_order_Nth)
                pnl.plot_wf_mesh_vert(redvox_id=station_id_str,
                                      wf_panel_2_sig=acc_raw_wf,
                                      wf_panel_2_time=df_skyfall_data[accelerometer_epoch_s_label][station],
                                      mesh_time=acc_raw_time_stft_s,
                                      mesh_frequency=acc_raw_frequency_stft_hz,
                                      mesh_panel_0_tfr=acc_raw_stft_bits,
                                      figure_title=EVENT_NAME + ": Acc " + axes[ax_n] + " raw, STFT and waveform",
                                      start_time_epoch=event_reference_time_epoch_s)

                acc_hp_wf = df_skyfall_data[accelerometer_data_highpass_label][station][ax_n]
                acc_hp_stft, acc_hp_stft_bits, acc_hp_time_stft_s, acc_hp_frequency_stft_hz = \
                    stft_from_sig(acc_hp_wf,
                                  df_skyfall_data[accelerometer_fs_label][station],
                                  band_order_Nth)
                pnl.plot_wf_mesh_vert(redvox_id=station_id_str,
                                      wf_panel_2_sig=acc_hp_wf,
                                      wf_panel_2_time=df_skyfall_data[accelerometer_epoch_s_label][station],
                                      mesh_time=acc_hp_time_stft_s,
                                      mesh_frequency=acc_hp_frequency_stft_hz,
                                      mesh_panel_0_tfr=acc_hp_stft_bits,
                                      figure_title=EVENT_NAME + ": Acc " + axes[ax_n] + " highpass, STFT and waveform",
                                      start_time_epoch=event_reference_time_epoch_s)

        if gyroscope_data_raw_label and gyroscope_fs_label and gyroscope_data_highpass_label \
                in df_skyfall_data.columns:

            if use_parquet is True and use_datawindow is False and use_pickle is False:
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

            if verbosity > 1:
                # Plot 3c raw gyroscope waveforms
                pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                       wf_panel_2_sig=df_skyfall_data[gyroscope_data_raw_label][station][2],
                                       wf_panel_2_time=df_skyfall_data[gyroscope_epoch_s_label][station],
                                       wf_panel_1_sig=df_skyfall_data[gyroscope_data_raw_label][station][1],
                                       wf_panel_1_time=df_skyfall_data[gyroscope_epoch_s_label][station],
                                       wf_panel_0_sig=df_skyfall_data[gyroscope_data_raw_label][station][0],
                                       wf_panel_0_time=df_skyfall_data[gyroscope_epoch_s_label][station],
                                       start_time_epoch=event_reference_time_epoch_s,
                                       wf_panel_2_units="Gyr Z, rad/s$",
                                       wf_panel_1_units="Gyr Y, rad/s$",
                                       wf_panel_0_units="Gyr X, rad/s$",
                                       figure_title=EVENT_NAME + ": Gyroscope raw",
                                       figure_title_show=True,
                                       label_panel_show=False,
                                       labels_fontweight='bold')

                # Plot 3c gyroscope highpass waveforms
                pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                       wf_panel_2_sig=df_skyfall_data[gyroscope_data_highpass_label][station][2],
                                       wf_panel_2_time=df_skyfall_data[gyroscope_epoch_s_label][station],
                                       wf_panel_1_sig=df_skyfall_data[gyroscope_data_highpass_label][station][1],
                                       wf_panel_1_time=df_skyfall_data[gyroscope_epoch_s_label][station],
                                       wf_panel_0_sig=df_skyfall_data[gyroscope_data_highpass_label][station][0],
                                       wf_panel_0_time=df_skyfall_data[gyroscope_epoch_s_label][station],
                                       start_time_epoch=event_reference_time_epoch_s,
                                       wf_panel_2_units="Gyr Z, rad/s$",
                                       wf_panel_1_units="Gyr Y, rad/s$",
                                       wf_panel_0_units="Gyr X, rad/s$",
                                       figure_title=EVENT_NAME + ": Gyroscope highpass",
                                       figure_title_show=True,
                                       label_panel_show=False,
                                       labels_fontweight='bold')

            for ax_n in range(3):
                gyr_raw_wf = df_skyfall_data[gyroscope_data_raw_label][station][ax_n]
                gyr_raw_stft, gyr_raw_stft_bits, gyr_raw_time_stft_s, gyr_raw_frequency_stft_hz = \
                    stft_from_sig(gyr_raw_wf,
                                  df_skyfall_data[gyroscope_fs_label][station],
                                  band_order_Nth)
                pnl.plot_wf_mesh_vert(redvox_id=station_id_str,
                                      wf_panel_2_sig=gyr_raw_wf,
                                      wf_panel_2_time=df_skyfall_data[gyroscope_epoch_s_label][station],
                                      mesh_time=gyr_raw_time_stft_s,
                                      mesh_frequency=gyr_raw_frequency_stft_hz,
                                      mesh_panel_0_tfr=gyr_raw_stft_bits,
                                      figure_title=EVENT_NAME + ": Gyr " + axes[ax_n] + " raw, STFT and waveform",
                                      start_time_epoch=event_reference_time_epoch_s)

                gyr_hp_wf = df_skyfall_data[gyroscope_data_highpass_label][station][ax_n]
                gyr_hp_stft, gyr_hp_stft_bits, gyr_hp_time_stft_s, gyr_hp_frequency_stft_hz = \
                    stft_from_sig(gyr_hp_wf,
                                  df_skyfall_data[gyroscope_fs_label][station],
                                  band_order_Nth)
                pnl.plot_wf_mesh_vert(redvox_id=station_id_str,
                                      wf_panel_2_sig=gyr_hp_wf,
                                      wf_panel_2_time=df_skyfall_data[gyroscope_epoch_s_label][station],
                                      mesh_time=gyr_hp_time_stft_s,
                                      mesh_frequency=gyr_hp_frequency_stft_hz,
                                      mesh_panel_0_tfr=gyr_hp_stft_bits,
                                      figure_title=EVENT_NAME + ": Gyr " + axes[ax_n] + " highpass, STFT and waveform",
                                      start_time_epoch=event_reference_time_epoch_s)

        if magnetometer_data_raw_label and magnetometer_fs_label and magnetometer_data_highpass_label \
                in df_skyfall_data.columns:
            if use_parquet is True and use_datawindow is False and use_pickle is False:
                # Reshape wf columns
                rpd_prep.df_column_unflatten(df=df_skyfall_data,
                                             col_wf_label=magnetometer_data_raw_label,
                                             col_ndim_label=magnetometer_data_raw_label + "_ndim")
                rpd_prep.df_column_unflatten(df=df_skyfall_data,
                                             col_wf_label=magnetometer_data_highpass_label,
                                             col_ndim_label=magnetometer_data_highpass_label + "_ndim")

            print('magnetometer_sample_rate_hz:', df_skyfall_data[magnetometer_fs_label][station])
            print('magnetometer_epoch_s_0:', df_skyfall_data[magnetometer_epoch_s_label][station][0],
                  df_skyfall_data[magnetometer_epoch_s_label][station][-1])

            if verbosity > 1:
                # Plot 3c raw magnetometer waveforms
                pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                       wf_panel_2_sig=df_skyfall_data[magnetometer_data_raw_label][station][2],
                                       wf_panel_2_time=df_skyfall_data[magnetometer_epoch_s_label][station],
                                       wf_panel_1_sig=df_skyfall_data[magnetometer_data_raw_label][station][1],
                                       wf_panel_1_time=df_skyfall_data[magnetometer_epoch_s_label][station],
                                       wf_panel_0_sig=df_skyfall_data[magnetometer_data_raw_label][station][0],
                                       wf_panel_0_time=df_skyfall_data[magnetometer_epoch_s_label][station],
                                       start_time_epoch=event_reference_time_epoch_s,
                                       wf_panel_2_units="Mag Z, $\mu$T",
                                       wf_panel_1_units="Mag Y, $\mu$T",
                                       wf_panel_0_units="Mag X, $\mu$T",
                                       figure_title=EVENT_NAME + ": Magnetometer raw",
                                       figure_title_show=True,
                                       label_panel_show=False,
                                       labels_fontweight='bold')

                # Plot 3c magnetometer highpass waveforms
                pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                       wf_panel_2_sig=df_skyfall_data[magnetometer_data_highpass_label][station][2],
                                       wf_panel_2_time=df_skyfall_data[magnetometer_epoch_s_label][station],
                                       wf_panel_1_sig=df_skyfall_data[magnetometer_data_highpass_label][station][1],
                                       wf_panel_1_time=df_skyfall_data[magnetometer_epoch_s_label][station],
                                       wf_panel_0_sig=df_skyfall_data[magnetometer_data_highpass_label][station][0],
                                       wf_panel_0_time=df_skyfall_data[magnetometer_epoch_s_label][station],
                                       start_time_epoch=event_reference_time_epoch_s,
                                       wf_panel_2_units="Mag Z, $\mu$T",
                                       wf_panel_1_units="Mag Y, $\mu$T",
                                       wf_panel_0_units="Mag X, $\mu$T",
                                       figure_title=EVENT_NAME + ": Magnetometer highpass",
                                       figure_title_show=True,
                                       label_panel_show=False,
                                       labels_fontweight='bold')

            for ax_n in range(3):
                mag_raw_wf = df_skyfall_data[magnetometer_data_raw_label][station][ax_n]
                mag_raw_stft, mag_raw_stft_bits, mag_raw_time_stft_s, mag_raw_frequency_stft_hz = \
                    stft_from_sig(mag_raw_wf,
                                  df_skyfall_data[magnetometer_fs_label][station],
                                  band_order_Nth)
                pnl.plot_wf_mesh_vert(redvox_id=station_id_str,
                                      wf_panel_2_sig=mag_raw_wf,
                                      wf_panel_2_time=df_skyfall_data[magnetometer_epoch_s_label][station],
                                      mesh_time=mag_raw_time_stft_s,
                                      mesh_frequency=mag_raw_frequency_stft_hz,
                                      mesh_panel_0_tfr=mag_raw_stft_bits,
                                      figure_title=EVENT_NAME + ": Mag " + axes[ax_n] + " raw, STFT and waveform",
                                      start_time_epoch=event_reference_time_epoch_s)

                mag_hp_wf = df_skyfall_data[magnetometer_data_highpass_label][station][ax_n]
                mag_hp_stft, mag_hp_stft_bits, mag_hp_time_stft_s, mag_hp_frequency_stft_hz = \
                    stft_from_sig(mag_hp_wf,
                                  df_skyfall_data[magnetometer_fs_label][station],
                                  band_order_Nth)
                pnl.plot_wf_mesh_vert(redvox_id=station_id_str,
                                      wf_panel_2_sig=mag_hp_wf,
                                      wf_panel_2_time=df_skyfall_data[magnetometer_epoch_s_label][station],
                                      mesh_time=mag_hp_time_stft_s,
                                      mesh_frequency=mag_hp_frequency_stft_hz,
                                      mesh_panel_0_tfr=mag_hp_stft_bits,
                                      figure_title=EVENT_NAME + ": Mag " + axes[ax_n] + " highpass, STFT and waveform",
                                      start_time_epoch=event_reference_time_epoch_s)

        # Plot sensor wiggles
        sensor_column_label_list = [audio_data_label, barometer_data_highpass_label,
                                    accelerometer_data_highpass_label, gyroscope_data_highpass_label,
                                    magnetometer_data_highpass_label]

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
                                            fig_title_show=False,
                                            fig_title='sensor waveforms',
                                            wf_color='midnightblue',
                                            sensor_yticks_label_list=sensor_ticklabels_list)

        plt.show()
