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
import redpandas.redpd_tfr as rpd_tfr
from redpandas.redpd_scales import METERS_TO_KM
from libquantum.plot_templates import plot_time_frequency_reps as pnl
from libquantum.spectra import stft_from_sig

# Configuration file
from examples.skyfall.skyfall_config import EVENT_NAME, INPUT_DIR, OUTPUT_DIR, EPISODE_START_EPOCH_S, \
    EPISODE_END_EPOCH_S, STATIONS, DW_FILE, use_datawindow, use_pickle, use_parquet, PD_PQT_FILE, SENSOR_LABEL, \
    ref_latitude_deg, ref_longitude_deg, ref_altitude_m, ref_epoch_s

band_order_Nth = 12
axes = ["X", "Y", "Z"]
verbosity = 2
# verbosity levels:
# 1: wiggle plots
# 2: wiggle plots, individual channel plots
# 3: wiggle plots, individual channel plots, 3 channel sensor plots
tfr_type = 'stft'
# tfr_type = 'cwt' # very slow
plot_raw_data = False  # raw plots currently run but are not updated

# TODO: optimize color dynamic range
# TODO: cleaner code for TFR wiggle plot

if __name__ == "__main__":
    """
    RedVox RedPandas time-frequency representation of API900 data. Example: Skyfall.
    Last updated: 4 June 2021
    """

    print('Let the sky fall')

    # Label columns in dataframe
    station_label: str = "station_id"
    redvox_sdk_version_label: str = 'redvox_sdk_version'

    # Audio columns
    audio_data_label: str = "audio_wf"
    audio_epoch_s_label: str = "audio_epoch_s"
    audio_fs_label: str = "audio_sample_rate_nominal_hz"
    audio_tfr_bits_label: str = "audio_tfr_bits"
    audio_tfr_frequency_hz_label: str = "audio_tfr_frequency_hz"
    audio_tfr_time_s_label: str = "audio_tfr_time_s"

    # Barometer columns
    barometer_data_raw_label: str = "barometer_wf_raw"
    barometer_data_highpass_label: str = "barometer_wf_highpass"
    barometer_epoch_s_label: str = "barometer_epoch_s"
    barometer_fs_label: str = "barometer_sample_rate_hz"
    barometer_tfr_bits_label: str = "barometer_tfr_bits"
    barometer_tfr_frequency_hz_label: str = "barometer_tfr_frequency_hz"
    barometer_tfr_time_s_label: str = "barometer_tfr_time_s"

    # Accelerometer columns
    accelerometer_data_raw_label: str = "accelerometer_wf_raw"
    accelerometer_data_highpass_label: str = "accelerometer_wf_highpass"
    accelerometer_epoch_s_label: str = "accelerometer_epoch_s"
    accelerometer_fs_label: str = "accelerometer_sample_rate_hz"
    accelerometer_tfr_bits_label: str = "accelerometer_tfr_bits"
    accelerometer_tfr_frequency_hz_label: str = "accelerometer_tfr_frequency_hz"
    accelerometer_tfr_time_s_label: str = "accelerometer_tfr_time_s"

    # Gyroscope columns
    gyroscope_data_raw_label: str = "gyroscope_wf_raw"
    gyroscope_data_highpass_label: str = "gyroscope_wf_highpass"
    gyroscope_epoch_s_label: str = "gyroscope_epoch_s"
    gyroscope_fs_label: str = "gyroscope_sample_rate_hz"
    gyroscope_tfr_bits_label: str = "gyroscope_tfr_bits"
    gyroscope_tfr_frequency_hz_label: str = "gyroscope_tfr_frequency_hz"
    gyroscope_tfr_time_s_label: str = "gyroscope_tfr_time_s"

    # Magnetometer columns
    magnetometer_data_raw_label: str = "magnetometer_wf_raw"
    magnetometer_data_highpass_label: str = "magnetometer_wf_highpass"
    magnetometer_epoch_s_label: str = "magnetometer_epoch_s"
    magnetometer_fs_label: str = "magnetometer_sample_rate_hz"
    magnetometer_tfr_bits_label: str = "magnetometer_tfr_bits"
    magnetometer_tfr_frequency_hz_label: str = "magnetometer_tfr_frequency_hz"
    magnetometer_tfr_time_s_label: str = "magnetometer_tfr_time_s"

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
    print("tfr_type:", tfr_type)
    print("order:", band_order_Nth)
    for station in df_skyfall_data.index:
        station_id_str = df_skyfall_data[station_label][station]  # Get the station id

        if audio_data_label and audio_fs_label in df_skyfall_data.columns:
            print('mic_sample_rate_hz: ', df_skyfall_data[audio_fs_label][station])
            print('mic_epoch_s_0: ', df_skyfall_data[audio_epoch_s_label][station][0])

            # Frame to mic start and end and plot
            event_reference_time_epoch_s = df_skyfall_data[audio_epoch_s_label][station][0]
            print(event_reference_time_epoch_s)

            df_skyfall_data = rpd_tfr.tfr_bits_panda(df=df_skyfall_data,
                                                     sig_wf_label=audio_data_label,
                                                     sig_sample_rate_label=audio_fs_label,
                                                     order_number_input=band_order_Nth,
                                                     tfr_type=tfr_type,
                                                     new_column_tfr_bits=audio_tfr_bits_label,
                                                     new_column_tfr_frequency_hz=audio_tfr_frequency_hz_label,
                                                     new_column_tfr_time_s=audio_tfr_time_s_label)
            if verbosity > 1:
                pnl.plot_wf_mesh_vert(redvox_id=station_id_str,
                                      wf_panel_2_sig=df_skyfall_data[audio_data_label][station],
                                      wf_panel_2_time=df_skyfall_data[audio_epoch_s_label][station],
                                      mesh_time=df_skyfall_data[audio_tfr_time_s_label][station],
                                      mesh_frequency=df_skyfall_data[audio_tfr_frequency_hz_label][station],
                                      mesh_panel_0_tfr=df_skyfall_data[audio_tfr_bits_label][station],
                                      figure_title=EVENT_NAME + f": Audio, {tfr_type.upper()} and waveform",
                                      start_time_epoch=event_reference_time_epoch_s,
                                      mesh_panel_0_color_range=21,
                                      mesh_panel_0_colormap_scaling='range')

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

            if plot_raw_data:
                baro_raw_wf = df_skyfall_data[barometer_data_raw_label][station][0]
                baro_raw_tfr, baro_raw_tfr_bits, baro_raw_time_tfr_s, baro_raw_frequency_tfr_hz = \
                    stft_from_sig(baro_raw_wf,
                                  df_skyfall_data[barometer_fs_label][station],
                                  band_order_Nth)
                pnl.plot_wf_mesh_vert(redvox_id=station_id_str,
                                      wf_panel_2_sig=baro_raw_wf,
                                      wf_panel_2_time=df_skyfall_data[barometer_epoch_s_label][station],
                                      mesh_time=baro_raw_time_tfr_s,
                                      mesh_frequency=baro_raw_frequency_tfr_hz,
                                      mesh_panel_0_tfr=baro_raw_tfr_bits,
                                      figure_title=EVENT_NAME + ": Baro raw, STFT and waveform",
                                      start_time_epoch=event_reference_time_epoch_s)

            rpd_prep.df_column_unflatten(df=df_skyfall_data,
                                         col_wf_label=accelerometer_data_highpass_label,
                                         col_ndim_label=accelerometer_data_highpass_label + "_ndim")

            barometer_tfr_start_epoch: float = df_skyfall_data[barometer_epoch_s_label][0][0]
            print('Starting tfr_bits_panda for barometer:')
            df_skyfall_data = rpd_tfr.tfr_bits_panda(df=df_skyfall_data,
                                                     sig_wf_label=barometer_data_highpass_label,
                                                     sig_sample_rate_label=barometer_fs_label,
                                                     order_number_input=band_order_Nth,
                                                     tfr_type=tfr_type,
                                                     new_column_tfr_bits=barometer_tfr_bits_label,
                                                     new_column_tfr_frequency_hz=barometer_tfr_frequency_hz_label,
                                                     new_column_tfr_time_s=barometer_tfr_time_s_label)

            if verbosity > 1:
                pnl.plot_wf_mesh_vert(redvox_id=station_id_str,
                                      wf_panel_2_sig=df_skyfall_data[barometer_data_highpass_label][station][0],
                                      wf_panel_2_time=df_skyfall_data[barometer_epoch_s_label][station],
                                      mesh_time=df_skyfall_data[barometer_tfr_time_s_label][station][0],
                                      mesh_frequency=df_skyfall_data[barometer_tfr_frequency_hz_label][station][0],
                                      mesh_panel_0_tfr=df_skyfall_data[barometer_tfr_bits_label][station][0],
                                      figure_title=EVENT_NAME + f": Baro highpass, {tfr_type.upper()} and waveform",
                                      start_time_epoch=barometer_tfr_start_epoch)

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

            if verbosity > 2:
                if plot_raw_data:
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

            rpd_prep.df_column_unflatten(df=df_skyfall_data,
                                         col_wf_label=accelerometer_data_highpass_label,
                                         col_ndim_label=accelerometer_data_highpass_label + "_ndim")

            acceleromter_tfr_start_epoch: float = df_skyfall_data[accelerometer_epoch_s_label][0][0]
            print('Starting tfr_bits_panda for 3 channel acceleration:')
            df_skyfall_data = rpd_tfr.tfr_bits_panda(df=df_skyfall_data,
                                                     sig_wf_label=accelerometer_data_highpass_label,
                                                     sig_sample_rate_label=accelerometer_fs_label,
                                                     order_number_input=band_order_Nth,
                                                     tfr_type=tfr_type,
                                                     new_column_tfr_bits=accelerometer_tfr_bits_label,
                                                     new_column_tfr_frequency_hz=accelerometer_tfr_frequency_hz_label,
                                                     new_column_tfr_time_s=accelerometer_tfr_time_s_label)
            if verbosity > 1:
                for ax_n in range(3):
                    if plot_raw_data:
                        acc_raw_wf = df_skyfall_data[accelerometer_data_raw_label][station][ax_n]
                        acc_raw_tfr, acc_raw_tfr_bits, acc_raw_time_tfr_s, acc_raw_frequency_tfr_hz = \
                            stft_from_sig(acc_raw_wf,
                                          df_skyfall_data[accelerometer_fs_label][station],
                                          band_order_Nth)
                        pnl.plot_wf_mesh_vert(redvox_id=station_id_str,
                                              wf_panel_2_sig=acc_raw_wf,
                                              wf_panel_2_time=df_skyfall_data[accelerometer_epoch_s_label][station],
                                              mesh_time=acc_raw_time_tfr_s,
                                              mesh_frequency=acc_raw_frequency_tfr_hz,
                                              mesh_panel_0_tfr=acc_raw_tfr_bits,
                                              figure_title=EVENT_NAME + ": Acc " + axes[
                                                  ax_n] + " raw, STFT and waveform",
                                              start_time_epoch=event_reference_time_epoch_s)

                    pnl.plot_wf_mesh_vert(redvox_id=station_id_str,
                                          wf_panel_2_sig=df_skyfall_data[accelerometer_data_highpass_label][station][
                                              ax_n],
                                          wf_panel_2_time=df_skyfall_data[accelerometer_epoch_s_label][station],
                                          mesh_time=df_skyfall_data[accelerometer_tfr_time_s_label][station][ax_n],
                                          mesh_frequency=df_skyfall_data[accelerometer_tfr_frequency_hz_label][station][
                                              ax_n],
                                          mesh_panel_0_tfr=df_skyfall_data[accelerometer_tfr_bits_label][station][ax_n],
                                          figure_title=EVENT_NAME + f": Acc highpass, {tfr_type.upper()} and waveform",
                                          start_time_epoch=acceleromter_tfr_start_epoch)

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

            if verbosity > 2:
                if plot_raw_data:
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

            gyroscope_tfr_start_epoch: float = df_skyfall_data[gyroscope_epoch_s_label][0][0]
            print('Starting tfr_bits_panda for 3 channel gyroscope:')
            df_skyfall_data = rpd_tfr.tfr_bits_panda(df=df_skyfall_data,
                                                     sig_wf_label=gyroscope_data_highpass_label,
                                                     sig_sample_rate_label=gyroscope_fs_label,
                                                     order_number_input=band_order_Nth,
                                                     tfr_type=tfr_type,
                                                     new_column_tfr_bits=gyroscope_tfr_bits_label,
                                                     new_column_tfr_frequency_hz=gyroscope_tfr_frequency_hz_label,
                                                     new_column_tfr_time_s=gyroscope_tfr_time_s_label)
            if verbosity > 1:
                for ax_n in range(3):
                    if plot_raw_data:
                        # SKIP RAW
                        gyr_raw_wf = df_skyfall_data[gyroscope_data_raw_label][station][ax_n]
                        gyr_raw_tfr, gyr_raw_tfr_bits, gyr_raw_time_tfr_s, gyr_raw_frequency_tfr_hz = \
                            stft_from_sig(gyr_raw_wf,
                                          df_skyfall_data[gyroscope_fs_label][station],
                                          band_order_Nth)
                        pnl.plot_wf_mesh_vert(redvox_id=station_id_str,
                                              wf_panel_2_sig=gyr_raw_wf,
                                              wf_panel_2_time=df_skyfall_data[gyroscope_epoch_s_label][station],
                                              mesh_time=gyr_raw_time_tfr_s,
                                              mesh_frequency=gyr_raw_frequency_tfr_hz,
                                              mesh_panel_0_tfr=gyr_raw_tfr_bits,
                                              figure_title=EVENT_NAME + ": Gyr " + axes[
                                                  ax_n] + " raw, STFT and waveform",
                                              start_time_epoch=event_reference_time_epoch_s)

                    pnl.plot_wf_mesh_vert(redvox_id=station_id_str,
                                          wf_panel_2_sig=df_skyfall_data[gyroscope_data_highpass_label][station][ax_n],
                                          wf_panel_2_time=df_skyfall_data[gyroscope_epoch_s_label][station],
                                          mesh_time=df_skyfall_data[gyroscope_tfr_time_s_label][station][ax_n],
                                          mesh_frequency=df_skyfall_data[gyroscope_tfr_frequency_hz_label][station][
                                              ax_n],
                                          mesh_panel_0_tfr=df_skyfall_data[gyroscope_tfr_bits_label][station][ax_n],
                                          figure_title=EVENT_NAME + f": Gyr highpass, {tfr_type.upper()} and waveform",
                                          start_time_epoch=gyroscope_tfr_start_epoch)

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

            if verbosity > 2:
                if plot_raw_data:
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

            magnetometer_tfr_start_epoch: float = df_skyfall_data[magnetometer_epoch_s_label][0][0]
            print('Starting tfr_bits_panda for 3 channel magnetometer:')
            df_skyfall_data = rpd_tfr.tfr_bits_panda(df=df_skyfall_data,
                                                     sig_wf_label=magnetometer_data_highpass_label,
                                                     sig_sample_rate_label=magnetometer_fs_label,
                                                     order_number_input=band_order_Nth,
                                                     tfr_type=tfr_type,
                                                     new_column_tfr_bits=magnetometer_tfr_bits_label,
                                                     new_column_tfr_frequency_hz=magnetometer_tfr_frequency_hz_label,
                                                     new_column_tfr_time_s=magnetometer_tfr_time_s_label)
            if verbosity > 1:
                for ax_n in range(3):
                    if plot_raw_data:
                        mag_raw_wf = df_skyfall_data[magnetometer_data_raw_label][station][ax_n]
                        mag_raw_tfr, mag_raw_tfr_bits, mag_raw_time_tfr_s, mag_raw_frequency_tfr_hz = \
                            stft_from_sig(mag_raw_wf,
                                          df_skyfall_data[magnetometer_fs_label][station],
                                          band_order_Nth)
                        pnl.plot_wf_mesh_vert(redvox_id=station_id_str,
                                              wf_panel_2_sig=mag_raw_wf,
                                              wf_panel_2_time=df_skyfall_data[magnetometer_epoch_s_label][station],
                                              mesh_time=mag_raw_time_tfr_s,
                                              mesh_frequency=mag_raw_frequency_tfr_hz,
                                              mesh_panel_0_tfr=mag_raw_tfr_bits,
                                              figure_title=EVENT_NAME + ": Mag " + axes[
                                                  ax_n] + " raw, STFT and waveform",
                                              start_time_epoch=event_reference_time_epoch_s)

                    pnl.plot_wf_mesh_vert(redvox_id=station_id_str,
                                          wf_panel_2_sig=df_skyfall_data[magnetometer_data_highpass_label][station][
                                              ax_n],
                                          wf_panel_2_time=df_skyfall_data[magnetometer_epoch_s_label][station],
                                          mesh_time=df_skyfall_data[magnetometer_tfr_time_s_label][station][ax_n],
                                          mesh_frequency=df_skyfall_data[magnetometer_tfr_frequency_hz_label][station][
                                              ax_n],
                                          mesh_panel_0_tfr=df_skyfall_data[magnetometer_tfr_bits_label][station][ax_n],
                                          figure_title=EVENT_NAME + f": Mag highpass, {tfr_type.upper()} and waveform",
                                          start_time_epoch=magnetometer_tfr_start_epoch)

        if verbosity > 0:
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

            # Plot TFR sensor wiggles
            # TODO: find better method for TFR wiggle plot, current method is very hacky mashup of
            #  plot_sensor_wiggles_pandas and plot_mesh_pandas
            sensor_names_list = ["audio", "barometer", "accelerometer", "gyroscope", "magnetometer"]
            df_skyfall_data_mesh_hack = pd.DataFrame(index=sensor_ticklabels_list,
                                                     columns=["tfr_bits", "tfr_freq_hz", "tfr_time_s"])

            n_sensors = len(sensor_names_list)
            n_channels = len(sensor_ticklabels_list)
            i, j = 0, 0
            while i < n_sensors:
                sensor_tfr_bits_label = f"{sensor_names_list[i]}_tfr_bits"
                sensor_tfr_freq_hz_label = f"{sensor_names_list[i]}_tfr_frequency_hz"
                sensor_tfr_time_s_label = f"{sensor_names_list[i]}_tfr_time_s"
                if sensor_names_list[i] not in ["audio", "barometer"]:
                    for k in range(3):
                        df_skyfall_data_mesh_hack["tfr_bits"][sensor_ticklabels_list[j]] = \
                            df_skyfall_data[sensor_tfr_bits_label][station][k]
                        df_skyfall_data_mesh_hack["tfr_freq_hz"][sensor_ticklabels_list[j]] = \
                            df_skyfall_data[sensor_tfr_freq_hz_label][station][k]
                        df_skyfall_data_mesh_hack["tfr_time_s"][sensor_ticklabels_list[j]] = \
                            df_skyfall_data[sensor_tfr_time_s_label][station][k]
                        j += 1
                    i += 1
                else:
                    if df_skyfall_data[sensor_column_label_list[i]][station].ndim == 1:
                        df_skyfall_data_mesh_hack["tfr_bits"][sensor_ticklabels_list[j]] = \
                            df_skyfall_data[sensor_tfr_bits_label][station]
                        df_skyfall_data_mesh_hack["tfr_freq_hz"][sensor_ticklabels_list[j]] = \
                            df_skyfall_data[sensor_tfr_freq_hz_label][station]
                        df_skyfall_data_mesh_hack["tfr_time_s"][sensor_ticklabels_list[j]] = \
                            df_skyfall_data[sensor_tfr_time_s_label][station]
                    else:
                        df_skyfall_data_mesh_hack["tfr_bits"][sensor_ticklabels_list[j]] = \
                            df_skyfall_data[sensor_tfr_bits_label][station][0]
                        df_skyfall_data_mesh_hack["tfr_freq_hz"][sensor_ticklabels_list[j]] = \
                            df_skyfall_data[sensor_tfr_freq_hz_label][station][0]
                        df_skyfall_data_mesh_hack["tfr_time_s"][sensor_ticklabels_list[j]] = \
                            df_skyfall_data[sensor_tfr_time_s_label][station][0]
                    j += 1
                    i += 1

            df_skyfall_data_mesh_hack["tick_labels"] = sensor_ticklabels_list

            rpd_plot.plot_mesh_pandas(df=df_skyfall_data_mesh_hack,
                                      mesh_time_label="tfr_time_s",
                                      mesh_tfr_label="tfr_bits",
                                      mesh_frequency_label="tfr_freq_hz",
                                      t0_sig_epoch_s=event_reference_time_epoch_s,
                                      sig_id_label="tick_labels",
                                      fig_title=tfr_type.upper())

        plt.show()
