# todo: address possible invalid values in building plots section
# Python libraries
import os.path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dtime
from scipy.signal import welch

# RedVox RedPandas and related RedVox modules
from redvox.common.data_window import DataWindow
import redvox.common.date_time_utils as dt
import redpandas.redpd_preprocess as rpd_prep
import redpandas.redpd_build_station as rpd_build_sta
import redpandas.redpd_plot as rpd_plot
import redpandas.redpd_geospatial as rpd_geo
from redpandas.redpd_scales import METERS_TO_KM
from libquantum.plot_templates import plot_time_frequency_reps as pnl

# Configuration file
from examples.skyfall.skyfall_config import EVENT_NAME, INPUT_DIR, OUTPUT_DIR, EPISODE_START_EPOCH_S, \
    EPISODE_END_EPOCH_S, STATIONS, DW_FILE, use_datawindow_tdr, use_pickle_tdr, use_parquet_tdr, PD_PQT_FILE, SENSOR_LABEL, \
    ref_latitude_deg, ref_longitude_deg, ref_altitude_m, ref_epoch_s, OTHER_INPUT_PATH, OTHER_PD_PQT_FILE

# TODO MC: plot 3c hp in same panel, sqrt(add squares) for power in another panel, top panel TBD

if __name__ == "__main__":
    """
    RedVox RedPandas time-domain representation of API900 data. Example: Skyfall.
    Last updated: 1 June 2021
    """

    # Load parquet with bounder data fields
    bounder_loc = pd.read_parquet(os.path.join(OTHER_INPUT_PATH, OTHER_PD_PQT_FILE))

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

    # Health columns
    health_battery_charge_label: str = 'battery_charge_remaining_per'
    health_internal_temp_deg_C_label: str = 'internal_temp_deg_C'
    health_network_type_label: str = 'network_type'
    health_epoch_s_label: str = 'health_epoch_s'

    # Location columns
    location_latitude_label: str = 'location_latitude'
    location_longitude_label: str = 'location_longitude'
    location_altitude_label: str = "location_altitude"
    location_speed_label: str = 'location_speed'
    location_epoch_s_label: str = 'location_epoch_s'
    location_provider_label: str = 'location_provider'

    # Synchronization columns
    synchronization_epoch_label: str = 'synchronization_epoch_s'
    synchronization_latency_label: str = 'synchronization_latency_ms'
    synchronization_offset_label: str = 'synchronization_offset_ms'
    synchronization_best_offset_label: str = 'synchronization_best_offset_ms'
    synchronization_offset_delta_label: str = 'synchronization_offset_delta_ms'
    synchronization_number_exchanges_label: str = 'synchronization_number_exchanges'

    # Load data options
    if use_datawindow_tdr is True or use_pickle_tdr is True:
        print("Initiating Conversion from RedVox DataWindow to RedVox RedPandas:")
        if use_datawindow_tdr:  # Option A: Create DataWindow object
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

    elif use_parquet_tdr:  # Option C: Open dataframe from parquet file
        print("Loading existing RedPandas Parquet...", end=" ")
        df_skyfall_data = pd.read_parquet(os.path.join(OUTPUT_DIR, PD_PQT_FILE))
        print(f"Done. RedVox SDK version: {df_skyfall_data[redvox_sdk_version_label][0]}")

    else:
        print('\nNo data loading method selected. '
              'Check that use_datawindow_tdr, use_pickle_tdr, or use_parquet_tdr in the Skyfall configuration file are set to True.')
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

        if barometer_data_raw_label and barometer_data_highpass_label and barometer_fs_label in df_skyfall_data.columns:
            if use_parquet_tdr is True and use_datawindow_tdr is False and use_pickle_tdr is False:
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
            baro_height_from_bounder_km = barometer_height_m*METERS_TO_KM



            # plt.figure()
            # # Actual plot
            # plt.plot(time_bar, barometer_height_m * METERS_TO_KM, label='Barometer Z')
            # plt.plot(time_bounder, bounder_loc['Alt_m'] * METERS_TO_KM, label='Bounder Z')
            # plt.ylabel('Height, km')
            # plt.xlabel(f"Time (s) from UTC "
            #            f"{dtime.datetime.utcfromtimestamp(t0_loc).strftime('%Y-%m-%d %H:%M:%S')}")
            # # plt.xlim([0, 1800])
            # plt.legend()
            # plt.tight_layout()



        # Repeat here
        if accelerometer_data_raw_label and accelerometer_fs_label and accelerometer_data_highpass_label\
                in df_skyfall_data.columns:
            if use_parquet_tdr is True and use_datawindow_tdr is False and use_pickle_tdr is False:
                # Reshape wf columns
                rpd_prep.df_column_unflatten(df=df_skyfall_data,
                                             col_wf_label=accelerometer_data_raw_label,
                                             col_ndim_label=accelerometer_data_raw_label + "_ndim")
                rpd_prep.df_column_unflatten(df=df_skyfall_data,
                                             col_wf_label=accelerometer_data_highpass_label,
                                             col_ndim_label=accelerometer_data_highpass_label + "_ndim")

            print('accelerometer_sample_rate_hz:', df_skyfall_data[accelerometer_fs_label][station])
            print('accelerometer_epoch_s_0:',  df_skyfall_data[accelerometer_epoch_s_label][station][0],
                  df_skyfall_data[accelerometer_epoch_s_label][station][-1])

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
                                   figure_title_show=False,  # for press
                                   label_panel_show=True,  # for press
                                   labels_fontweight='bold')

        if accelerometer_data_raw_label and barometer_data_raw_label in df_skyfall_data.columns:
            # Plot aligned waveforms for sensor payload
            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=df_skyfall_data[audio_data_label][station],
                                   wf_panel_2_time=df_skyfall_data[audio_epoch_s_label][station],
                                   wf_panel_1_sig=df_skyfall_data[accelerometer_data_highpass_label][station][2],
                                   wf_panel_1_time=df_skyfall_data[accelerometer_epoch_s_label][station],
                                   wf_panel_0_sig=df_skyfall_data[barometer_data_highpass_label][station][0],
                                   wf_panel_0_time=df_skyfall_data[barometer_epoch_s_label][station],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Mic, Norm,",
                                   wf_panel_1_units="Acc Z hp, m/$s^2$",
                                   wf_panel_0_units="Bar hp, kPa",
                                   figure_title=EVENT_NAME + " with Acc and Bar Highpass",
                                   figure_title_show=False,
                                   label_panel_show=True,  # for press
                                   labels_fontweight='bold')

            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=df_skyfall_data[audio_data_label][station],
                                   wf_panel_2_time=df_skyfall_data[audio_epoch_s_label][station],
                                   wf_panel_1_sig=df_skyfall_data[accelerometer_data_raw_label][station][2],
                                   wf_panel_1_time=df_skyfall_data[accelerometer_epoch_s_label][station],
                                   wf_panel_0_sig=baro_height_from_bounder_km,
                                   wf_panel_0_time=df_skyfall_data[barometer_epoch_s_label][station],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Mic, Norm",
                                   wf_panel_1_units="Acc Z, m/$s^2$",
                                   wf_panel_0_units="Bar Z Height, km",
                                   figure_title=EVENT_NAME,
                                   figure_title_show=False,
                                   label_panel_show=True,  # for press
                                   labels_fontweight='bold')

        if gyroscope_data_raw_label and gyroscope_fs_label and gyroscope_data_highpass_label\
                in df_skyfall_data.columns:

            if use_parquet_tdr is True and use_datawindow_tdr is False and use_pickle_tdr is False:
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
            # Plot 3c raw gyroscope waveforms
            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=df_skyfall_data[gyroscope_data_raw_label][station][2],
                                   wf_panel_2_time=df_skyfall_data[gyroscope_epoch_s_label][station],
                                   wf_panel_1_sig=df_skyfall_data[gyroscope_data_raw_label][station][1],
                                   wf_panel_1_time=df_skyfall_data[gyroscope_epoch_s_label][station],
                                   wf_panel_0_sig=df_skyfall_data[gyroscope_data_raw_label][station][0],
                                   wf_panel_0_time=df_skyfall_data[gyroscope_epoch_s_label][station],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Gyr Z, rad/s",
                                   wf_panel_1_units="Gyr Y, rad/s",
                                   wf_panel_0_units="Gyr X, rad/s",
                                   figure_title=EVENT_NAME + ": Gyroscope raw",
                                   figure_title_show=False,
                                   label_panel_show=True,  # for press
                                   labels_fontweight='bold')

        if magnetometer_data_raw_label and magnetometer_fs_label and magnetometer_data_highpass_label\
                in df_skyfall_data.columns:
            if use_parquet_tdr is True and use_datawindow_tdr is False and use_pickle_tdr is False:
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
            # Plot 3c magnetometer raw waveforms
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
                                   figure_title_show=False,
                                   label_panel_show=True,  # for press
                                   labels_fontweight='bold')

        if location_latitude_label and location_longitude_label and location_altitude_label and location_speed_label \
                in df_skyfall_data.columns:

            print("Bounder End EPOCH:", ref_epoch_s)
            print("Bounder End LAT LON ALT:", ref_latitude_deg, ref_longitude_deg, ref_altitude_m)

            # Compute ENU projections
            df_range_z_speed = \
                rpd_geo.compute_t_r_z_speed(unix_s=df_skyfall_data[location_epoch_s_label][station],
                                            lat_deg=df_skyfall_data[location_latitude_label][station],
                                            lon_deg=df_skyfall_data[location_longitude_label][station],
                                            alt_m=df_skyfall_data[location_altitude_label][station],
                                            ref_unix_s=ref_epoch_s,
                                            ref_lat_deg=ref_latitude_deg,
                                            ref_lon_deg=ref_longitude_deg,
                                            ref_alt_m=ref_altitude_m)

            # Plot location framework
            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=df_range_z_speed['Range_m']*METERS_TO_KM,
                                   wf_panel_2_time=df_skyfall_data[location_epoch_s_label][station],
                                   wf_panel_1_sig=df_range_z_speed['Z_m']*METERS_TO_KM,
                                   wf_panel_1_time=df_skyfall_data[location_epoch_s_label][station],
                                   wf_panel_0_sig=df_skyfall_data[location_speed_label][station],
                                   wf_panel_0_time=df_skyfall_data[location_epoch_s_label][station],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Range, km",
                                   wf_panel_1_units="Altitude, km",
                                   wf_panel_0_units="Speed, m/s",
                                   figure_title=EVENT_NAME + ": Location Framework",
                                   figure_title_show=False,
                                   label_panel_show=True,  # for press
                                   labels_fontweight='bold')

            if location_epoch_s_label and location_altitude_label and barometer_epoch_s_label and \
                barometer_data_raw_label in df_skyfall_data.columns:

                plt.figure()
                time_bar = df_skyfall_data[barometer_epoch_s_label][station] - EPISODE_START_EPOCH_S
                time_bounder = bounder_loc['Epoch_s'] - EPISODE_START_EPOCH_S
                time_loc = df_skyfall_data[location_epoch_s_label][station] - EPISODE_START_EPOCH_S

                ax1 = plt.subplot(211)
                plt.semilogy(time_bar, df_skyfall_data[barometer_data_raw_label][station][0], 'midnightblue',
                             label='Barometer kPa')
                plt.semilogy(time_bounder, bounder_loc['Pres_kPa'], 'g', label='Bounder kPa')
                plt.ylabel('Pressure, kPa')
                plt.legend(loc='lower right')
                plt.xlim([0, 1800])
                plt.text(0.01, 0.9, "(b)", transform=ax1.transAxes,  fontweight='bold')
                ax1.set_xticklabels([])
                plt.grid(True)

                ax2=plt.subplot(212)
                plt.plot(time_loc, df_skyfall_data[location_altitude_label][station] * METERS_TO_KM, 'r',
                         label='Location sensor')
                plt.plot(time_bar, barometer_height_m * METERS_TO_KM, 'midnightblue', label='Barometer Z')
                plt.plot(time_bounder, bounder_loc['Alt_m'] * METERS_TO_KM, 'g', label='Bounder Z')
                plt.ylabel('Height, km')
                plt.xlabel(f"Time (s) from UTC "
                           f"{dtime.datetime.utcfromtimestamp(EPISODE_START_EPOCH_S).strftime('%Y-%m-%d %H:%M:%S')}")
                plt.legend(loc='upper right')
                plt.xlim([0, 1800])
                plt.text(0.01, 0.05, "(a)", transform=ax2.transAxes,  fontweight='bold')
                plt.grid(True)
                plt.tight_layout()

        if health_battery_charge_label and health_internal_temp_deg_C_label and health_network_type_label \
                and barometer_data_raw_label and location_provider_label in df_skyfall_data.columns:

            print(f"location_provider_epoch_s_0: {df_skyfall_data[location_provider_label][station][0]}",
                  f", location_provider_epoch_s_end: {df_skyfall_data[location_provider_label][station][-1]}")
            print(f"network_type_epoch_s_0: {df_skyfall_data[health_network_type_label][station][0]}",
                  f", network_type_epoch_s_end: {df_skyfall_data[health_network_type_label][station][-1]}")

            # Other interesting fields: Estimated Height ASL, Internal Temp, % Battery
            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=barometer_height_m,
                                   wf_panel_2_time=df_skyfall_data[barometer_epoch_s_label][station],
                                   wf_panel_1_sig=df_skyfall_data[health_internal_temp_deg_C_label][station],
                                   wf_panel_1_time=df_skyfall_data[health_epoch_s_label][station],
                                   wf_panel_0_sig=df_skyfall_data[health_battery_charge_label][station],
                                   wf_panel_0_time=df_skyfall_data[health_epoch_s_label][station],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Bar Z Height, m",
                                   wf_panel_1_units="Temp, $^oC$",
                                   wf_panel_0_units="Battery %",
                                   figure_title=EVENT_NAME + ": Station Status",
                                   figure_title_show=False,
                                   label_panel_show=True,  # for press
                                   labels_fontweight='bold')

        if synchronization_epoch_label and synchronization_latency_label and synchronization_offset_label \
                and synchronization_best_offset_label and synchronization_offset_delta_label and \
                synchronization_number_exchanges_label in df_skyfall_data.columns:

            # Plot synchronization framework
            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=df_skyfall_data[synchronization_latency_label][station],
                                   wf_panel_2_time=df_skyfall_data[synchronization_epoch_label][station],
                                   wf_panel_1_sig=df_skyfall_data[synchronization_offset_label][station],
                                   wf_panel_1_time=df_skyfall_data[synchronization_epoch_label][station],
                                   wf_panel_0_sig=df_skyfall_data[synchronization_offset_delta_label][station],
                                   wf_panel_0_time=df_skyfall_data[synchronization_epoch_label][station],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Latency, ms",
                                   wf_panel_1_units="Offset, s",
                                   wf_panel_0_units="Offset delta, s",
                                   figure_title=EVENT_NAME + ": Synchronization Framework",
                                   figure_title_show=False,
                                   label_panel_show=True,  # for press
                                   labels_fontweight='bold')

            # Plot synchronization framework
            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=df_skyfall_data[synchronization_latency_label][station],
                                   wf_panel_2_time=df_skyfall_data[synchronization_epoch_label][station],
                                   wf_panel_1_sig=df_skyfall_data[synchronization_offset_label][station],
                                   wf_panel_1_time=df_skyfall_data[synchronization_epoch_label][station],
                                   wf_panel_0_sig=df_skyfall_data[location_altitude_label][station] * METERS_TO_KM,
                                   wf_panel_0_time=df_skyfall_data[location_epoch_s_label][station],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Latency, ms",
                                   wf_panel_1_units="Offset, s",
                                   wf_panel_0_units="Height, km",
                                   figure_title=EVENT_NAME + ": Synchronization Framework",
                                   figure_title_show=False,
                                   label_panel_show=True,  # for press
                                   labels_fontweight='bold')

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

        # plt.show()

    acc_x_energy = sum(df_skyfall_data[accelerometer_data_highpass_label][0][0] ** 2)
    acc_y_energy = sum(df_skyfall_data[accelerometer_data_highpass_label][0][1] ** 2)
    acc_z_energy = sum(df_skyfall_data[accelerometer_data_highpass_label][0][2] ** 2)
    magnitude_acc = np.sqrt(acc_x_energy + acc_y_energy + acc_z_energy)  # euclidean sum
    print(f'magitude acc vector: {magnitude_acc}',
          f'power: {(acc_x_energy + acc_y_energy + acc_z_energy) / len(df_skyfall_data[accelerometer_epoch_s_label][0])}',
          f'rms: {np.sqrt((acc_x_energy + acc_y_energy + acc_z_energy) / len(df_skyfall_data[accelerometer_epoch_s_label][0]))}')

    # Power spectrum density using welch
    f_x, Pxx_x = welch(x=df_skyfall_data[accelerometer_data_highpass_label][0][0],
                        fs=df_skyfall_data[accelerometer_fs_label][0])
    f_y, Pxx_y = welch(x=df_skyfall_data[accelerometer_data_highpass_label][0][1],
                       fs=df_skyfall_data[accelerometer_fs_label][0])
    f_z, Pxx_z = welch(x=df_skyfall_data[accelerometer_data_highpass_label][0][2],
                       fs=df_skyfall_data[accelerometer_fs_label][0])

    fig = plt.figure()
    plt.suptitle('PSD Acc')
    ax0 = fig.add_subplot(411)
    ax0.plot(f_x, Pxx_x, color='midnightblue')
    ax0.set_yscale('log')
    ax1 = fig.add_subplot(412, sharex=ax0)
    ax1.plot(f_y, Pxx_y, color='midnightblue')
    ax1.set_yscale('log')
    ax2 = fig.add_subplot(413, sharex=ax0)
    ax2.plot(f_z, Pxx_z, color='midnightblue')
    ax2.set_yscale('log')
    ax2.set_xlabel('Frequency [Hz]')
    ax3 = plt.subplot(414)
    ax3.plot(df_skyfall_data[accelerometer_epoch_s_label][0]-df_skyfall_data[accelerometer_epoch_s_label][0][0],
             df_skyfall_data[accelerometer_data_highpass_label][0][0], label="X")
    ax3.plot(df_skyfall_data[accelerometer_epoch_s_label][0]-df_skyfall_data[accelerometer_epoch_s_label][0][0],
             df_skyfall_data[accelerometer_data_highpass_label][0][1], label="Y")
    ax3.plot(df_skyfall_data[accelerometer_epoch_s_label][0]-df_skyfall_data[accelerometer_epoch_s_label][0][0],
             df_skyfall_data[accelerometer_data_highpass_label][0][2], label="Z")
    ax3.set_xlabel('Time (s)')
    plt.legend()

    # Power spectrum density using welch
    f_x, Pxx_x = welch(x=df_skyfall_data[gyroscope_data_highpass_label][0][0],
                       fs=df_skyfall_data[gyroscope_fs_label][0])
    f_y, Pxx_y = welch(x=df_skyfall_data[gyroscope_data_highpass_label][0][1],
                       fs=df_skyfall_data[gyroscope_fs_label][0])
    f_z, Pxx_z = welch(x=df_skyfall_data[gyroscope_data_highpass_label][0][2],
                       fs=df_skyfall_data[gyroscope_fs_label][0])

    fig, ax = plt.subplots(nrows=3, sharex='col')
    plt.suptitle('PSD Gyr')
    ax[0].semilogy(f_x, Pxx_x, color='midnightblue')
    ax[1].semilogy(f_y, Pxx_y, color='midnightblue')
    ax[2].semilogy(f_z, Pxx_z, color='midnightblue')
    ax[2].set_xlabel('Frequency [Hz]')

    # Power spectrum density using welch
    f_x, Pxx_x = welch(x=df_skyfall_data[magnetometer_data_highpass_label][0][0],
                       fs=df_skyfall_data[magnetometer_fs_label][0])
    f_y, Pxx_y = welch(x=df_skyfall_data[magnetometer_data_highpass_label][0][1],
                       fs=df_skyfall_data[magnetometer_fs_label][0])
    f_z, Pxx_z = welch(x=df_skyfall_data[magnetometer_data_highpass_label][0][2],
                       fs=df_skyfall_data[magnetometer_fs_label][0])

    fig, ax = plt.subplots(nrows=3, sharex='col')
    plt.suptitle('PSD Mag')
    ax[0].semilogy(f_x, Pxx_x, color='midnightblue')
    ax[1].semilogy(f_y, Pxx_y, color='midnightblue')
    ax[2].semilogy(f_z, Pxx_z, color='midnightblue')
    ax[2].set_xlabel('Frequency [Hz]')
    plt.show()

