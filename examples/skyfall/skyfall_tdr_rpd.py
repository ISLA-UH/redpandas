# todo: address possible invalid values in building plots section
# Python libraries
import os.path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# RedVox and Red Pandas modules
# from redvox.common.data_window import DataWindow
from redvox.common.data_window import DataWindowFast
import redvox.common.date_time_utils as dt
import redpandas.redpd_preprocess as rpd_prep
import redpandas.redpd_scales as rpd_scales
import redpandas.redpd_build_station as rpd_build_sta
import redpandas.redpd_plot as rpd_plot
from libquantum.plot_templates import plot_time_frequency_reps as pnl

# Configuration file
from examples.skyfall.skyfall_config import EVENT_NAME, INPUT_DIR, OUTPUT_DIR, EPISODE_START_EPOCH_S, \
    EPISODE_END_EPOCH_S, STATIONS, DW_FILE, use_datawindow, use_pickle, use_parquet, PD_PQT_FILE, SENSOR_LABEL


# Verify points to correct config file.
# TODO MC: plot hp in same panel, sqrt(add squares) for power in another panel, top panel TBD
# TODO MC: wiggles for sensors_raw and sensors_highpass

# TODO: decide where this fcn lives
def df_column_unflatten(df: pd.DataFrame,
                        col_wf_label: str,
                        col_ndim_label: str):
    """
    Restores original shape of elements in column

    :param df: pandas DataFrame
    :param col_wf_label: column label for data that needs reshaping, usually waveform arrays.
    :param col_ndim_label: column label with dimensions for reshaping. Elements in column need to be a numpy array.
    :return: original df, replaces column values with reshaped ones
    """

    col_values = df[col_wf_label].to_numpy()
    for index_array in df.index:
        col_values[index_array].shape = (df[col_ndim_label][index_array][0],
                                         df[col_ndim_label][index_array][1])


if __name__ == "__main__":
    """
    Red Pandas time-frequency representation of API900 data
    Last updated: 11 May 2021
    """

    print('Let the sky fall')

    # Label columns in dataframe
    station_label: str = "station_id"

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

    # Load data options
    if use_datawindow is True or use_pickle is True:
        if use_datawindow:  # Option A: Create DataWindow object
            rdvx_data = DataWindowFast(input_dir=INPUT_DIR,
                                       station_ids=STATIONS,
                                       start_datetime=dt.datetime_from_epoch_seconds_utc(EPISODE_START_EPOCH_S),
                                       end_datetime=dt.datetime_from_epoch_seconds_utc(EPISODE_END_EPOCH_S),
                                       apply_correction=True,
                                       structured_layout=True)

        else:  # Option B: Load pickle with DataWindow object. Assume compressed
            rdvx_data: DataWindowFast = DataWindowFast.from_json_file(base_dir=OUTPUT_DIR, file_name=DW_FILE)

        # BEGIN RED PANDAS
        list_df_stations = []  # list to store dataframes with sensors for one station

        for station in rdvx_data.stations:

            list_df_sensors_per_station = []  # list to store sensor dataframes for one station

            # TODO: update to copy skyfall template basic
            dict_for_station_id = {'station_id': [station.id],
                                   'station_make': [station.metadata.make],
                                   'station_model': [station.metadata.model],
                                   'station_app_version': [station.metadata.app_version],
                                   'datawin_sdk_version': ["3.0.0rc31"]}

            df_station_id = pd.DataFrame.from_dict(data=dict_for_station_id)
            list_df_sensors_per_station.append(df_station_id)

            print(f"Prep {station.id}...", end=" ")

            for label in SENSOR_LABEL:
                print(f"{label} sensor...", end=" ")
                df_sensor = rpd_build_sta.build_station(station=station,
                                                        sensor_label=label)
                list_df_sensors_per_station.append(df_sensor)

            print(f"Done.")

            # convert list of sensor dataframes into one station dataframe
            df_all_sensors_one_station = pd.concat(list_df_sensors_per_station, axis=1)
            list_df_stations.append(df_all_sensors_one_station)

        # convert list of station dataframes into one master dataframe to later parquet
        df_skyfall_data = df_all_sensors_one_station = pd.concat(list_df_stations, axis=0)
        df_skyfall_data.sort_values(by="station_id", ignore_index=True, inplace=True)  # sort by station id

    elif use_parquet:  # Option C: Open dataframe from parquet file
        df_skyfall_data = pd.read_parquet(os.path.join(OUTPUT_DIR, PD_PQT_FILE))

    else:
        print('No data loading method selected. '
              'Check use_datawindow, use_pickle, or use_parquet in the Skyfall configuration file are set to True.')
        exit()

    # Start of building plots
    for station in df_skyfall_data.index:
        station_id_str = df_skyfall_data[station_label][station]  # Get the station id

        if audio_data_label and audio_fs_label in df_skyfall_data.columns:

            print('\nmic_sample_rate_hz: ', df_skyfall_data[audio_fs_label][station])
            print('mic_epoch_s_0: ', df_skyfall_data[audio_epoch_s_label][station][0])

            # Frame to mic start and end and plot
            event_reference_time_epoch_s = df_skyfall_data[audio_epoch_s_label][station][0]

        if barometer_data_raw_label and barometer_data_highpass_label and barometer_fs_label in df_skyfall_data.columns:
            if use_parquet:
                # Reshape wf columns
                df_column_unflatten(df=df_skyfall_data,
                                    col_wf_label=barometer_data_raw_label,
                                    col_ndim_label=barometer_data_raw_label + "_ndim")

                df_column_unflatten(df=df_skyfall_data,
                                    col_wf_label=barometer_data_highpass_label,
                                    col_ndim_label=barometer_data_highpass_label + "_ndim")

            print('barometer_sample_rate_hz:', df_skyfall_data[barometer_fs_label][station])
            print('barometer_epoch_s_0:', df_skyfall_data[barometer_epoch_s_label][station][0])

            barometer_height_m = \
                rpd_prep.model_height_from_pressure_skyfall(df_skyfall_data[barometer_data_raw_label][station][0])
            baro_height_from_bounder_km = barometer_height_m/1E3

        # Repeat here
        if accelerometer_data_raw_label and accelerometer_fs_label and accelerometer_data_highpass_label\
                in df_skyfall_data.columns:
            if use_parquet:
                # Reshape wf columns
                df_column_unflatten(df=df_skyfall_data,
                                    col_wf_label=accelerometer_data_raw_label,
                                    col_ndim_label=accelerometer_data_raw_label + "_ndim")
                df_column_unflatten(df=df_skyfall_data,
                                    col_wf_label=accelerometer_data_highpass_label,
                                    col_ndim_label=accelerometer_data_highpass_label + "_ndim")

            print('accelerometer_sample_rate_hz:', df_skyfall_data[accelerometer_fs_label][station])
            print('accelerometer_epoch_s_0:',  df_skyfall_data[accelerometer_epoch_s_label][station][0],
                  df_skyfall_data[accelerometer_epoch_s_label][station][-1])

            # Plot aligned raw waveforms
            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=df_skyfall_data[accelerometer_data_raw_label][station][2],
                                   wf_panel_2_time=df_skyfall_data[accelerometer_epoch_s_label][station],
                                   wf_panel_1_sig=df_skyfall_data[accelerometer_data_raw_label][station][1],
                                   wf_panel_1_time=df_skyfall_data[accelerometer_epoch_s_label][station],
                                   wf_panel_0_sig=df_skyfall_data[accelerometer_data_raw_label][station][0],
                                   wf_panel_0_time=df_skyfall_data[accelerometer_epoch_s_label][station],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Z, m/$s^2$",
                                   wf_panel_1_units="Y, m/$s^2$",
                                   wf_panel_0_units="X, m/$s^2$",
                                   figure_title=EVENT_NAME + ": Accelerometer raw")

            # Plot aligned highpassed waveforms
            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=df_skyfall_data[accelerometer_data_highpass_label][station][2],
                                   wf_panel_2_time=df_skyfall_data[accelerometer_epoch_s_label][station],
                                   wf_panel_1_sig=df_skyfall_data[accelerometer_data_highpass_label][station][1],
                                   wf_panel_1_time=df_skyfall_data[accelerometer_epoch_s_label][station],
                                   wf_panel_0_sig=df_skyfall_data[accelerometer_data_highpass_label][station][0],
                                   wf_panel_0_time=df_skyfall_data[accelerometer_epoch_s_label][station],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Z, m/$s^2$",
                                   wf_panel_1_units="Y, m/$s^2$",
                                   wf_panel_0_units="X, m/$s^2$",
                                   figure_title=EVENT_NAME + ": Accelerometer highpass")

        if accelerometer_data_raw_label and barometer_data_raw_label in df_skyfall_data.columns:
            # Plot aligned waveforms
            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=df_skyfall_data[audio_data_label][station],
                                   wf_panel_2_time=df_skyfall_data[audio_epoch_s_label][station],
                                   wf_panel_1_sig=df_skyfall_data[accelerometer_data_highpass_label][station][2],
                                   wf_panel_1_time=df_skyfall_data[accelerometer_epoch_s_label][station],
                                   wf_panel_0_sig=df_skyfall_data[barometer_data_highpass_label][station][0],
                                   wf_panel_0_time=df_skyfall_data[barometer_epoch_s_label][station],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Mic, Bits",
                                   wf_panel_1_units="Acc Z, m/$s^2$",
                                   wf_panel_0_units="Bar, kPa",
                                   figure_title=EVENT_NAME + " with Acc and Bar Highpass")

            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=df_skyfall_data[audio_data_label][station],
                                   wf_panel_2_time=df_skyfall_data[audio_epoch_s_label][station],
                                   wf_panel_1_sig=df_skyfall_data[accelerometer_data_raw_label][station][2],
                                   wf_panel_1_time=df_skyfall_data[accelerometer_epoch_s_label][station],
                                   wf_panel_0_sig=baro_height_from_bounder_km,
                                   wf_panel_0_time=df_skyfall_data[barometer_epoch_s_label][station],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Mic, Norm",
                                   wf_panel_1_units="Raw Acc Z, m/$s^2$",
                                   wf_panel_0_units="Height (/Bar), km",
                                   figure_title=EVENT_NAME)

        if gyroscope_data_raw_label and gyroscope_fs_label and gyroscope_data_highpass_label\
                in df_skyfall_data.columns:

            if use_parquet:
                # Reshape wf columns
                df_column_unflatten(df=df_skyfall_data,
                                    col_wf_label=gyroscope_data_raw_label,
                                    col_ndim_label=gyroscope_data_raw_label + "_ndim")

                df_column_unflatten(df=df_skyfall_data,
                                    col_wf_label=gyroscope_data_highpass_label,
                                    col_ndim_label=gyroscope_data_highpass_label + "_ndim")

            print('gyroscope_sample_rate_hz:', df_skyfall_data[gyroscope_fs_label][station])
            print('gyroscope_epoch_s_0:', df_skyfall_data[gyroscope_epoch_s_label][station][0],
                  df_skyfall_data[gyroscope_epoch_s_label][station][-1])
            # Plot raw aligned waveforms
            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=df_skyfall_data[gyroscope_data_raw_label][station][2],
                                   wf_panel_2_time=df_skyfall_data[gyroscope_epoch_s_label][station],
                                   wf_panel_1_sig=df_skyfall_data[gyroscope_data_raw_label][station][1],
                                   wf_panel_1_time=df_skyfall_data[gyroscope_epoch_s_label][station],
                                   wf_panel_0_sig=df_skyfall_data[gyroscope_data_raw_label][station][0],
                                   wf_panel_0_time=df_skyfall_data[gyroscope_epoch_s_label][station],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Z, rad/s",
                                   wf_panel_1_units="Y, rad/s",
                                   wf_panel_0_units="X, rad/s",
                                   figure_title=EVENT_NAME + ": Gyroscope raw")

            # Plot highpass aligned waveforms
            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=df_skyfall_data[gyroscope_data_highpass_label][station][2],
                                   wf_panel_2_time=df_skyfall_data[gyroscope_epoch_s_label][station],
                                   wf_panel_1_sig=df_skyfall_data[gyroscope_data_highpass_label][station][1],
                                   wf_panel_1_time=df_skyfall_data[gyroscope_epoch_s_label][station],
                                   wf_panel_0_sig=df_skyfall_data[gyroscope_data_highpass_label][station][0],
                                   wf_panel_0_time=df_skyfall_data[gyroscope_epoch_s_label][station],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Z, rad/s",
                                   wf_panel_1_units="Y, rad/s",
                                   wf_panel_0_units="X, rad/s",
                                   figure_title=EVENT_NAME + ": Gyroscope highpass")

        if magnetometer_data_raw_label and magnetometer_fs_label and magnetometer_data_highpass_label\
                in df_skyfall_data.columns:
            if use_parquet:
                # Reshape wf columns
                df_column_unflatten(df=df_skyfall_data,
                                    col_wf_label=magnetometer_data_raw_label,
                                    col_ndim_label=magnetometer_data_raw_label + "_ndim")
                df_column_unflatten(df=df_skyfall_data,
                                    col_wf_label=magnetometer_data_highpass_label,
                                    col_ndim_label=magnetometer_data_highpass_label + "_ndim")

            print('magnetometer_sample_rate_hz:', df_skyfall_data[magnetometer_fs_label][station])
            print('magnetometer_epoch_s_0:', df_skyfall_data[magnetometer_epoch_s_label][station][0],
                  df_skyfall_data[magnetometer_epoch_s_label][station][-1])
            # Plot aligned raw waveforms
            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=df_skyfall_data[magnetometer_data_raw_label][station][2],
                                   wf_panel_2_time=df_skyfall_data[magnetometer_epoch_s_label][station],
                                   wf_panel_1_sig=df_skyfall_data[magnetometer_data_raw_label][station][1],
                                   wf_panel_1_time=df_skyfall_data[magnetometer_epoch_s_label][station],
                                   wf_panel_0_sig=df_skyfall_data[magnetometer_data_raw_label][station][0],
                                   wf_panel_0_time=df_skyfall_data[magnetometer_epoch_s_label][station],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Z, $\mu$T",
                                   wf_panel_1_units="Y, $\mu$T",
                                   wf_panel_0_units="X, $\mu$T",
                                   figure_title=EVENT_NAME + ": Magnetometer raw")

            # Plot aligned highpass waveforms
            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=df_skyfall_data[magnetometer_data_highpass_label][station][2],
                                   wf_panel_2_time=df_skyfall_data[magnetometer_epoch_s_label][station],
                                   wf_panel_1_sig=df_skyfall_data[magnetometer_data_highpass_label][station][1],
                                   wf_panel_1_time=df_skyfall_data[magnetometer_epoch_s_label][station],
                                   wf_panel_0_sig=df_skyfall_data[magnetometer_data_highpass_label][station][0],
                                   wf_panel_0_time=df_skyfall_data[magnetometer_epoch_s_label][station],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Z, $\mu$T",
                                   wf_panel_1_units="Y, $\mu$T",
                                   wf_panel_0_units="X, $\mu$T",
                                   figure_title=EVENT_NAME + ": Magnetometer highpass")

        if location_latitude_label and location_longitude_label and location_altitude_label and location_speed_label \
                in df_skyfall_data.columns:
            # Range vs reference lat lon
            location_latitude_reference = 35.83728684
            location_longitude_reference = -115.57228988
            print("LAT LON at landing:", location_latitude_reference, location_longitude_reference)
            range_lat = (df_skyfall_data[location_latitude_label][station] - location_latitude_reference) \
                        * rpd_scales.DEGREES_TO_M
            range_lon = (df_skyfall_data[location_longitude_label][station] - location_longitude_reference) \
                        * rpd_scales.DEGREES_TO_M
            range_m = np.sqrt(np.array(range_lat**2 + range_lon**2).astype(np.float64))

            # TODO, TYLER: ADDRESS AND PROVIDE RECOMMENDATIONS
            # location_provider = station.location_sensor().get_data_channel("location_provider")
            # location_bearing = station.location_sensor().get_channel("bearing")
            # print('Why does bearing only have nans?', location_bearing)
            # print('Why does vertical accuracy only have nans?', location_vertical_accuracy)

            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=range_m,
                                   wf_panel_2_time=df_skyfall_data[location_epoch_s_label][station],
                                   wf_panel_1_sig=df_skyfall_data[location_altitude_label][station],
                                   wf_panel_1_time=df_skyfall_data[location_epoch_s_label][station],
                                   wf_panel_0_sig=df_skyfall_data[location_speed_label][station],
                                   wf_panel_0_time=df_skyfall_data[location_epoch_s_label][station],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Range, m",
                                   wf_panel_1_units="Altitude, m",
                                   wf_panel_0_units="Speed, m/s",
                                   figure_title=EVENT_NAME + ": Location Framework")

        if health_battery_charge_label and health_internal_temp_deg_C_label and health_network_type_label \
                and barometer_data_raw_label in df_skyfall_data.columns:

            # TODO, ANTHONY: How to handle these
            # print("\nLocation provider:", location_provider)
            print(f"network_type_epoch_s_0:", df_skyfall_data[health_network_type_label][station][0],
                  f", network_type_epoch_s_end:", df_skyfall_data[health_network_type_label][station][-1])

            # Other interesting fields: Estimated Height ASL, Internal Temp, % Battery
            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=barometer_height_m,
                                   wf_panel_2_time=df_skyfall_data[barometer_epoch_s_label][station],
                                   wf_panel_1_sig=df_skyfall_data[health_internal_temp_deg_C_label][station],
                                   wf_panel_1_time=df_skyfall_data[health_epoch_s_label][station],
                                   wf_panel_0_sig=df_skyfall_data[health_battery_charge_label][station],
                                   wf_panel_0_time=df_skyfall_data[health_epoch_s_label][station],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Baro Z, m",
                                   wf_panel_1_units="Temp C",
                                   wf_panel_0_units="Battery %",
                                   figure_title=EVENT_NAME + ": Station Status")

        # TODO: Convert synchronization to a sensor object. In fact, convert everything to a sensor object.
        # TODO: Get number of synch exchanges per packet
        # synchronization = station.timesync_analysis
        # synchronization_epoch_s = synchronization.get_start_times() * rpd_scales.MICROS_TO_S
        # synchronization_latency_ms = synchronization.get_latencies() * rpd_scales.MICROS_TO_MILLIS
        # synchronization_offset_ms = synchronization.get_offsets() * rpd_scales.MICROS_TO_MILLIS
        # synchronization_best_offset_ms = synchronization.get_best_offset() * rpd_scales.MICROS_TO_MILLIS
        # synchronization_offset_delta_ms = synchronization_offset_ms - synchronization_best_offset_ms
        # # TODO, TYLER: Get number of synch exchanges per packet as a time series
        # synchronization_number_exchanges = synchronization.timesync_data[0].num_tri_messages()
        #
        # pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
        #                        wf_panel_2_sig=synchronization_latency_ms,
        #                        wf_panel_2_time=synchronization_epoch_s,
        #                        wf_panel_1_sig=synchronization_offset_ms,
        #                        wf_panel_1_time=synchronization_epoch_s,
        #                        wf_panel_0_sig=synchronization_offset_delta_ms,
        #                        wf_panel_0_time=synchronization_epoch_s,
        #                        start_time_epoch=event_reference_time_epoch_s,
        #                        wf_panel_2_units="Latency, ms",
        #                        wf_panel_1_units="Offset, s",
        #                        wf_panel_0_units="Offset delta, s",
        #                        figure_title=EVENT_NAME + ": Synchronization Framework")
        # TODO: Address nan padding on location framework

        if health_internal_temp_deg_C_label and barometer_data_raw_label and location_altitude_label \
                in df_skyfall_data.columns:

            # Finally, barometric and altitude estimates
            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=barometer_height_m,
                                   wf_panel_2_time=df_skyfall_data[barometer_epoch_s_label][station],
                                   wf_panel_1_sig=df_skyfall_data[location_altitude_label][station],
                                   wf_panel_1_time=df_skyfall_data[location_epoch_s_label][station],
                                   wf_panel_0_sig=df_skyfall_data[health_internal_temp_deg_C_label][station],
                                   wf_panel_0_time=df_skyfall_data[health_epoch_s_label][station],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Baro Height, m",
                                   wf_panel_1_units="Loc Height, m",
                                   wf_panel_0_units="Temp C",
                                   figure_title=EVENT_NAME + ": Height and Temperature")
        # plt.show()

        sensor_column_label_list = [audio_data_label, barometer_data_highpass_label,
                                    accelerometer_data_highpass_label, gyroscope_data_highpass_label, magnetometer_data_highpass_label]

        sensor_epoch_column_label_list = [audio_epoch_s_label, barometer_epoch_s_label,
                                          accelerometer_epoch_s_label, gyroscope_epoch_s_label, magnetometer_epoch_s_label]

        sensor_ticklabels_list = ['Audio', 'Baro highpass', 'Acc X highpass', 'Acc Y highpass',
                                  'Acc Z highpass', 'Gyro X highpass', 'Gyro Y highpass', 'Gyro Z highpass',
                                  'Mag X highpass', 'Mag Y highpass', 'Mag Z highpass']

        rpd_plot.plot_sensor_wiggles_pandas(df=df_skyfall_data,
                                            station_id_str='1637610021',
                                            sensor_wf_label_list=sensor_column_label_list,
                                            sensor_timestamps_label_list=sensor_epoch_column_label_list,
                                            sig_id_label='station_id',
                                            x_label='Time',
                                            y_label='Sensor',
                                            fig_title='sensor waveforms',
                                            wf_color='midnightblue',
                                            sensor_yticks_label_list=sensor_ticklabels_list)

        plt.show()
