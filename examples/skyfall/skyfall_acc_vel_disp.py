# todo: address possible invalid values in building plots section
# Python libraries
import os.path
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumtrapz
from scipy.spatial.distance import euclidean
from typing import List, Tuple

# RedVox RedPandas and related RedVox modules
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


def get_xy_rotation(accel_x: float, accel_y: float, accel_z: float) -> Tuple[float, float]:
    """
    Returns the x rotation (roll angle) and y rotation (pitch angle) from accelerometer data
    :param accel_x: x-axis acceleration value
    :param accel_y: y-axis acceleration value
    :param accel_z: z-axis acceleration value
    :return: x_rotation, y_rotation
    """
    # get angle in radians
    x_radians = np.arctan2(accel_x, euclidean(accel_y, accel_z))
    y_radians = np.arctan2(accel_y, euclidean(accel_x, accel_z))

    # convert to degrees
    return np.rad2deg((x_radians + np.pi) % (2 * np.pi) - np.pi), np.rad2deg((y_radians + np.pi) % (2 * np.pi) - np.pi)


def get_xy_rotation_array(accelerometers: List) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the x rotation (roll angle) and y rotation (pitch angle) array from accelerometer data
    Loops through the get_xy_rotation function
    :param accelerometers: List of the xyz components of accelerometer data
    :return: x_rotation_array, y_rotation_array
    """
    # Loop through get_xy_rotation
    x_rotation_array = []
    y_rotation_array = []

    for i in range(len(accelerometers[0])):
        x_rot, y_rot = get_xy_rotation(accel_x=accelerometers[0][i],
                                       accel_y=accelerometers[1][i],
                                       accel_z=accelerometers[2][i])

        x_rotation_array.append(x_rot)
        y_rotation_array.append(y_rot)

    return np.array(x_rotation_array), np.array(y_rotation_array)


def complimentary_filtering(accelerometers: List, gyroscope: List, new_rate: float, update_time: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Complimentary Filter for Accelereometer and Gyroscope.
    Returns roll angle (rotation around X-axis) and pitch angle (rotation around Y-axis).
    Based on the works from https://stackoverflow.com/questions/1586658/combine-gyroscope-and-accelerometer-data and
    http://blog.bitify.co.uk/2013/11/using-complementary-filter-to-combine.html
    :param accelerometers: List of the xyz components of accelerometer data
    :param gyroscope: List of the xyz components of the gyroscope data
    :param new_rate: new sample rate that determines the sensitivity to the accelerometer.
    :param update_time: Time between accelerometer / gyroscope samples
    :return: roll_angle, pitch_angle
    """


if __name__ == "__main__":

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

    # Location columns
    location_latitude_label: str = 'location_latitude'
    location_longitude_label: str = 'location_longitude'
    location_altitude_label: str = "location_altitude"
    location_speed_label: str = 'location_speed'
    location_epoch_s_label: str = 'location_epoch_s'
    location_provider_label: str = 'location_provider'

    # stop time (until stop or full)
    end_ind = -1        # full
    #end_ind = -43500    # landfall

    # Load data options
    if use_datawindow is True or use_pickle is True:
        print("Initiating Conversion from RedVox DataWindow to RedVox RedPandas:")
        if use_datawindow:  # Option A: Create DataWindow object
            print("Constructing RedVox DataWindow Fast...", end=" ")
            rdvx_data = DataWindowFast(input_dir=INPUT_DIR,
                                       station_ids=STATIONS,
                                       start_datetime=dt.datetime_from_epoch_seconds_utc(EPISODE_START_EPOCH_S),
                                       end_datetime=dt.datetime_from_epoch_seconds_utc(EPISODE_END_EPOCH_S),
                                       apply_correction=True,
                                       structured_layout=True)
            print(f"Done. RedVox SDK version: {rdvx_data.sdk_version}")

        else:  # Option B: Load pickle with DataWindow object. Assume compressed
            print("Unpickling existing compressed RedVox DataWindow with JSON...", end=" ")
            rdvx_data: DataWindowFast = DataWindowFast.from_json_file(base_dir=OUTPUT_DIR, file_name=DW_FILE)
            print(f"Done. RedVox SDK version: {rdvx_data.sdk_version}")

        # For option A or B, begin RedPandas
        print("\nInitiating RedVox Redpandas:")
        df_skyfall_data = pd.DataFrame([rpd_build_sta.station_to_dict_from_dw(station=station,
                                                                              sdk_version=rdvx_data.sdk_version,
                                                                              sensor_labels=SENSOR_LABEL)
                                        for station in rdvx_data.stations])
        df_skyfall_data.sort_values(by="station_id", ignore_index=True, inplace=True)

    elif use_parquet:  # Option C: Open dataframe from parquet file
        print("Loading exisiting RedPandas Parquet...", end=" ")
        df_skyfall_data = pd.read_parquet(os.path.join(OUTPUT_DIR, PD_PQT_FILE))
        print(f"Done. RedVox SDK version: {df_skyfall_data[redvox_sdk_version_label][0]}")

    else:
        print('\nNo data loading method selected. '
              'Check that use_datawindow, use_pickle, or use_parquet in the Skyfall configuration file are set to True.')
        exit()

    # Get Data
    print("\nInitiating time-domain representation of Skyfall:")
    for station in df_skyfall_data.index:
        station_id_str = df_skyfall_data[station_label][station]  # Get the station id

        if audio_data_label and audio_fs_label in df_skyfall_data.columns:
            print('mic_sample_rate_hz: ', df_skyfall_data[audio_fs_label][station])
            print('mic_epoch_s_0: ', df_skyfall_data[audio_epoch_s_label][station][0])

            # Frame to mic start and end and plot
            event_reference_time_epoch_s = df_skyfall_data[audio_epoch_s_label][station][0]

        if accelerometer_data_raw_label and accelerometer_fs_label and accelerometer_data_highpass_label \
                in df_skyfall_data.columns:
            if use_parquet:
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

            # Now let's try doing some integrals
            vel_z = cumtrapz(x=df_skyfall_data[accelerometer_epoch_s_label][station][:end_ind],
                             y=df_skyfall_data[accelerometer_data_highpass_label][station][2][:end_ind] -
                               np.mean(df_skyfall_data[accelerometer_data_highpass_label][station][2][:40000]))
            # vel_y = cumtrapz(x=df_skyfall_data[accelerometer_epoch_s_label][station][:end_id],
            #                  y=df_skyfall_data[accelerometer_data_raw_label][station][1][:end_ind] -
            #                    np.mean(df_skyfall_data[accelerometer_data_raw_label][station][1][:40000]))
            # vel_x = cumtrapz(x=df_skyfall_data[accelerometer_epoch_s_label][station][:end_ind],
            #                  y=df_skyfall_data[accelerometer_data_raw_label][station][0][:end_ind] -
            #                    np.mean(df_skyfall_data[accelerometer_data_raw_label][station][0][:40000]))

            # vel_z = cumtrapz(x=df_skyfall_data[accelerometer_epoch_s_label][station][:-43500],
            #                  y=df_skyfall_data[accelerometer_data_raw_label][station][2][:-43500])
            vel_y = cumtrapz(x=df_skyfall_data[accelerometer_epoch_s_label][station][:end_ind],
                             y=df_skyfall_data[accelerometer_data_highpass_label][station][1][:end_ind])
            vel_x = cumtrapz(x=df_skyfall_data[accelerometer_epoch_s_label][station][:end_ind],
                             y=df_skyfall_data[accelerometer_data_highpass_label][station][0][:end_ind])

            speed_xyz = np.sqrt(vel_x ** 2 + vel_y ** 2 + vel_x ** 2)
            speed_z = np.abs(vel_z)

            # Now let's try doing some integrals
            disp_z = cumtrapz(x=df_skyfall_data[accelerometer_epoch_s_label][station][:end_ind-1],
                              y=vel_z)
            disp_y = cumtrapz(x=df_skyfall_data[accelerometer_epoch_s_label][station][:end_ind-1],
                              y=vel_y)
            disp_x = cumtrapz(x=df_skyfall_data[accelerometer_epoch_s_label][station][:end_ind-1],
                              y=vel_x)

            disp_xyz = np.sqrt(disp_x ** 2 + disp_y ** 2 + disp_z ** 2)

            # print(pd.to_datetime(df_skyfall_data[accelerometer_epoch_s_label][station][40000], unit='s'))

            # Plot vel
            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=vel_z,
                                   wf_panel_2_time=df_skyfall_data[accelerometer_epoch_s_label][station][:end_ind-1],
                                   wf_panel_1_sig=vel_y,
                                   wf_panel_1_time=df_skyfall_data[accelerometer_epoch_s_label][station][:end_ind-1],
                                   wf_panel_0_sig=vel_x,
                                   wf_panel_0_time=df_skyfall_data[accelerometer_epoch_s_label][station][:end_ind-1],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Z, m/s",
                                   wf_panel_1_units="Y, m/s",
                                   wf_panel_0_units="X, m/s",
                                   figure_title=EVENT_NAME + ": Velocity (demeaned from start values)")

            # Plot disp
            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=disp_z,
                                   wf_panel_2_time=df_skyfall_data[accelerometer_epoch_s_label][station][:end_ind-2],
                                   wf_panel_1_sig=disp_y,
                                   wf_panel_1_time=df_skyfall_data[accelerometer_epoch_s_label][station][:end_ind-2],
                                   wf_panel_0_sig=disp_x,
                                   wf_panel_0_time=df_skyfall_data[accelerometer_epoch_s_label][station][:end_ind-2],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Z, m",
                                   wf_panel_1_units="Y, m",
                                   wf_panel_0_units="X, m",
                                   figure_title=EVENT_NAME + ": Displacement (demeaned from start values)")

        if gyroscope_data_raw_label and gyroscope_fs_label and gyroscope_data_highpass_label \
                in df_skyfall_data.columns:

            if use_parquet:
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

            gyro_rad_z = cumtrapz(df_skyfall_data[gyroscope_data_highpass_label][station][2],
                                  df_skyfall_data[gyroscope_epoch_s_label][station])

            gyro_rad_y = cumtrapz(df_skyfall_data[gyroscope_data_highpass_label][station][1],
                                  df_skyfall_data[gyroscope_epoch_s_label][station])

            gyro_rad_x = cumtrapz(df_skyfall_data[gyroscope_data_highpass_label][station][0],
                                  df_skyfall_data[gyroscope_epoch_s_label][station])

            # Plot raw aligned waveforms
            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=gyro_rad_z,
                                   wf_panel_2_time=df_skyfall_data[gyroscope_epoch_s_label][station][:-1],
                                   wf_panel_1_sig=gyro_rad_y,
                                   wf_panel_1_time=df_skyfall_data[gyroscope_epoch_s_label][station][:-1],
                                   wf_panel_0_sig=gyro_rad_x,
                                   wf_panel_0_time=df_skyfall_data[gyroscope_epoch_s_label][station][:-1],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Z, rad",
                                   wf_panel_1_units="Y, rad",
                                   wf_panel_0_units="X, rad",
                                   figure_title=EVENT_NAME + ": Gyroscope total raw")

        if location_latitude_label and location_longitude_label and location_altitude_label and location_speed_label \
                in df_skyfall_data.columns:
            # apply mask to location stuff
            # location speed
            list_bool_speed = [True] + [False] * (len(df_skyfall_data[location_speed_label][0]) - 1)
            mask_speed = np.ma.masked_array(df_skyfall_data[location_speed_label][0], mask=list_bool_speed)
            # df_skyfall_data.loc[0, location_speed_label] = mx.tolist()
            # location latitude
            list_bool_lat = [True] + [False] * (len(df_skyfall_data[location_latitude_label][0]) - 1)
            mask_lat = np.ma.masked_array(df_skyfall_data[location_latitude_label][0], mask=list_bool_lat)
            # df_skyfall_data.loc[0, location_latitude_label] = mx.tolist()
            # location longitude
            list_bool_long = [True] + [False] * (len(df_skyfall_data[location_longitude_label][0]) - 1)
            mask_long = np.ma.masked_array(df_skyfall_data[location_longitude_label][0], mask=list_bool_long)
            # location altitude
            list_bool_alt = [True] + [False] * (len(df_skyfall_data[location_altitude_label][0]) - 1)
            mask_alt = np.ma.masked_array(df_skyfall_data[location_altitude_label][0], mask=list_bool_alt)

            # Range vs reference lat lon
            location_latitude_reference = 35.83728684
            location_longitude_reference = -115.57228988
            print("LAT LON at landing:", location_latitude_reference, location_longitude_reference)
            # range_lat = (df_skyfall_data[location_latitude_label][station] - location_latitude_reference) \
            #             * rpd_scales.DEGREES_TO_M
            range_lat = (mask_lat - location_latitude_reference) * rpd_scales.DEGREES_TO_M
            # range_lon = (df_skyfall_data[location_longitude_label][station] - location_longitude_reference) \
            #             * rpd_scales.DEGREES_TO_M
            range_lon = (mask_long - location_longitude_reference) * rpd_scales.DEGREES_TO_M

            range_m = np.sqrt(np.array(range_lat ** 2 + range_lon ** 2).astype(np.float64))
            list_bool_range = [True] + [False] * (len(range_m) - 1)
            range_m = np.ma.masked_array(df_skyfall_data[location_altitude_label][0], mask=list_bool_range)

            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=range_m,
                                   wf_panel_2_time=df_skyfall_data[location_epoch_s_label][station],
                                   # wf_panel_1_sig=df_skyfall_data[location_altitude_label][station],
                                   wf_panel_1_sig=mask_alt,
                                   wf_panel_1_time=df_skyfall_data[location_epoch_s_label][station],
                                   # wf_panel_0_sig=df_skyfall_data[location_speed_label][station],
                                   wf_panel_0_sig=mask_speed,
                                   wf_panel_0_time=df_skyfall_data[location_epoch_s_label][station],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Range, m",
                                   wf_panel_1_units="Altitude, m",
                                   wf_panel_0_units="Speed, m/s",
                                   figure_title=EVENT_NAME + ": Location Framework")

            # Check rotation
            acceleration_list = [df_skyfall_data[accelerometer_data_highpass_label][station][0],
                                 df_skyfall_data[accelerometer_data_highpass_label][station][1],
                                 df_skyfall_data[accelerometer_data_highpass_label][station][2]]

            raw_x_rot, raw_y_rot = get_xy_rotation_array(accelerometers=acceleration_list)

            # (gyro_rad_x + np.pi) % (2 * np.pi) - np.pi
            f, ax = plt.subplots(nrows=2)
            ax[0].plot(df_skyfall_data[accelerometer_epoch_s_label][station], raw_x_rot)
            ax[0].plot(df_skyfall_data[gyroscope_epoch_s_label][station][:-1],
                       np.rad2deg(gyro_rad_x), alpha=0.6)

            ax[1].plot(df_skyfall_data[accelerometer_epoch_s_label][station], raw_y_rot)
            ax[1].plot(df_skyfall_data[gyroscope_epoch_s_label][station][:-1],
                       np.rad2deg(gyro_rad_y), alpha=0.6)

            # Check Speed
            # plt.figure()
            # plt.plot(df_skyfall_data[accelerometer_epoch_s_label][station],
            #          df_skyfall_data[accelerometer_data_highpass_label][station][2], 'C1')
            # plt.vlines(x=df_skyfall_data[accelerometer_epoch_s_label][station][end_ind], ymin=-1e2, ymax=1e2)

            f, ax = plt.subplots(nrows=2, sharex=True)
            ax[0].plot(df_skyfall_data[accelerometer_epoch_s_label][station][:end_ind-1], speed_z)
            ax[1].plot(df_skyfall_data[location_epoch_s_label][station], mask_speed)

            f, ax = plt.subplots(nrows=2, sharex=True)
            ax[0].plot(df_skyfall_data[accelerometer_epoch_s_label][station][:end_ind-2], np.abs(disp_z - disp_z[-1]))
            ax[1].plot(df_skyfall_data[location_epoch_s_label][station], mask_alt)

            plt.show()
