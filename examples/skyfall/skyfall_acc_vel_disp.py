# todo: address possible invalid values in building plots section
# Python libraries
import os.path
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumtrapz, cumulative_trapezoid
from obspy.signal.filter import highpass, lowpass
from scipy.spatial.distance import euclidean
from typing import List, Tuple

# RedVox RedPandas and related RedVox modules
from redvox.common.data_window import DataWindow
from redvox import settings
import redvox.common.date_time_utils as dt
import redpandas.redpd_preprocess as rpd_prep
import redpandas.redpd_scales as rpd_scales
import redpandas.redpd_build_station as rpd_build_sta
import redpandas.redpd_plot as rpd_plot
from libquantum.plot_templates import plot_time_frequency_reps as pnl

# Configuration file
from examples.skyfall.skyfall_config import EVENT_NAME, INPUT_DIR, OUTPUT_DIR, EPISODE_START_EPOCH_S, \
    EPISODE_END_EPOCH_S, STATIONS, DW_FILE, use_datawindow, use_pickle, use_parquet, PD_PQT_FILE, SENSOR_LABEL

# enable parallel
settings.set_parallelism_enabled(True)


def get_pitch_and_roll(accel_x: float, accel_y: float, accel_z: float) -> Tuple[float, float]:
    """
    Returns the pitch (rotation around y axis) and roll (rotation around x axis) from accelerometer data
    :param accel_x: x-axis acceleration value
    :param accel_y: y-axis acceleration value
    :param accel_z: z-axis acceleration value
    :return: pitch, roll
    """
    # get angle in radians
    pitch = np.arctan2(-accel_x, np.sqrt(accel_y * accel_y + accel_z * accel_z))
    roll = np.arctan2(accel_y, np.sqrt(accel_x * accel_x + accel_z * accel_z))

    # convert to degrees
    return np.rad2deg(pitch), np.rad2deg(roll)


def get_pitch_and_roll_array(accelerometers: List) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the pitch (rotation around y axis) and roll (rotation around x axis) array from accelerometer data
    Loops through the get_pitch_and_roll function
    :param accelerometers: List of the xyz components of accelerometer data
    :return: pitch_array, roll_array
    """
    # Loop through get_xy_rotation
    pitch_array = []
    roll_array = []

    for i in range(len(accelerometers[0])):
        pitch, roll = get_pitch_and_roll(accel_x=accelerometers[0][i],
                                         accel_y=accelerometers[1][i],
                                         accel_z=accelerometers[2][i])

        pitch_array.append(pitch)
        roll_array.append(roll)

    return np.array(pitch_array), np.array(roll_array)


def complimentary_filtering(accelerometer_angle: np.ndarray, gyroscope_angle: np.ndarray, alpha: float) -> np.ndarray:
    """
    Complimentary Filter for Accelereometer and Gyroscope.
    Returns filtered angle
    Based on the works from https://stackoverflow.com/questions/1586658/combine-gyroscope-and-accelerometer-data and
    http://blog.bitify.co.uk/2013/11/using-complementary-filter-to-combine.html
    :param accelerometer_angle: the calculated angle from the accelerometer (roll, pitch, yaw)
    :param gyroscope_angle: the calculated angle from the gyroscope (roll, pitch, yaw)
    :param alpha: determines the sensitivity of the accelerometer
    :return: roll_angle, pitch_angle
    """
    # Get the change in gyroscope angle initiate with zero
    gyroscope_angle_change = np.diff(gyroscope_angle)

    # Loop through the data to apply complimentary filter
    filtered_angle = gyroscope_angle
    for i in range(len(accelerometer_angle) - 1):
        filtered_angle[i + 1] = alpha * (filtered_angle[i] + gyroscope_angle_change[i]) \
                                + (1 - alpha) * accelerometer_angle[i + 1]

    return filtered_angle


def remove_DC_offset(sensor_wf: np.ndarray, start_loc: int, end_loc: int) -> np.ndarray:
    """
    removes "DC offset" from the data by subtracting the mean of the specified subsection of the data.
    :param sensor_wf: data to remove the "DC offset"
    :param start_loc: location of the start of the DC offset subset
    :param end_loc: location of the end of the DC offset subset
    :return: data with DC offset removed
    """
    return sensor_wf - np.mean(sensor_wf[start_loc:end_loc])


def integrate_cumtrapz(timestamps_s: np.ndarray, sensor_wf: np.ndarray, initial_value: float = 0) -> np.ndarray:
    """
    cumulative trapazoid integration using scipy.integrate.cumulative_trapezoid
    :param timestamps_s: timestamps corresponding to the data in seconds
    :param sensor_wf: data to integrate using cumulative trapezoid
    :param initial_value: the value to add in the initial of the integrated data to match length of input (default is 0)
    :return: integrated data with the same length as the input
    """

    integrated_data = cumulative_trapezoid(x=timestamps_s,
                                           y=sensor_wf,
                                           initial=initial_value)

    return integrated_data


def highpass_obspy(sensor_wf: np.ndarray, sample_rate_hz: float, frequency_low_hz: float, filter_order: int = 4) \
        -> np.ndarray:
    """
    applies a high pass filter using obspy.signal.filter.highpass with the specified low frequency
    :param sensor_wf: data to apply a high pass filter
    :param sample_rate_hz: sample rate of the data
    :param frequency_low_hz: cut off frequency for the high pass filter
    :param filter_order: the order of the filter (default = 4)
    :return: high passed data
    """
    return highpass(np.copy(sensor_wf),
                    frequency_low_hz,
                    sample_rate_hz, corners=filter_order,
                    zerophase=True)


def standardized_cumtrapz(timestamps_s: np.ndarray, sensor_wf: np.ndarray, dc_range: List,
                          cutoff_low_frequency: float = 1 / 100.) -> np.ndarray:
    """
    standard procedure to integrate accelerometer to velocity
    removes the DC offset, integrates, and applies a highpass filter
    :param timestamps_s: timestamps corresponding to the data in seconds
    :param sensor_wf: data to standardized integration
    :param dc_range: range of data to use for removing the dc offset
    :param cutoff_low_frequency:  cut off frequency for the high pass filter
    :return: standardized integrated and high passed data
    """
    # get pseudo sample rate
    pseudo_sample_rate = 1 / np.median(np.diff(timestamps_s))

    # remove the DC offset
    dc_removed_wf = remove_DC_offset(sensor_wf=sensor_wf, start_loc=dc_range[0], end_loc=dc_range[1])

    # integrate the waveform (initial value is the default (0)
    integrated_wf = integrate_cumtrapz(timestamps_s=timestamps_s, sensor_wf=dc_removed_wf)

    # apply the high pass filter
    return highpass_obspy(sensor_wf=integrated_wf,
                          sample_rate_hz=pseudo_sample_rate,
                          frequency_low_hz=cutoff_low_frequency)


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
    #stop = -1
    stop = -43700  # landfall

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
            print(f"Done. RedVox SDK version: {rdvx_data.sdk_version}")

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
                                   wf_panel_2_sig=df_skyfall_data[accelerometer_data_raw_label][station][2][:stop],
                                   wf_panel_2_time=df_skyfall_data[accelerometer_epoch_s_label][station][:stop],
                                   wf_panel_1_sig=df_skyfall_data[accelerometer_data_raw_label][station][1][:stop],
                                   wf_panel_1_time=df_skyfall_data[accelerometer_epoch_s_label][station][:stop],
                                   wf_panel_0_sig=df_skyfall_data[accelerometer_data_raw_label][station][0][:stop],
                                   wf_panel_0_time=df_skyfall_data[accelerometer_epoch_s_label][station][:stop],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Z, m/$s^2$",
                                   wf_panel_1_units="Y, m/$s^2$",
                                   wf_panel_0_units="X, m/$s^2$",
                                   figure_title=EVENT_NAME + ": Accelerometer raw")

            # remove DC offset
            accelerometer_time_s = df_skyfall_data[accelerometer_epoch_s_label][station][:stop]

            #accel_z = df_skyfall_data[accelerometer_data_raw_label][station][2][:stop]
            #accel_z = df_skyfall_data[accelerometer_data_highpass_label][station][2][:stop]
            accel_z = remove_DC_offset(sensor_wf=df_skyfall_data[accelerometer_data_raw_label][station][2][:stop],
                                       start_loc=0, end_loc=100)
            accel_y = remove_DC_offset(sensor_wf=df_skyfall_data[accelerometer_data_raw_label][station][1][:stop],
                                       start_loc=0, end_loc=100)
            accel_x = remove_DC_offset(sensor_wf=df_skyfall_data[accelerometer_data_raw_label][station][0][:stop],
                                       start_loc=0, end_loc=100)

            # high pass separately (to get rid of landfall effects) (don't high pass for z -> dragggg)
            pseudo_sample_rate = 1 / np.median(np.diff(accelerometer_time_s))

            # accel_z = lowpass(accel_z, 100, pseudo_sample_rate)
            # accel_z, _ = rpd_prep.highpass_from_diff(sig_wf=accel_z,
            #                                          sig_epoch_s=accelerometer_time_s,
            #                                          sample_rate_hz=pseudo_sample_rate)
            # accel_z = highpass_obspy(sensor_wf=accel_z, sample_rate_hz=pseudo_sample_rate, frequency_low_hz=1 / 100.)
            accel_y = highpass_obspy(sensor_wf=accel_y, sample_rate_hz=pseudo_sample_rate, frequency_low_hz=1 / 100.)
            accel_x = highpass_obspy(sensor_wf=accel_x, sample_rate_hz=pseudo_sample_rate, frequency_low_hz=1 / 100.)

            # Plot aligned high passed waveforms
            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=df_skyfall_data[accelerometer_data_highpass_label][station][2][:stop]
                                   - accel_z,
                                   wf_panel_2_time=accelerometer_time_s,
                                   wf_panel_1_sig=df_skyfall_data[accelerometer_data_highpass_label][station][2][:stop],
                                   wf_panel_1_time=accelerometer_time_s,
                                   wf_panel_0_sig=accel_z,
                                   wf_panel_0_time=accelerometer_time_s,
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="HP - KHP Z, m/$s^2$",
                                   wf_panel_1_units="HP Z, m/$s^2$",
                                   wf_panel_0_units="KHP Z, m/$s^2$",
                                   figure_title=EVENT_NAME + ": Accelerometer Z comparison")


            # Plot aligned high passed waveforms
            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=accel_z,
                                   wf_panel_2_time=accelerometer_time_s,
                                   wf_panel_1_sig=accel_y,
                                   wf_panel_1_time=accelerometer_time_s,
                                   wf_panel_0_sig=accel_x,
                                   wf_panel_0_time=accelerometer_time_s,
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Z, m/$s^2$",
                                   wf_panel_1_units="Y, m/$s^2$",
                                   wf_panel_0_units="X, m/$s^2$",
                                   figure_title=EVENT_NAME + ": Accelerometer high passed")

            # Convert to Velocity
            velocity_z = integrate_cumtrapz(timestamps_s=accelerometer_time_s,
                                            sensor_wf=accel_z)
            velocity_y = integrate_cumtrapz(timestamps_s=accelerometer_time_s,
                                            sensor_wf=accel_y)
            velocity_x = integrate_cumtrapz(timestamps_s=accelerometer_time_s,
                                            sensor_wf=accel_x)

            speed_xyz = np.sqrt(velocity_z ** 2 + velocity_y ** 2 + velocity_x ** 2)
            speed_z = np.abs(velocity_z)

            # integrate the velocity to get displacement
            disp_z = integrate_cumtrapz(timestamps_s=accelerometer_time_s,
                                        sensor_wf=velocity_z)
            disp_y = integrate_cumtrapz(timestamps_s=accelerometer_time_s,
                                        sensor_wf=velocity_y)
            disp_x = integrate_cumtrapz(timestamps_s=accelerometer_time_s,
                                        sensor_wf=velocity_x)

            disp_xyz = np.sqrt(disp_x ** 2 + disp_y ** 2 + disp_z ** 2)

            # print(pd.to_datetime(df_skyfall_data[accelerometer_epoch_s_label][station][40000], unit='s'))
            # Plot vel
            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=velocity_z,
                                   wf_panel_2_time=accelerometer_time_s,
                                   wf_panel_1_sig=velocity_y,
                                   wf_panel_1_time=accelerometer_time_s,
                                   wf_panel_0_sig=velocity_x,
                                   wf_panel_0_time=accelerometer_time_s,
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Z, m/s",
                                   wf_panel_1_units="Y, m/s",
                                   wf_panel_0_units="X, m/s",
                                   figure_title=EVENT_NAME + ": Velocity (demeaned from start values)")

            # Plot disp
            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=disp_z,
                                   wf_panel_2_time=accelerometer_time_s,
                                   wf_panel_1_sig=disp_y,
                                   wf_panel_1_time=accelerometer_time_s,
                                   wf_panel_0_sig=disp_x,
                                   wf_panel_0_time=accelerometer_time_s,
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
                                   wf_panel_2_sig=df_skyfall_data[gyroscope_data_raw_label][station][2][:stop],
                                   wf_panel_2_time=df_skyfall_data[gyroscope_epoch_s_label][station][:stop],
                                   wf_panel_1_sig=df_skyfall_data[gyroscope_data_raw_label][station][1][:stop],
                                   wf_panel_1_time=df_skyfall_data[gyroscope_epoch_s_label][station][:stop],
                                   wf_panel_0_sig=df_skyfall_data[gyroscope_data_raw_label][station][0][:stop],
                                   wf_panel_0_time=df_skyfall_data[gyroscope_epoch_s_label][station][:stop],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Z, rad/s",
                                   wf_panel_1_units="Y, rad/s",
                                   wf_panel_0_units="X, rad/s",
                                   figure_title=EVENT_NAME + ": Gyroscope raw")


            # clean gyro
            gyro_time_s = df_skyfall_data[gyroscope_epoch_s_label][station][:stop]
            gyro_sample_rate = 1 / np.median(np.diff(gyro_time_s))

            gyro_z = remove_DC_offset(sensor_wf=df_skyfall_data[gyroscope_data_raw_label][station][2][:stop],
                                      start_loc=0,
                                      end_loc=100)
            gyro_y = remove_DC_offset(sensor_wf=df_skyfall_data[gyroscope_data_raw_label][station][1][:stop],
                                      start_loc=0,
                                      end_loc=100)
            gyro_x = remove_DC_offset(sensor_wf=df_skyfall_data[gyroscope_data_raw_label][station][0][:stop],
                                      start_loc=0,
                                      end_loc=100)

            gyro_z = highpass_obspy(sensor_wf=gyro_z, sample_rate_hz=gyro_sample_rate, frequency_low_hz=1/100)
            gyro_y = highpass_obspy(sensor_wf=gyro_y, sample_rate_hz=gyro_sample_rate, frequency_low_hz=1/100)
            gyro_x = highpass_obspy(sensor_wf=gyro_x, sample_rate_hz=gyro_sample_rate, frequency_low_hz=1/100)

            # Plot highpass aligned waveforms
            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=gyro_z,
                                   wf_panel_2_time=gyro_time_s,
                                   wf_panel_1_sig=gyro_y,
                                   wf_panel_1_time=gyro_time_s,
                                   wf_panel_0_sig=gyro_x,
                                   wf_panel_0_time=gyro_time_s,
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Z, rad/s",
                                   wf_panel_1_units="Y, rad/s",
                                   wf_panel_0_units="X, rad/s",
                                   figure_title=EVENT_NAME + ": Gyroscope highpass")

            gyro_rad_z = integrate_cumtrapz(timestamps_s=gyro_time_s, sensor_wf=gyro_z)
            gyro_rad_y = integrate_cumtrapz(timestamps_s=gyro_time_s, sensor_wf=gyro_y)
            gyro_rad_x = integrate_cumtrapz(timestamps_s=gyro_time_s, sensor_wf=gyro_x)


            # Plot raw aligned waveforms
            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=gyro_rad_z,
                                   wf_panel_2_time=gyro_time_s,
                                   wf_panel_1_sig=gyro_rad_y,
                                   wf_panel_1_time=gyro_time_s,
                                   wf_panel_0_sig=gyro_rad_x,
                                   wf_panel_0_time=gyro_time_s,
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
            range_lat = (mask_lat - location_latitude_reference) * rpd_scales.DEGREES_TO_METERS
            # range_lon = (df_skyfall_data[location_longitude_label][station] - location_longitude_reference) \
            #             * rpd_scales.DEGREES_TO_M
            range_lon = (mask_long - location_longitude_reference) * rpd_scales.DEGREES_TO_METERS

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

            acceleration_list = [accel_x,
                                 accel_y,
                                 accel_z]

            # acceleration_list = [acc_adj_x, acc_adj_y, acc_adj_z]

            raw_pitch, raw_roll = get_pitch_and_roll_array(accelerometers=acceleration_list)

            # Complimentary filter
            comp_pitch = complimentary_filtering(accelerometer_angle=raw_pitch,
                                                 gyroscope_angle=np.rad2deg(gyro_rad_y),
                                                 alpha=0.95)

            comp_roll = complimentary_filtering(accelerometer_angle=raw_roll,
                                                gyroscope_angle=np.rad2deg(gyro_rad_x),
                                                alpha=0.95)

            # (gyro_rad_x + np.pi) % (2 * np.pi) - np.pi
            f, ax = plt.subplots(nrows=2)
            ax[0].plot(accelerometer_time_s, raw_roll)
            ax[0].plot(gyro_time_s, np.rad2deg(gyro_rad_x), alpha=0.6)
            ax[0].plot(gyro_time_s, comp_roll, alpha=0.6)
            ax[0].set_title("roll (deg)")

            ax[1].plot(accelerometer_time_s, raw_pitch)
            ax[1].plot(gyro_time_s, np.rad2deg(gyro_rad_y), alpha=0.6)
            ax[1].plot(gyro_time_s, comp_pitch, alpha=0.6)
            ax[1].set_title("pitch (deg)")

            # Check Speed
            # plt.figure()
            # plt.plot(df_skyfall_data[accelerometer_epoch_s_label][station],
            #          df_skyfall_data[accelerometer_data_highpass_label][station][2], 'C1')
            # plt.vlines(x=df_skyfall_data[accelerometer_epoch_s_label][station][end_ind], ymin=-1e2, ymax=1e2)

            f, ax = plt.subplots(nrows=2, sharex=True)
            ax[0].plot(accelerometer_time_s, speed_z)
            ax[1].plot(df_skyfall_data[location_epoch_s_label][station], mask_speed)

            f, ax = plt.subplots(nrows=2, sharex=True)
            ax[0].plot(accelerometer_time_s, np.abs(disp_z - disp_z[-1]))
            ax[1].plot(df_skyfall_data[location_epoch_s_label][station], mask_alt)

            plt.show()
