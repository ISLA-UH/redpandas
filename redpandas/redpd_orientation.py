"""
Functions to integrate, and apply complimentary filters for phone orientation.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from typing import List, Tuple


def remove_dc_offset(sensor_wf: np.ndarray, start_loc: int = None, end_loc: int = None) -> np.ndarray:
    """
    removes "DC offset" from the data by subtracting the mean of the specified subsection of the data.
    If start and end location is None, it uses the whole array, if one is given it will take the other to the max.

    :param sensor_wf: data to remove the "DC offset"
    :param start_loc: location of the start of the DC offset subset
    :param end_loc: location of the end of the DC offset subset
    :return: data with DC offset removed
    """
    if start_loc and end_loc is None:
        removed = np.nanmean(sensor_wf)
    elif start_loc is None:
        removed = np.nanmean(sensor_wf[:end_loc])
    elif end_loc is None:
        removed = np.nanmean(sensor_wf[start_loc:])
    else:
        removed = np.nanmean(sensor_wf[start_loc:end_loc])
    return sensor_wf - removed


def remove_dc_offset_s(timestamps_s: np.ndarray, sensor_wf: np.ndarray,
                       start_s: int = None, end_s: int = None) -> np.ndarray:
    """
    removes "DC offset" from the data by subtracting the mean of the specified subsection of the data.
    If start and end time is None, it uses the whole array, if one is given it will take the other to the max.

    :param timestamps_s: timestamps corresponding to the data in seconds
    :param sensor_wf: data to remove the "DC offset"
    :param start_s: seconds from the first timestamp to use as the start of the range for the DC offset subset
    :param end_s: seconds from the first timestamp to use as the end of the range for the DC offset subset
    :return: data with DC offset removed
    """
    # adjust timestamps to be relative from the start
    timestamps_s_adj = timestamps_s - timestamps_s[0]

    # find location closest to the given start and end
    if start_s and end_s is None:
        start_loc = None
        end_loc = None
    elif start_s is None:
        start_loc = None
        end_loc = np.abs(timestamps_s_adj - end_s).argmin()
    elif end_s is None:
        start_loc = np.abs(timestamps_s_adj - start_s).argmin()
        end_loc = None
    else:
        start_loc = np.abs(timestamps_s_adj - start_s).argmin()
        end_loc = np.abs(timestamps_s_adj - end_s).argmin()

    # use remove_dc_offset to find the offset
    return remove_dc_offset(sensor_wf=sensor_wf, start_loc=start_loc, end_loc=end_loc)


def integrate_cumtrapz(timestamps_s: np.ndarray, sensor_wf: np.ndarray, initial_value: float = 0) -> np.ndarray:
    """
    cumulative trapazoid integration using scipy.integrate.cumulative_trapezoid

    :param timestamps_s: timestamps corresponding to the data in seconds
    :param sensor_wf: data to integrate using cumulative trapezoid
    :param initial_value: the value to add in the initial of the integrated data to match length of input (default is 0)
    :return: integrated data with the same length as the input
    """
    return cumulative_trapezoid(x=timestamps_s, y=sensor_wf, initial=initial_value)


def get_roll_pitch(accel_x: float, accel_y: float, accel_z: float) -> Tuple[float, float]:
    """
    Returns the pitch (rotation around y-axis) and roll (rotation around x-axis) from accelerometer data
    http://www.geekmomprojects.com/gyroscopes-and-accelerometers-on-a-chip/

    :param accel_x: x-axis acceleration value
    :param accel_y: y-axis acceleration value
    :param accel_z: z-axis acceleration value
    :return: pitch, roll
    """
    # get angle in radians
    # convert to degrees
    return np.arctan2(accel_y, np.sqrt(accel_x * accel_x + accel_z * accel_z)), \
        np.arctan2(-accel_x, np.sqrt(accel_y * accel_y + accel_z * accel_z))


def get_yaw(roll: float, pitch: float, mag_x: float, mag_y: float, mag_z: float) -> np.ndarray:
    """
    Returns yaw based on roll / pitch data and the magnetometer data
    https://roboticsclubiitk.github.io/2017/12/21/Beginners-Guide-to-IMU.html

    :param roll: rotation around the x-axis
    :param pitch: rotation around the y-axis
    :param mag_x: x-axis magnetometer value
    :param mag_y: y-axis magnetometer value
    :param mag_z: z-axis magnetometer value
    :return: yaw
    """
    return np.arctan2(-mag_y * np.cos(roll) - mag_z*np.sin(roll),
                      mag_x*np.cos(pitch) + mag_y*np.sin(roll)*np.sin(pitch) + mag_z*np.cos(roll)*np.sin(pitch))


def get_roll_pitch_array(accelerometers: List) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the pitch (rotation around y-axis) and roll (rotation around x-axis) array from accelerometer data
    Loops through the get_pitch_and_roll function

    :param accelerometers: List of the xyz components of accelerometer data
    :return: pitch_array, roll_array
    """
    # Loop through get_xy_rotation
    roll_array = []
    pitch_array = []
    for i in range(len(accelerometers[0])):
        pitch, roll = get_roll_pitch(accel_x=accelerometers[0][i],
                                     accel_y=accelerometers[1][i],
                                     accel_z=accelerometers[2][i])
        roll_array.append(roll)
        pitch_array.append(pitch)

    return np.array(roll_array), np.array(pitch_array)


def get_yaw_array(roll_array: np.ndarray, pitch_array: np.ndarray, magnetometers: List) -> np.ndarray:
    """
    Returns the yaw array from roll (rotation around x-axis), pitch (rotation around y-axis), and gyroscope data

    :param roll_array: roll (rotation around x-axis) calculated from sensors
    :param pitch_array: pitch (rotation around y-axis) calculated from sensors
    :param magnetometers: List of xyz components of magnetometer data
    :return: yaw_array
    """
    # Loop through get_xy_rotation
    yaw_array = []
    for i in range(len(magnetometers[0])):
        yaw = get_yaw(roll=roll_array[i],
                      pitch=pitch_array[i],
                      mag_x=magnetometers[0][i],
                      mag_y=magnetometers[1][i],
                      mag_z=magnetometers[2][i])
        yaw_array.append(yaw)

    return np.array(yaw_array)


def complimentary_filtering(gyroscope_time_s: np.ndarray, gyroscope_angle: np.ndarray,
                            accelerometer_angle: np.ndarray, smoothing_factor: float) -> np.ndarray:
    """
    Complimentary Filter for Accelerometer and Gyroscope.
    Returns filtered angle
    Based on the works from https://stackoverflow.com/questions/1586658/combine-gyroscope-and-accelerometer-data and
    http://blog.bitify.co.uk/2013/11/using-complementary-filter-to-combine.html

    :param gyroscope_time_s: timestamps corresponding to the gyroscope data in seconds
    :param gyroscope_angle: the calculated angle from the gyroscope (roll, pitch, yaw)
    :param accelerometer_angle: the calculated angle from the accelerometer (roll, pitch, yaw)
    :param smoothing_factor: determines the sensitivity of the accelerometer
    :return: filtered angle
    """
    # Get the change in gyroscope angle initiate with zero
    gyroscope_angle_change = np.diff(gyroscope_angle)
    gyroscope_time_delta = np.diff(gyroscope_time_s)

    # Loop through the data to apply complimentary filter
    filtered_angle = gyroscope_angle.copy()
    for i in range(len(accelerometer_angle) - 1):
        filtered_angle[i + 1] = \
            smoothing_factor * (filtered_angle[i] + gyroscope_angle_change[i] * gyroscope_time_delta[i]) \
            + smoothing_factor * accelerometer_angle[i + 1]

    return filtered_angle
