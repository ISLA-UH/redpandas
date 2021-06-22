"""
This module contains functions to integrate, and apply complimentary filters for phone orientation

Last updated: 10 June 2021
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from typing import List, Tuple


def remove_dc_offset(sensor_wf: np.ndarray, start_loc: int, end_loc: int) -> np.ndarray:
    """
    removes "DC offset" from the data by subtracting the mean of the specified subsection of the data.
    :param sensor_wf: data to remove the "DC offset"
    :param start_loc: location of the start of the DC offset subset
    :param end_loc: location of the end of the DC offset subset
    :return: data with DC offset removed
    """
    return sensor_wf - np.nanmean(sensor_wf[start_loc:end_loc])


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


def get_roll_pitch_yaw(accel_x: float, accel_y: float, accel_z: float) -> Tuple[float, float, float]:
    """
    Returns the pitch (rotation around y axis) and roll (rotation around x axis) from accelerometer data
    http://www.geekmomprojects.com/gyroscopes-and-accelerometers-on-a-chip/
    :param accel_x: x-axis acceleration value
    :param accel_y: y-axis acceleration value
    :param accel_z: z-axis acceleration value
    :return: pitch, roll
    """
    # get angle in radians
    roll = np.arctan2(accel_y, np.sqrt(accel_x * accel_x + accel_z * accel_z))
    pitch = np.arctan2(-accel_x, np.sqrt(accel_y * accel_y + accel_z * accel_z))
    yaw = np.arctan2(np.sqrt(accel_x * accel_x + accel_y * accel_y), accel_z)

    # convert to degrees
    return np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)


def get_roll_pitch_yaw_array(accelerometers: List) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the pitch (rotation around y axis) and roll (rotation around x axis) array from accelerometer data
    Loops through the get_pitch_and_roll function
    :param accelerometers: List of the xyz components of accelerometer data
    :return: pitch_array, roll_array
    """
    # Loop through get_xy_rotation
    roll_array = []
    pitch_array = []
    yaw_array = []

    for i in range(len(accelerometers[0])):
        pitch, roll, yaw = get_roll_pitch_yaw(accel_x=accelerometers[0][i],
                                              accel_y=accelerometers[1][i],
                                              accel_z=accelerometers[2][i])

        roll_array.append(roll)
        pitch_array.append(pitch)
        yaw_array.append(yaw)

    return np.array(roll_array), np.array(pitch_array), np.array(yaw_array)


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
