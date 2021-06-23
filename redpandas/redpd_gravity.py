"""
This module contains functions to apply exponential filter on accelerometer to separate gravity and linear acceleration
Based on the "past-gen" Android code without incorporation of the gyroscope
http://josejuansanchez.org/android-sensors-overview/gravity_and_linear_acceleration/README.html

Last updated: 22 June 2021
"""

import numpy as np
from typing import Tuple


def get_smoothing_factor(sensor_sample_rate_hz: float, low_pass_sample_rate_hz: float = 1) -> float:
    """

    :param sensor_sample_rate_hz:
    :param low_pass_sample_rate_hz:
    :return: alpha
    """
    # assuming sensor_sample_rate_hz << low_pass_sample_rate_hz
    return low_pass_sample_rate_hz / sensor_sample_rate_hz


def get_gravity(accelerometer: np.ndarray, smoothing_factor: float):
    """
    based on the slack thread: https://tinyurl.com/f6t3h2fp
    :param accelerometer:
    :param smoothing_factor:
    :return:
    """

    # initialize gravity array
    gravity = np.zeros(len(accelerometer)) * np.mean(accelerometer)

    # loop through to update gravity information
    for i in range(len(gravity) - 1):
        gravity[i + 1] = (1 - smoothing_factor) * gravity[i] + smoothing_factor * accelerometer[i + 1]

    return gravity


def get_gravity_and_linear_acceleration(accelerometer: np.ndarray, sensor_sample_rate_hz: float,
                                        low_pass_sample_rate_hz: float = 1) -> Tuple[np.ndarray, np.ndarray]:
    """

    :param accelerometer:
    :param sensor_sample_rate_hz:
    :param low_pass_sample_rate_hz:
    :return:
    """
    # get smoothing factor (alpha)
    alpha = get_smoothing_factor(sensor_sample_rate_hz=sensor_sample_rate_hz,
                                 low_pass_sample_rate_hz=low_pass_sample_rate_hz)

    # extract gravity via exponential filtering
    gravity = get_gravity(accelerometer=accelerometer, smoothing_factor=alpha)

    # subtract gravity from acceleration
    linear_acceleration = accelerometer - gravity

    return gravity, linear_acceleration
