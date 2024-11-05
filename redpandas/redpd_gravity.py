"""
Functions to apply exponential filter on accelerometer to separate gravity and linear acceleration.
Based on the "past-gen" Android code without incorporation of the gyroscope
http://josejuansanchez.org/android-sensors-overview/gravity_and_linear_acceleration/README.html
"""

import numpy as np
from typing import Tuple


# This is comparable to RC filter, smoothing factor is an approximation to alpha
# See redpd_iterator
def get_smoothing_factor(sensor_sample_rate_hz: float, low_pass_sample_rate_hz: float = 1) -> float:
    """
    Obtain smoothing factor

    :param sensor_sample_rate_hz: sample rate of sensor in Hz
    :param low_pass_sample_rate_hz: sample rate of low pass filter in Hz
    :return: alpha
    """
    # assuming sensor_sample_rate_hz << low_pass_sample_rate_hz
    return low_pass_sample_rate_hz / sensor_sample_rate_hz


def get_gravity(accelerometer: np.ndarray, smoothing_factor: float) -> np.ndarray:
    """
    based on the slack thread: https://tinyurl.com/f6t3h2fp

    :param accelerometer: accelerometer signal waveform
    :param smoothing_factor: from get_smoothing_factor function
    :return: numpy array with gravity values
    """
    # initialize gravity array
    gravity = np.zeros(len(accelerometer)) * np.mean(accelerometer)
    # loop through to update gravity information
    for i in range(len(gravity) - 1):
        gravity[i + 1] = (1 - smoothing_factor) * gravity[i] + smoothing_factor * accelerometer[i + 1]

    return gravity


def get_gravity_and_linear_acceleration(accelerometer: np.ndarray,
                                        sensor_sample_rate_hz: float,
                                        low_pass_sample_rate_hz: float = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Obtain gravity and linear acceleration from smartphone accelerometer sensor

    :param accelerometer: accelerometer signal waveform
    :param sensor_sample_rate_hz: sample rate of accelerometer in Hz
    :param low_pass_sample_rate_hz: sample rate of low pass filter in Hz
    :return: numpy array with gravity and numpy array with linear acceleration
    """
    # get smoothing factor (alpha)
    alpha = get_smoothing_factor(sensor_sample_rate_hz=sensor_sample_rate_hz,
                                 low_pass_sample_rate_hz=low_pass_sample_rate_hz)
    # extract gravity via exponential filtering
    gravity = get_gravity(accelerometer=accelerometer, smoothing_factor=alpha)
    # linear acceleration is acceleration - gravity
    return gravity, accelerometer - gravity


"""
Generalized to any DC offset, for comparison
"""


def get_sensor_lowpass(sensor_wf: np.ndarray,
                       sensor_sample_rate_hz: float,
                       lowpass_frequency_hz: float = 1) -> np.ndarray:
    """
    based on the slack thread: https://tinyurl.com/f6t3h2fp

    :param sensor_wf: signal waveform
    :param sensor_sample_rate_hz: sample rate of sensor in Hz
    :param lowpass_frequency_hz: sample rate of low pass filter in Hz
    :return: sensor low pass
    """
    smoothing_factor = lowpass_frequency_hz / sensor_sample_rate_hz
    # initialize gravity array
    sensor_lowpass = np.zeros(len(sensor_wf))
    # loop through to update gravity information
    for i in range(len(sensor_lowpass) - 1):
        sensor_lowpass[i + 1] = (1 - smoothing_factor) * sensor_lowpass[i] + smoothing_factor * sensor_wf[i + 1]
    return sensor_lowpass


def get_lowpass_and_highpass(sensor_wf: np.ndarray,
                             sensor_sample_rate_hz: float,
                             lowpass_frequency_hz: float = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param sensor_wf: signal waveform
    :param sensor_sample_rate_hz: sample rate of sensor in Hz
    :param lowpass_frequency_hz: sample rate of low pass filter in Hz
    :return: sensor low pass and high pass
    """
    # extract low-frequency component via exponential filtering
    sensor_lowpass = get_sensor_lowpass(sensor_wf, sensor_sample_rate_hz, lowpass_frequency_hz)
    # subtract low-frequency component from waveform
    return sensor_lowpass, sensor_wf - sensor_lowpass
