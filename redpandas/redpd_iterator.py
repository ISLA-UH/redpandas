"""
This module contains iterators used in redpd_preprocess.

Last updated: 10 June 2021
"""

import numpy as np
from typing import Tuple, Generator, Iterator

# RC filter response: mag first contribution to stack overflow as slipstream
# https://stackoverflow.com/questions/62448904/how-to-implement-continuous-time-high-low-pass-filter-in-python


def rc_high_pass(x_new,  # TODO MAG: add type and -> return
                 x_old,
                 y_old,
                 sample_rate_hz: int,
                 frequency_cut_low_hz: float):
    """
    High pass RC filter
    :param x_new: TODO MAG: complete me
    :param x_old: TODO MAG: complete me
    :param y_old: TODO MAG: complete me
    :param sample_rate_hz: sample rate in Hz
    :param frequency_cut_low_hz: low cutoff frequency in Hz
    :return: TODO MAG: complete me
    """
    sample_interval_s = 1/sample_rate_hz
    rc = 1/(2 * np.pi * frequency_cut_low_hz)
    alpha = rc/(rc + sample_interval_s)
    y_new = alpha * (y_old + x_new - x_old)
    return y_new


def rc_low_pass(x_new,  # TODO MAG: add type and -> return
                y_old,
                sample_rate_hz: int,
                frequency_cut_high_hz: float):
    """
    Low pass RC filter
    :param x_new: TODO MAG: complete me
    :param y_old: TODO MAG: complete me
    :param sample_rate_hz: sample rate in Hz
    :param frequency_cut_high_hz: high cutoff frequency in Hz
    :return: TODO MAG: complete me
    """
    sample_interval_s = 1/sample_rate_hz
    rc = 1/(2 * np.pi * frequency_cut_high_hz)
    alpha = sample_interval_s/(rc + sample_interval_s)
    y_new = x_new * alpha + (1 - alpha) * y_old
    return y_new


def rc_iterator_highlow(sig_wf: np.ndarray,
                        sample_rate_hz: int,
                        frequency_cut_low_hz: float,
                        frequency_cut_high_hz: float) -> Iterator[Tuple[float, float]]:
    """
    TODO MAG: complete my description

    :param sig_wf: signal waveform
    :param sample_rate_hz: sample rate in Hz
    :param frequency_cut_low_hz: low cutoff frequency in Hz
    :param frequency_cut_high_hz: high cutoff frequency in Hz
    :return: TODO MAG: complete me
    """
    # Initialize. This can be improved to match wikipedia.
    x_prev = 0
    y_prev_high = 0
    y_prev_low = 0

    for x in sig_wf:
        y_prev_high = rc_high_pass(x, x_prev, y_prev_high, sample_rate_hz,
                                   frequency_cut_low_hz)
        y_prev_low = rc_low_pass(x, y_prev_low, sample_rate_hz,
                                 frequency_cut_high_hz)
        x_prev = x
        yield y_prev_high, y_prev_low


def rc_iterator_high_pass(sig_wf: np.ndarray,
                          sample_rate_hz: int,
                          frequency_cut_low_hz: float) -> Iterator[Tuple[float, float]]:
    """
    TODO MAG: complete my description
    :param sig_wf: signal waveform
    :param sample_rate_hz: sample rate in Hz
    :param frequency_cut_low_hz: low cutoff frequency in Hz
    :return: TODO MAG: complete me
    """
    # Initialize. This can be improved to match wikipedia.
    # x_prev = np.mean(sensor_wf)
    x_prev = 0
    y_prev_high = 0

    for x in sig_wf:
        y_prev_high = rc_high_pass(x, x_prev, y_prev_high, sample_rate_hz,
                                   frequency_cut_low_hz)
        x_prev = x
        yield y_prev_high


def rc_iterator_lowpass(sig_wf: np.ndarray,
                        sample_rate_hz: int,
                        frequency_cut_high_hz: float) -> Iterator[float]:
    """
    TODO MAG: complete my description
    :param sig_wf: signal waveform
    :param sample_rate_hz: sample rate in Hz
    :param frequency_cut_high_hz: low cutoff frequency in Hz
    :return: TODO MAG: complete me
    """
    # Initialize. This can be improved to match wikipedia.
    y_prev_low = 0

    for x in sig_wf:
        y_prev_low = rc_low_pass(x, y_prev_low, sample_rate_hz,
                                 frequency_cut_high_hz)
        yield y_prev_low
