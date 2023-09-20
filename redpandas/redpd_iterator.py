"""
Iterators used in redpd_preprocess.
"""

import numpy as np
from typing import Tuple, Iterator

# RC filter response: mag first contribution to stack overflow as slipstream
# https://stackoverflow.com/questions/62448904/how-to-implement-continuous-time-high-low-pass-filter-in-python


# todo: put types onto the variables
def rc_high_pass(x_new,
                 x_old,
                 y_old,
                 sample_rate_hz: int,
                 frequency_cut_low_hz: float) -> float:
    """
    High pass RC filter

    :param x_new: new x
    :param x_old: old x
    :param y_old: old y
    :param sample_rate_hz: sample rate in Hz
    :param frequency_cut_low_hz: low cutoff frequency in Hz
    :return: new y
    """
    rc = 1. / (2. * np.pi * frequency_cut_low_hz)
    return (rc / (rc + (1. / sample_rate_hz))) * (y_old + x_new - x_old)


# todo: put types onto the variables
def rc_low_pass(x_new,
                y_old,
                sample_rate_hz: int,
                frequency_cut_high_hz: float) -> float:
    """
    Low pass RC filter

    :param x_new: new x
    :param y_old: old y
    :param sample_rate_hz: sample rate in Hz
    :param frequency_cut_high_hz: high cutoff frequency in Hz
    :return: new y
    """
    sample_interval_s = 1. / sample_rate_hz
    alpha = sample_interval_s / ((1. / (2. * np.pi * frequency_cut_high_hz)) + sample_interval_s)
    return x_new * alpha + (1. - alpha) * y_old


def rc_iterator_highlow(sig_wf: np.ndarray,
                        sample_rate_hz: int,
                        frequency_cut_low_hz: float,
                        frequency_cut_high_hz: float) -> Iterator[Tuple[float, float]]:
    """
    RC filter high and low pass iterator

    :param sig_wf: signal waveform
    :param sample_rate_hz: sample rate in Hz
    :param frequency_cut_low_hz: low cutoff frequency in Hz
    :param frequency_cut_high_hz: high cutoff frequency in Hz
    :return: yield new y high pass, new y low pass
    """
    # Initialize. This can be improved to match wikipedia.
    x_prev = 0
    y_prev_high = 0
    y_prev_low = 0

    for x in sig_wf:
        y_prev_high = rc_high_pass(x, x_prev, y_prev_high, sample_rate_hz, frequency_cut_low_hz)
        y_prev_low = rc_low_pass(x, y_prev_low, sample_rate_hz, frequency_cut_high_hz)
        x_prev = x
        yield y_prev_high, y_prev_low


def rc_iterator_high_pass(sig_wf: np.ndarray,
                          sample_rate_hz: int,
                          frequency_cut_low_hz: float) -> Iterator[Tuple[float, float]]:
    """
    RC filter high pass iterator

    :param sig_wf: signal waveform
    :param sample_rate_hz: sample rate in Hz
    :param frequency_cut_low_hz: low cutoff frequency in Hz
    :return: new y high pass
    """
    # Initialize. This can be improved to match wikipedia.
    # x_prev = np.mean(sensor_wf)
    x_prev = 0
    y_prev_high = 0

    for x in sig_wf:
        y_prev_high = rc_high_pass(x, x_prev, y_prev_high, sample_rate_hz, frequency_cut_low_hz)
        x_prev = x
        yield y_prev_high


def rc_iterator_lowpass(sig_wf: np.ndarray,
                        sample_rate_hz: int,
                        frequency_cut_high_hz: float) -> Iterator[float]:
    """
    RC filter low pass iterator

    :param sig_wf: signal waveform
    :param sample_rate_hz: sample rate in Hz
    :param frequency_cut_high_hz: low cutoff frequency in Hz
    :return: new y low pass
    """
    # Initialize. This can be improved to match wikipedia.
    y_prev_low = 0

    for x in sig_wf:
        y_prev_low = rc_low_pass(x, y_prev_low, sample_rate_hz, frequency_cut_high_hz)
        yield y_prev_low
