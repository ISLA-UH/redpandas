"""
This module contains general utilities that can work with values containing nans.
"""
# todo: add types in function definitions
# todo: finish function documentation
from enum import Enum
from typing import Tuple

import numpy as np
from scipy import signal
import obspy.signal.filter
import pandas as pd

# RedVox and RedPandas
from redvox.common import date_time_utils as dt
import redpandas.redpd_iterator as rdp_iter
import redpandas.redpd_scales as rpd_scales


MG_RT = 0.00012  # Molar mass of air x gravity / (gas constant x standard temperature)
PRESSURE_REF_kPa = 101.325


# Define classes
class NormType(Enum):
    """
    Enumeration of normalization types.
    """
    MAX: str = "max"
    L1: str = "l1"
    L2: str = "l2"
    OTHER: str = "other"


# Auxiliary modules for building stations
def datetime_now_epoch_s() -> float:
    """
    Returns the invocation Unix time in seconds
    :return: The current epoch timestamp as seconds since the epoch UTC
    """
    return dt.datetime_to_epoch_seconds_utc(dt.now())


def datetime_now_epoch_micros() -> float:
    """
    Returns the invocation Unix time in microseconds
    :return: The current epoch timestamp as microseconds since the epoch UTC
    """
    return dt.datetime_to_epoch_microseconds_utc(dt.now())


def normalize(sig: np.ndarray, scaling: float = 1., norm_type: NormType = NormType.MAX) -> np.ndarray:
    """
    Scale a 1D time series
    :param sig: time series signature
    :param scaling: scaling parameter, division
    :param norm_type: {'max', l1, l2}, optional
    :return: The scaled series
    """
    if norm_type == NormType.MAX:
        return sig / np.nanmax(np.abs(sig))
    elif norm_type == NormType.L1:
        return sig / np.nansum(sig)
    elif norm_type == NormType.L2:
        return sig / np.sqrt(np.nansum(sig * sig))
    else:  # Must be NormType.Other
        return sig / scaling


def demean_nan(sig: np.ndarray) -> np.ndarray:
    """
    Detrend and normalize a 1D time series
    :param sig: time series with (possibly) non-zero mean
    :return: Detrended and normalized time series
    """
    return np.nan_to_num(sig - np.nanmean(sig))


def detrend_nan(sig: np.ndarray) -> np.ndarray:
    """
    Detrend and normalize a 1D time series
    :param sig: time series with (possibly) non-zero mean
    :return: Detrended and normalized time series
    """
    return signal.detrend(demean_nan(sig))


def demean_nan_norm(sig: np.ndarray, scaling: float = 1., norm_type: NormType = NormType.MAX) -> np.ndarray:
    """
    Detrend and normalize a 1D time series
    :param sig: time series with (possibly) non-zero mean
    :param scaling: scaling parameter, division
    :param norm_type: {'max', l1, l2}, overrides scikit default of 'l2' by 'max'
    :return: The detrended and denormalized series.
    """
    return normalize(demean_nan(sig), scaling=scaling, norm_type=norm_type)


def demean_nan_matrix(sig: np.ndarray) -> np.ndarray:
    """
    Detrend and normalize a matrix of time series
    :param sig: time series with (possibly) non-zero mean
    :return: The detrended and normalized signature
    """
    return np.nan_to_num(np.subtract(sig.transpose(), np.nanmean(sig, axis=1))).transpose()


def taper_tukey(sig_or_time: np.ndarray, fraction_cosine: float) -> np.ndarray:
    """
    Constructs a symmetric Tukey window with the same dimensions as a time or signal numpy array.
    fraction_cosine = 0 is a rectangular window, 1 is a Hann window
    :param sig_or_time: input signal or time
    :param fraction_cosine: fraction of the window inside the cosine tapered window, shared between the head and tail
    :return: tukey taper window amplitude
    """
    return signal.windows.tukey(M=np.size(sig_or_time), alpha=fraction_cosine, sym=True)


def pad_reflection_symmetric(sig_wf) -> Tuple[np.ndarray, int]:
    """
    Apply reflection transformation
    :param sig_wf:
    :return:
    """
    number_points_to_flip_per_edge = int(len(sig_wf)//2)
    wf_folded = np.pad(np.copy(sig_wf),
                       (number_points_to_flip_per_edge, number_points_to_flip_per_edge),
                       'reflect')
    wf_folded *= taper_tukey(wf_folded, fraction_cosine=0.5)
    return wf_folded, number_points_to_flip_per_edge


def filter_reflection_highpass(sig_wf, filter_cutoff_hz, sample_rate_hz) -> np.ndarray:
    """
    Apply filter
    :param sig_wf:
    :param filter_cutoff_hz:
    :param sample_rate_hz:
    :return: sig filtered
    """
    wf_folded, number_points_to_flip_per_edge = pad_reflection_symmetric(sig_wf)

    sig_folded_filtered = highpass_obspy(sensor_wf=wf_folded,
                                         frequency_low_hz=filter_cutoff_hz,
                                         sample_rate_hz=sample_rate_hz)

    return sig_folded_filtered[number_points_to_flip_per_edge:-number_points_to_flip_per_edge]


def height_asl_from_pressure_below10km(bar_waveform: np.ndarray) -> np.ndarray:
    """
    Simple model for troposphere
    :param bar_waveform: barometric pressure in kPa
    :return: height ASL in m
    """
    return -np.log(bar_waveform/rpd_scales.Slice.PREF_KPA)/MG_RT


def model_height_from_pressure_skyfall(pressure_kPa) -> float:
    """
    Returns empirical height in m from input pressure
    :param pressure_kPa: barometric pressure in kPa
    :return: height in m
    """
    scaled_pressure = -np.log(pressure_kPa/PRESSURE_REF_kPa)
    # Empirical model constructed from
    # c, stats = np.polynomial.polynomial.polyfit(poly_x, bounder_loc['Alt_m'], 8, full=True)
    c = [1.52981286e+02, 7.39552295e+03, 2.44663285e+03, -3.57402081e+03, 2.02653051e+03,
         -6.26581722e+02, 1.11758211e+02, -1.08674469e+01, 4.46784010e-01]
    return np.polynomial.polynomial.polyval(scaled_pressure, c, tensor=False)


def rc_high_pass_signal(sig, sample_rate, highpass_cutoff) -> np.array:
    """
    todo: complete me
    :param sig:
    :param sample_rate:
    :param highpass_cutoff:
    :return: highpass_signal
    """
    return np.array([[high]
                     for high
                     in rdp_iter.rc_iterator_high_pass(sig, sample_rate, highpass_cutoff)])


# "Traditional" solution, up to Nyquist
def bandpass_butter_uneven(sensor_wf, filter_order, frequency_cut_low_hz, sample_rate_hz) -> np.ndarray:
    """
    todo: complete me
    :param sensor_wf:
    :param filter_order:
    :param frequency_cut_low_hz:
    :param sample_rate_hz:
    :return:
    """
    # Frequencies are scaled by Nyquist, with 1 = Nyquist
    # filter_order = 4,
    nyquist = 0.5 * sample_rate_hz
    edge_low = frequency_cut_low_hz / nyquist
    edge_high = 0.5
    [b, a] = signal.butter(N=filter_order, Wn=[edge_low, edge_high], btype='bandpass')
    return signal.filtfilt(b, a, np.copy(sensor_wf))


def highpass_obspy(sensor_wf, frequency_low_hz, sample_rate_hz, filter_order=4) -> np.ndarray:
    """
    todo: complete me
    :param sensor_wf:
    :param frequency_low_hz:
    :param sample_rate_hz:
    :param filter_order:
    :return: sensor_highpass
    """
    return obspy.signal.filter.highpass(np.copy(sensor_wf),
                                        frequency_low_hz,
                                        sample_rate_hz, corners=filter_order,
                                        zerophase=True)


def xcorr_uneven(sig_x: np.ndarray, sig_ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, int, np.ndarray]:
    """
    Variation of cross-correlation function cross_stas.xcorr_all for unevenly sampled data
    with identical sampling and duration.
    :param sig_x: processed signal
    :param sig_ref: reference signal
    :return: cross-correlation metrics
    """
    nx = len(sig_x)
    nref = len(sig_ref)
    if nx > nref:
        print('Vectors must have equal sampling and lengths')
    elif nx < nref:
        print('Vectors must have equal sampling and lengths')
    elif nx == nref:
        """Cross correlation is centered in the middle of the record and has length NX"""
        # Fastest, o(NX) and can use FFT solution
        if nx % 2 == 0:
            xcorr_indexes = np.arange(-int(nx/2), int(nx/2))
        else:
            xcorr_indexes = np.arange(-int(nx/2), int(nx/2)+1)

        xcorr = signal.correlate(sig_ref, sig_x, mode='same')
        # Normalize
        xcorr /= nx * sig_x.std() * sig_ref.std()
        xcorr_offset_index = np.argmax(np.abs(xcorr))
        xcorr_offset_samples = xcorr_indexes[xcorr_offset_index]
        xcorr_peak = xcorr[xcorr_offset_index]

    else:
        print('One of the waveforms is broken')
        # todo: use better defaults
        return np.array([]), np.array([]), np.nan, np.nan, np.array([])

    return xcorr, xcorr_indexes, xcorr_peak, xcorr_offset_index, xcorr_offset_samples


def highpass_from_diff(sensor_waveform: np.ndarray,
                       sensor_epoch_s: np.ndarray,
                       sample_rate: int or float,
                       highpass_type: str = 'obspy',
                       frequency_filter_low: float = 1./rpd_scales.Slice.T100S,
                       filter_order: int = 4) -> Tuple[np.ndarray, float]:
    """
    Preprocess barometer data:
    - remove nans and DC offset by getting the differential pressure in kPa
    - apply highpass filter at 100 second periods
    - reconstruct Pressure in kPa from differential pressure: P(i) = dP(i) + P(i-1)
    :param sensor_waveform:
    :param sensor_epoch_s:
    :param sample_rate:
    :param highpass_type: 'obspy', 'butter', 'rc'
    :param frequency_filter_low: 100s default
    :param filter_order: Default is 4.
    :zero phase filters are acausal
    :return:
    """

    # Apply diff to remove DC offset; difference of nans is a nan
    # Replace nans with zeros, otherwise most things don't run
    # Using gradient instead of diff seems to fix off by zero issue!
    sensor_waveform_grad_dm = demean_nan(np.gradient(sensor_waveform))

    # Override default high pass at 100 seconds if signal is too short
    # May be able to zero pad ... with ringing. Or fold as needed.
    if sensor_epoch_s[-1] - sensor_epoch_s[0] < 2/frequency_filter_low:
        frequency_filter_low = 2/(sensor_epoch_s[-1] - sensor_epoch_s[0])
        print('Default 100s highpass override. New highpass period = ', 1/frequency_filter_low)

    # Fold edges of wf
    sensor_waveform_fold, number_points_folded = pad_reflection_symmetric(sensor_waveform_grad_dm)

    if highpass_type == "obspy":
        # Zero phase, acausal
        sensor_waveform_dp_filtered = \
            obspy.signal.filter.highpass(corners=filter_order,
                                         data=np.copy(sensor_waveform_fold),
                                         freq=frequency_filter_low,
                                         df=sample_rate,
                                         zerophase=True)

    elif highpass_type == "butter":
        [b, a] = signal.butter(N=filter_order,
                               Wn=frequency_filter_low,
                               fs=sample_rate,
                               btype='highpass',
                               output='ba')
        # Zero phase, acausal
        sensor_waveform_dp_filtered = signal.filtfilt(b, a, sensor_waveform_fold)

    elif highpass_type == "rc":
        # RC is slow and not zero-phase, does not need a taper to work (but it doesn't hurt)
        sensor_waveform_dp_filtered = \
            rc_high_pass_signal(sig=np.copy(sensor_waveform_fold),
                                sample_rate=sample_rate,
                                highpass_cutoff=frequency_filter_low)

    else:
        raise Exception("No filter selected. Type 'obspy', 'butter', or 'rc'.")

    # Cut fold edges of wf
    sensor_waveform_dp_filtered = sensor_waveform_dp_filtered[number_points_folded:-number_points_folded]

    # Reconstruct Function dP: P(0), P(i) = dP(i) + P(i-1)
    sensor_waveform_reconstruct = np.zeros((len(sensor_waveform_dp_filtered)))
    # Initialize
    sensor_waveform_reconstruct[0] = sensor_waveform_dp_filtered[0]

    for i in range(1, len(sensor_waveform_dp_filtered) - 1):
        sensor_waveform_reconstruct[i] = sensor_waveform_dp_filtered[i] + sensor_waveform_reconstruct[i-1]

    return sensor_waveform_reconstruct, frequency_filter_low


def df_column_unflatten(df: pd.DataFrame,
                        col_wf_label: str,
                        col_ndim_label: str):
    """
    Restores original shape of elements in column. Used for loading columns in dataframe from parquet.

    :param df: pandas DataFrame
    :param col_wf_label: column label for data that needs reshaping, usually waveform arrays.
    :param col_ndim_label: column label with dimensions for reshaping. Elements in column need to be a numpy array.
    :return: original df, replaces column values with reshaped ones
    """

    col_values = df[col_wf_label].to_numpy()
    for index_array in df.index:
        col_values[index_array].shape = (df[col_ndim_label][index_array][0],
                                         df[col_ndim_label][index_array][1])

