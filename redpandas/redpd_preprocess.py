"""
Utilities that can work with values containing nans. Mainly used for data manipulation
before construction of RedPandas DataFrame.
"""

from enum import Enum
from typing import Tuple, Union

import numpy as np
from scipy import signal
import obspy.signal.filter
import pandas as pd

# RedVox and RedPandas
from redvox.common import date_time_utils as dt
import redpandas.redpd_iterator as rdp_iter
import redpandas.redpd_scales as rpd_scales


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
def find_nearest_idx(array: np.ndarray, value: float) -> np.array:
    """
    :param array: np.array
    :param value: float/int
    :return: nearest idx for value in array
    """
    return (np.abs(np.asarray(array) - value)).argmin()


def datetime_now_epoch_s() -> float:
    """
    :return: The current epoch timestamp as seconds since the epoch UTC
    """
    return dt.datetime_to_epoch_seconds_utc(dt.now())


def datetime_now_epoch_micros() -> float:
    """
    :return: The current epoch timestamp as microseconds since the epoch UTC
    """
    return dt.datetime_to_epoch_microseconds_utc(dt.now())


def normalize(sig_wf: np.ndarray, scaling: float = 1., norm_type: NormType = NormType.MAX) -> np.ndarray:
    """
    :param sig_wf: signal waveform
    :param scaling: scaling parameter, division
    :param norm_type: {'max', l1, l2}, optional
    :return: The scaled 1D time series
    """
    if norm_type == NormType.MAX:
        return sig_wf / np.nanmax(np.abs(sig_wf))
    elif norm_type == NormType.L1:
        return sig_wf / np.nansum(sig_wf)
    elif norm_type == NormType.L2:
        return sig_wf / np.sqrt(np.nansum(sig_wf * sig_wf))
    else:  # Must be NormType.Other
        return sig_wf / scaling


def demean_nan(sig_wf: np.ndarray) -> np.ndarray:
    """
    :param sig_wf: signal waveform
    :return: Detrended and normalized time series
    """
    return np.nan_to_num(sig_wf - np.nanmean(sig_wf))


def detrend_nan(sig_wf: np.ndarray) -> np.ndarray:
    """
    :param sig_wf: signal waveform
    :return: Detrended and normalized time series
    """
    return signal.detrend(demean_nan(sig_wf))


def demean_nan_norm(sig_wf: np.ndarray, scaling: float = 1., norm_type: NormType = NormType.MAX) -> np.ndarray:
    """
    :param sig_wf: signal waveform
    :param scaling: scaling parameter, division
    :param norm_type: {'max', l1, l2}, overrides scikit default of 'l2' by 'max'
    :return: The detrended and denormalized 1D time series.
    """
    return normalize(demean_nan(sig_wf), scaling=scaling, norm_type=norm_type)


def demean_nan_matrix(sig_wf: np.ndarray) -> np.ndarray:
    """
    :param sig_wf: signal waveform
    :return: The detrended and normalized signature of a matrix of time series
    """
    return np.nan_to_num(np.subtract(sig_wf.transpose(), np.nanmean(sig_wf, axis=1))).transpose()


def taper_tukey(sig_wf_or_time: np.ndarray,
                fraction_cosine: float) -> np.ndarray:
    """
    Constructs a symmetric Tukey window with the same dimensions as a time or signal numpy array.
    fraction_cosine = 0 is a rectangular window, 1 is a Hann window

    :param sig_wf_or_time: input signal or time
    :param fraction_cosine: fraction of the window inside the cosine tapered window, shared between the head and tail
    :return: tukey taper window amplitude
    """
    return signal.windows.tukey(M=np.size(sig_wf_or_time), alpha=fraction_cosine, sym=True)


def pad_reflection_symmetric(sig_wf: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    :param sig_wf: signal waveform
    :return: input signal with reflected edges, numbers of points folded per edge
    """
    number_points_to_flip_per_edge = int(len(sig_wf) // 2)
    wf_folded = np.pad(np.copy(sig_wf),
                       (number_points_to_flip_per_edge, number_points_to_flip_per_edge),
                       'reflect')
    wf_folded *= taper_tukey(wf_folded, fraction_cosine=0.5)
    return wf_folded, number_points_to_flip_per_edge


def filter_reflection_highpass(sig_wf: np.ndarray,
                               sample_rate_hz: Union[float, int],
                               filter_cutoff_hz: float) -> np.ndarray:
    """
    Apply fold filter to input signal (edges reflected) and highpass

    :param sig_wf: signal waveform
    :param filter_cutoff_hz: filter corner frequency in Hz
    :param sample_rate_hz: sampling rate in Hz
    :return: signal folded and filtered
    """
    wf_folded, number_points_to_flip_per_edge = pad_reflection_symmetric(sig_wf)
    sig_folded_filtered = obspy.signal.filter.highpass(np.copy(wf_folded),
                                                       filter_cutoff_hz,
                                                       sample_rate_hz, corners=4,
                                                       zerophase=True)
    return sig_folded_filtered[number_points_to_flip_per_edge:-number_points_to_flip_per_edge]


def height_asl_from_pressure_below10km(bar_waveform: np.ndarray) -> np.ndarray:
    """
    Simple model for troposphere

    :param bar_waveform: barometric pressure in kPa
    :return: height ASL in m
    """
    return -np.log(bar_waveform / rpd_scales.Slice.PREF_KPA) / rpd_scales.MG_RT


def model_height_from_pressure_skyfall(pressure_kpa: np.ndarray) -> np.ndarray:
    """
    Returns empirical height in m from input pressure

    :param pressure_kpa: barometric pressure in kPa
    :return: height in m
    """
    scaled_pressure = -np.log(pressure_kpa / rpd_scales.PRESSURE_REF_kPa)
    # Empirical model constructed from
    # c, stats = np.polynomial.polynomial.polyfit(poly_x, bounder_loc['Alt_m'], 8, full=True)
    c = [1.52981286e+02, 7.39552295e+03, 2.44663285e+03, -3.57402081e+03, 2.02653051e+03,
         -6.26581722e+02, 1.11758211e+02, -1.08674469e+01, 4.46784010e-01]
    return np.polynomial.polynomial.polyval(scaled_pressure, c, tensor=False)


def rc_high_pass_signal(sig_wf: np.ndarray,
                        sample_rate_hz: int,
                        highpass_cutoff: float) -> np.ndarray:
    """
    Apply RC high pass filter to signal

    :param sig_wf: signal waveform
    :param sample_rate_hz: sampling rate in Hz
    :param highpass_cutoff: filter corner frequency in Hz
    :return: highpass signal
    """
    return np.array([[high]
                     for high
                     in rdp_iter.rc_iterator_high_pass(sig_wf, sample_rate_hz, highpass_cutoff)])


# "Traditional" solution, up to Nyquist
def bandpass_butter_uneven(sig_wf: np.ndarray,
                           sample_rate_hz: int,
                           frequency_cut_low_hz: float,
                           filter_order: int) -> np.ndarray:
    """
    Apply butterworth filter to a 1D signal

    :param sig_wf: signal waveform
    :param sample_rate_hz: sampling rate in Hz
    :param frequency_cut_low_hz: filter corner frequency in Hz
    :param filter_order: filter corners / order
    :return: bandpassed signal
    """
    # Frequencies are scaled by Nyquist, with 1 = Nyquist
    edge_low = frequency_cut_low_hz / (0.5 * sample_rate_hz)
    [b, a] = signal.butter(N=filter_order, Wn=[edge_low, 0.5], btype='bandpass')
    return signal.filtfilt(b, a, np.copy(sig_wf))


# todo: return types?  -> Tuple[np.ndarray, np.ndarray, float, float, np.ndarray]
def xcorr_uneven(sig_x: np.ndarray, sig_ref: np.ndarray):
    """
    Variation of cross-correlation function cross_stas.xcorr_all for unevenly sampled data
    with identical sampling and duration.

    :param sig_x: processed signal
    :param sig_ref: reference signal
    :return: cross-correlation metrics
    """
    nx = len(sig_x)
    nref = len(sig_ref)
    if nx != nref:
        print('Vectors must have equal sampling and lengths')
    elif nx == nref:
        """Cross correlation is centered in the middle of the record and has length NX"""
        # Fastest, o(NX) and can use FFT solution
        xcorr_indexes = np.arange(-int(nx / 2), int(nx / 2) + (nx % 2))

        xcorr = signal.correlate(sig_ref, sig_x, mode='same')
        # Normalize
        xcorr /= nx * sig_x.std() * sig_ref.std()
        xcorr_offset_index = np.argmax(np.abs(xcorr))
        xcorr_offset_samples = xcorr_indexes[xcorr_offset_index]
        xcorr_peak = xcorr[xcorr_offset_index]

        return xcorr, xcorr_indexes, xcorr_peak, xcorr_offset_index, xcorr_offset_samples
    else:
        print('One of the waveforms is broken')
    return np.array([]), np.array([]), np.nan, np.nan, np.array([])


def highpass_from_diff(sig_wf: np.ndarray,
                       sig_epoch_s: np.ndarray,
                       sample_rate_hz: int or float,
                       fold_signal: bool = True,
                       highpass_type: str = 'obspy',
                       frequency_filter_low: float = 1./rpd_scales.Slice.T100S,
                       filter_order: int = 4) -> Tuple[np.ndarray, float]:
    """
    Preprocess barometer data:

    - remove nans and DC offset by getting the differential pressure in kPa
    - apply highpass filter at 100 second periods
    - reconstruct Pressure in kPa from differential pressure: P(i) = dP(i) + P(i-1)

    zero phase filters are acausal

    :param sig_wf: signal waveform
    :param sig_epoch_s: signal time in epoch s
    :param sample_rate_hz: sampling rate in Hz
    :param fold_signal: apply reflection transformation and fold edges
    :param highpass_type: 'obspy', 'butter', 'rc'
    :param frequency_filter_low: apply highpass filter. Default is 100-second periods
    :param filter_order: filter corners / order. Default is 4.
    :return: filtered signal waveform, frequency_filter_low value used
    """
    # Apply diff to remove DC offset; difference of nans is a nan
    # Replace nans with zeros, otherwise most things don't run
    # Using gradient instead of diff seems to fix off by zero issue!
    sensor_waveform_grad_dm = demean_nan(np.gradient(sig_wf))

    # Override default high pass at 100 seconds if signal is too short
    # May be able to zero pad ... with ringing. Or fold as needed.
    if (sig_epoch_s[-1] - sig_epoch_s[0]) < (2 / frequency_filter_low):
        frequency_filter_low = 2 / (sig_epoch_s[-1] - sig_epoch_s[0])
        print(f'Default 100s highpass override. New highpass period = {1 / frequency_filter_low}')

    number_points_folded = 0  # set just in case
    # Fold edges of wf
    if fold_signal is True:
        sensor_waveform_fold, number_points_folded = pad_reflection_symmetric(sensor_waveform_grad_dm)
    else:
        sensor_waveform_fold = sensor_waveform_grad_dm

    if highpass_type == "obspy":
        # Zero phase, acausal
        sensor_waveform_dp_filtered = \
            obspy.signal.filter.highpass(corners=filter_order,
                                         data=np.copy(sensor_waveform_fold),
                                         freq=frequency_filter_low,
                                         df=sample_rate_hz,
                                         zerophase=True)
    elif highpass_type == "butter":
        [b, a] = signal.butter(N=filter_order,
                               Wn=frequency_filter_low,
                               fs=sample_rate_hz,
                               btype='highpass',
                               output='ba')
        # Zero phase, acausal
        sensor_waveform_dp_filtered = signal.filtfilt(b, a, sensor_waveform_fold)

    elif highpass_type == "rc":
        # RC is slow and not zero-phase, does not need a taper to work (but it doesn't hurt)
        sensor_waveform_dp_filtered = \
            rc_high_pass_signal(sig_wf=np.copy(sensor_waveform_fold),
                                sample_rate_hz=sample_rate_hz,
                                highpass_cutoff=frequency_filter_low)

    else:
        raise Exception("No filter selected. Type 'obspy', 'butter', or 'rc'.")

    if fold_signal is True:
        # Cut fold edges of wf
        sensor_waveform_dp_filtered = sensor_waveform_dp_filtered[number_points_folded:-number_points_folded]

    # Reconstruct Function dP: P(0), P(i) = dP(i) + P(i-1)
    sensor_waveform_reconstruct = np.zeros((len(sensor_waveform_dp_filtered)))
    # Initialize
    sensor_waveform_reconstruct[0] = sensor_waveform_dp_filtered[0]

    for i in range(1, len(sensor_waveform_dp_filtered) - 1):
        sensor_waveform_reconstruct[i] = sensor_waveform_dp_filtered[i] + sensor_waveform_reconstruct[i-1]

    return sensor_waveform_reconstruct, frequency_filter_low


# Auxiliary functions to open parquets
def df_unflatten(df: pd.DataFrame) -> None:
    """
    Restores original shape of elements in all column. Used for loading dataframe from parquet.

    :param df: pandas DataFrame
    :return: original df
    """
    for col_name in [col.replace('_ndim', '') for col in df.filter(like='_ndim', axis=1).columns]:
        col_ndim_label = col_name + "_ndim"
        col_values = df[col_name].to_numpy()
        for index_array in df.index:
            if len(df[col_ndim_label][index_array]) > 1:  # check that there is data
                if len(df[col_ndim_label][index_array]) == 2:
                    col_values[index_array].shape = (int(df[col_ndim_label][index_array][0]),
                                                     int(df[col_ndim_label][index_array][1]))
                if len(df[col_ndim_label][index_array]) == 3:  # tfr
                    col_values[index_array].shape = (int(df[col_ndim_label][index_array][0]),
                                                     int(df[col_ndim_label][index_array][1]),
                                                     int(df[col_ndim_label][index_array][2]))


def df_column_unflatten(df: pd.DataFrame,
                        col_wf_label: str,
                        col_ndim_label: str) -> None:
    """
    Restores original shape of elements in column. Used for loading columns in dataframe from parquet.

    :param df: pandas DataFrame
    :param col_wf_label: column label for data that needs reshaping, usually waveform arrays.
    :param col_ndim_label: column label with dimensions for reshaping. Elements in column need to be a numpy array.
    :return: original df, replaces column values with reshaped ones
    """
    col_values = df[col_wf_label].to_numpy()
    for index_array in df.index:
        if len(df[col_ndim_label][index_array]) > 1:  # check that there is data
            if len(df[col_ndim_label][index_array]) == 2:
                col_values[index_array].shape = (int(df[col_ndim_label][index_array][0]),
                                                 int(df[col_ndim_label][index_array][1]))

            if len(df[col_ndim_label][index_array]) == 3:  # tfr
                col_values[index_array].shape = (int(df[col_ndim_label][index_array][0]),
                                                 int(df[col_ndim_label][index_array][1]),
                                                 int(df[col_ndim_label][index_array][2]))
