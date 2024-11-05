"""
Calculate correlation.
"""

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from typing import Tuple


def find_nearest(array: np.ndarray, value: any) -> np.ndarray:
    """
    Find nearest value in numpy array

    :param array: a numpy array
    :param value: value to search for in array
    :return:
    """
    # https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    return np.argmin(np.abs(np.ceil(array[None].T - value)), axis=0)


# todo: type hints, figure out return value if any
def plot_square(xnorm_max,
                xoffset_s,
                xoffset_points,
                sig_descriptor: str = "Signal"):
    """
    Plot cross-correlation results

    :param xnorm_max: normalized cross correlation
    :param xoffset_s: offset cross correlation in seconds
    :param xoffset_points: offset points in cross correlation
    :param sig_descriptor: label to describe signal. Default is "Signal"
    """
    color_map = plt.get_cmap("Spectral_r")
    fig, ax = plt.subplots()
    im = ax.imshow(xnorm_max, cmap=color_map, origin='lower')
    fig.colorbar(im)
    ax.set_title(sig_descriptor + ' max cross-correlation in the time domain')

    fig, ax = plt.subplots()
    im = ax.imshow(xoffset_s, cmap=color_map, origin='lower')
    fig.colorbar(im)
    ax.set_title(sig_descriptor + ' cross-correlation offset in s')

    fig, ax = plt.subplots()
    im = ax.imshow(xoffset_points, cmap=color_map, origin='lower')
    fig.colorbar(im)
    ax.set_title(sig_descriptor + ' cross-correlation offset in points')


# todo: type hints
def most_similar_station_index(xnorm_max) -> Tuple[int, float]:
    """
    Sums over column, subtract self xcorr (1), divides by number of stations - 1

    :param xnorm_max: normalized cross correlation
    :return: index of most self-similar station to the ensemble, maximum of mean
    """
    xnorm_max_sum = np.sum(xnorm_max, axis=1)
    xnorm_stats = (xnorm_max_sum - 1) / (len(xnorm_max_sum) - 1)

    xcorr_ref_index = int(np.argmax(xnorm_stats))

    return xcorr_ref_index, xnorm_stats[xcorr_ref_index]


# Sort out time first: time gate input, refer to shared datum, correct times
def xcorr_pandas(df: pd.DataFrame,
                 sig_wf_label: str,
                 sig_sample_rate_label: str,
                 fs_fractional_tolerance: float = 0.02,
                 abs_xcorr: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns square matrix, a concise snapshot of the self-similarity of the input data set.

    :param df: input pandas data frame
    :param sig_wf_label: string for the waveform column name in df
    :param sig_sample_rate_label: string for the sample rate in Hz column name in df
    :param fs_fractional_tolerance: difference in sample rate (in Hz) tolerated. Default is 0.02
    :param abs_xcorr: Default is True
    :return: xcorr normalized, offset in seconds, and offset points
    """
    number_sig = len(df.index)
    print("Number of signals:", number_sig)

    # Initialize
    xcorr_offset_points = np.zeros((number_sig, number_sig))
    xcorr_offset_seconds = np.copy(xcorr_offset_points)
    xcorr_normalized_max = np.copy(xcorr_offset_points)

    for m in df.index:
        for n in df.index:
            sample_rate_condition = np.abs(df[sig_sample_rate_label][m] - df[sig_sample_rate_label][n]) \
                                    > fs_fractional_tolerance*df[sig_sample_rate_label][m]
            if sample_rate_condition:
                print(f"Sample rates out of tolerance for index m, n = {m}, {n}")
                continue
            else:
                sig_n = np.copy(df[sig_wf_label][n])
                sig_m = np.copy(df[sig_wf_label][m])
                # Generalized sensor cross correlations, including unequal lengths
                n_points = len(sig_n)
                m_points = len(sig_m)
                # Faster as floats
                if n_points > m_points:
                    """Cross Correlation 'full' sums over the dimension of sig_n"""
                    xcorr_indexes = np.arange(1 - n_points, m_points)
                    # Ensure it is a float
                    xcorr = signal.correlate(sig_m, sig_n, mode='full')
                elif n_points < m_points:
                    """Cross Correlation 'full' sums over the dimension of sig_m"""
                    xcorr_indexes = np.arange(1 - m_points, n_points)
                    xcorr = signal.correlate(sig_n, sig_m, mode='full')
                elif n_points == m_points:
                    """Cross correlation is centered in the middle of the record and has length n_points"""
                    # Fastest, o(NX) and can use FFT solution
                    xcorr_indexes = np.arange(-int(n_points / 2), int(n_points / 2) + (n_points % 2))
                    xcorr = signal.correlate(sig_m, sig_n, mode='same')
                else:
                    print('One of the waveforms is broken')
                    continue
                # Normalize
                xcorr /= np.sqrt(n_points * m_points) * sig_n.std() * sig_m.std()
                # Allows negative peak in cross correlation (pi phase shift)
                # Must be in phase -  for array processing
                xcorr_offset_index = np.argmax(np.abs(xcorr)) if abs_xcorr else np.argmax(xcorr)
                xcorr_offset_samples = -xcorr_indexes[xcorr_offset_index] if n_points < m_points \
                    else xcorr_indexes[xcorr_offset_index]
                xcorr_normalized_max[m, n] = xcorr[xcorr_offset_index]
                xcorr_offset_seconds[m, n] = xcorr_offset_samples / df[sig_sample_rate_label][n]
                xcorr_offset_points[m, n] = xcorr_offset_samples

    return xcorr_normalized_max, xcorr_offset_seconds, xcorr_offset_points


def xcorr_re_ref_pandas(df: pd.DataFrame,
                        ref_id_label: str,
                        sig_id_label: str,
                        sig_wf_label: str,
                        sig_sample_rate_label: str,
                        fs_fractional_tolerance: float = 0.02,
                        abs_xcorr: bool = True,
                        return_xcorr_full: bool = False,
                        new_column_label_xcorr_offset_points: str = 'xcorr_offset_points',
                        new_column_label_xcorr_offset_seconds: str = 'xcorr_offset_seconds',
                        new_column_label_xcorr_normalized_max: str = 'xcorr_normalized_max',
                        new_column_label_xcorr_full_array: str = 'xcorr_full') -> pd.DataFrame:
    """
    Returns new pandas columns per station with cross-correlation results relative to a reference station

    :param df: input pandas data frame
    :param ref_id_label: string for reference station id column name in df
    :param sig_id_label: string for station id column name in df
    :param sig_wf_label: string for the waveform column name in df
    :param sig_sample_rate_label: string for the sample rate in Hz column name in df
    :param fs_fractional_tolerance: difference in sample rate (in Hz) tolerated. Default is 0.02
    :param abs_xcorr: Default is True
    :param return_xcorr_full: default is False
    :param new_column_label_xcorr_offset_points: label for new column with xcorr offset points
    :param new_column_label_xcorr_offset_seconds: label for new column with xcorr offset seconds
    :param new_column_label_xcorr_normalized_max: label for new column with xcorr normalized
    :param new_column_label_xcorr_full_array: label for new column with xcorr full array
    :return: input dataframe with new columns
    """
    number_sig = len(df.index)
    print("XCORR Number of signals:", number_sig)
    m_list = df.index[df[sig_id_label] == ref_id_label]
    m = m_list[0]
    if len(m_list) > 1:
        raise Warning("More than one station meets the id spec. Picked first instance")
    # Initialize
    xcorr_offset_points = []
    xcorr_offset_seconds = []
    xcorr_normalized_max = []
    xcorr_full = []
    if m is not None:
        print("XCORR Reference station ", df[sig_id_label][m])
        sig_m = np.copy(df[sig_wf_label][m])
        m_points = len(sig_m)
        for n in df.index:
            sample_rate_condition = np.abs(df[sig_sample_rate_label][m] - df[sig_sample_rate_label][n]) \
                                    > fs_fractional_tolerance*df[sig_sample_rate_label][m]
            if sample_rate_condition:
                print("Sample rates out of tolerance")
                continue
            else:
                # Generalized sensor cross correlations, including unequal lengths
                sig_n = np.copy(df[sig_wf_label][n])
                n_points = len(sig_n)
                if n_points > m_points:
                    """Cross Correlation 'full' sums over the dimension of sig_n"""
                    xcorr_indexes = np.arange(1 - n_points, m_points)
                    # Ensure it is a float
                    xcorr = signal.correlate(sig_m, sig_n, mode='full')
                elif n_points < m_points:
                    """Cross Correlation 'full' sums over the dimension of sig_m"""
                    xcorr_indexes = np.arange(1 - m_points, n_points)
                    xcorr = signal.correlate(sig_n, sig_m, mode='full')
                elif n_points == m_points:
                    """Cross correlation is centered in the middle of the record and has length n_points"""
                    # Fastest, o(NX) and can use FFT solution
                    xcorr_indexes = np.arange(-int(n_points / 2), int(n_points / 2) + (n_points % 2))
                    xcorr = signal.correlate(sig_m, sig_n, mode='same')
                else:
                    print('One of the waveforms is broken')
                    continue
                # Normalize
                xcorr /= np.sqrt(n_points * m_points) * sig_n.std() * sig_m.std()
                # Allows negative peak in cross correlation (pi phase shift)
                # Must be in phase -  for array processing
                xcorr_offset_index = np.argmax(np.abs(xcorr)) if abs_xcorr else np.argmax(xcorr)
                xcorr_offset_samples = -xcorr_indexes[xcorr_offset_index] if n_points < m_points \
                    else xcorr_indexes[xcorr_offset_index]
                # Main export parameters
                # Allows negative peak in cross correlation (pi phase shift) in raw waveform, unless the input is power
                xcorr_normalized_max.append(xcorr[xcorr_offset_index])
                xcorr_offset_points.append(xcorr_offset_samples)
                xcorr_offset_seconds.append(xcorr_offset_samples / df[sig_sample_rate_label][n])
                if return_xcorr_full:
                    xcorr_full.append(xcorr)
        # Convert to columns and add it to df
        df[new_column_label_xcorr_normalized_max] = xcorr_normalized_max
        df[new_column_label_xcorr_offset_points] = xcorr_offset_points
        df[new_column_label_xcorr_offset_seconds] = xcorr_offset_seconds
        if return_xcorr_full:
            df[new_column_label_xcorr_full_array] = xcorr_full
    else:
        raise ValueError('ERROR: Incorrect reference station id')
    return df


def spectcorr_re_ref_pandas(df: pd.DataFrame,
                            ref_id_label: str,
                            sig_id_label: str,
                            sig_tfr_label: str,
                            sig_tfr_frequency_label: str,
                            sig_sample_rate_label: str,
                            sig_tfr_frequency_low_hz_label: str,
                            sig_tfr_frequency_high_hz_label: str,
                            return_xcorr_full: bool = False,
                            new_column_label_xcorr_offset_points: str = 'spectcorr_offset_points',
                            new_column_label_xcorr_offset_seconds: str = 'spectcorr_offset_seconds',
                            new_column_label_xcorr_normalized_max: str = 'spectcorr_normalized_max',
                            new_column_label_xcorr_peak_frequency_hz: str = 'spectcorr_peak_frequency_hz',
                            new_column_label_xcorr_full_array: str = 'spectcorr_full',
                            new_column_label_xcorr_full_per_band: str = 'spectcorr_per_band_full',
                            new_column_label_xcorr_full_frequency_hz: str = 'spectcorr_frequency_hz') -> pd.DataFrame:
    """
    Returns new pandas columns per station with spectral correlation results relative to a reference station

    :param df: input pandas data frame
    :param ref_id_label: string for reference station id column name in df
    :param sig_id_label: string for station id column name in df
    :param sig_tfr_label: string for tfr column name in df
    :param sig_tfr_frequency_label: string for tfr frequency column name in df
    :param sig_sample_rate_label: string for sample rate in Hz column name in df
    :param sig_tfr_frequency_low_hz_label: string for tfr low frequency values column name in df
    :param sig_tfr_frequency_high_hz_label: string for tfr high frequency values column name in df
    :param return_xcorr_full: default is False
    :param new_column_label_xcorr_offset_points: label for new column with xcorr offset points
    :param new_column_label_xcorr_offset_seconds: label for new column with xcorr offset seconds
    :param new_column_label_xcorr_normalized_max: label for new column with xcorr normalized max
    :param new_column_label_xcorr_peak_frequency_hz: label for new column with xcorr peak frequency
    :param new_column_label_xcorr_full_array: label for new column with xcorr full array
    :param new_column_label_xcorr_full_per_band: label for new column with xcorr full per band
    :param new_column_label_xcorr_full_frequency_hz: label for new column with xcorr frequencies
    :return: input df with new columns
    """
    # Have to learn how to use/validate correlate2D
    number_sig = len(df.index)
    print("SPECTCORR number of signals:", number_sig)
    # M is the reference station
    m_list = df.index[df[sig_id_label] == ref_id_label]
    m = m_list[0]
    if len(m_list) > 1:
        raise Warning("More than one station meets the id spec. Picked first instance")
    # Find frequency edges
    freq_index_low = find_nearest(df[sig_tfr_frequency_low_hz_label][m], df[sig_tfr_frequency_label][m])
    freq_index_high = find_nearest(df[sig_tfr_frequency_high_hz_label][m], df[sig_tfr_frequency_label][m])
    print(freq_index_low, freq_index_high)
    # Initialize
    xcorr_offset_points = []
    xcorr_offset_seconds = []
    xcorr_normalized_max = []
    xcorr_peak_frequency_hz = []
    xcorr_full = []
    xcorr_full_per_band = []
    xcorr_full_frequency = []

    if m is not None:
        print("XCORR Reference station ", df[sig_id_label][m])
        # Extract the passband of interest
        ref_tfr_m = np.copy(df[sig_tfr_label][m])[freq_index_low:freq_index_high, :]
        spect_corr_frequency = np.copy(df[sig_tfr_frequency_label][m])[freq_index_low:freq_index_high]
        # Improve error check
        ref_rows, ref_columns = ref_tfr_m.shape
        # MUST have equal rectangular matrices
        # Cross correlation is centered in the middle of the record and has length n_points
        # Fastest, o(NX) and can use FFT solution
        xcorr_index = np.arange(-int(ref_columns / 2), int(ref_columns / 2) + (ref_columns % 2))
        # Index matrix
        xcorr_index_mat = np.tile(xcorr_index, (ref_rows, 1))

        if np.amax(ref_tfr_m) <= 0:
            ref_tfr_m -= np.min(ref_tfr_m)
        for n in df.index:
            # Generalized sensor cross correlations, including unequal time lengths
            sig_tfr_n = np.copy(df[sig_tfr_label][n])[freq_index_low:freq_index_high, :]
            n_rows, n_columns = sig_tfr_n.shape
            if n_rows != ref_rows:
                print("TFR does not have the same frequency dimensions:", df[sig_id_label][n])
                continue
            if n_columns != ref_columns:
                print("TFR does not have the same time grid dimensions:", df[sig_id_label][n])
                continue
            # Frequency-by-frequency
            spect_corr = np.zeros(xcorr_index_mat.shape)
            spect_corr_per_band = np.zeros(xcorr_index_mat.shape)
            # Condition so there is always a positive component
            if np.amax(sig_tfr_n) <= 0:
                sig_tfr_n -= np.min(sig_tfr_n)
            # normalize per band
            for k in np.arange(ref_rows):
                spect_corr[k, :] = signal.correlate(ref_tfr_m[k, :], sig_tfr_n[k, :], mode='same')
                spect_corr_per_band[k, :] = spect_corr[k, :]/np.max(np.abs(spect_corr[k, :]))
            # Normalize by max
            spect_corr /= np.max(np.abs(spect_corr))
            frequency_index, time_index = np.unravel_index(np.argmax(spect_corr), spect_corr.shape)
            spect_xcorr_normalized_max = 1.  # By definition, refine
            # Main export parameters
            xcorr_normalized_max.append(spect_xcorr_normalized_max)
            xcorr_offset_points.append(xcorr_index_mat[frequency_index, time_index])
            xcorr_offset_seconds.append(xcorr_index_mat[frequency_index, time_index]/df[sig_sample_rate_label][n])
            xcorr_peak_frequency_hz.append(spect_corr_frequency[frequency_index])
            if return_xcorr_full:
                xcorr_full.append(spect_corr)
                xcorr_full_per_band.append(spect_corr_per_band)
                xcorr_full_frequency.append(spect_corr_frequency)

        # Convert to columns and add it to df
        df[new_column_label_xcorr_normalized_max] = xcorr_normalized_max
        df[new_column_label_xcorr_offset_points] = xcorr_offset_points
        df[new_column_label_xcorr_offset_seconds] = xcorr_offset_seconds
        df[new_column_label_xcorr_peak_frequency_hz] = xcorr_peak_frequency_hz
        if return_xcorr_full:
            df[new_column_label_xcorr_full_array] = xcorr_full
            df[new_column_label_xcorr_full_per_band] = xcorr_full_per_band
            df[new_column_label_xcorr_full_frequency_hz] = xcorr_full_frequency
    else:
        raise ValueError('ERROR: Incorrect reference station id')
    return df
