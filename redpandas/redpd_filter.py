"""
Utils for filtering pandas dataframes.
"""
import numpy as np
import pandas as pd
from scipy import signal
import redpandas.redpd_preprocess as rpd_prep

from typing import List, Tuple, Union


# Utils for filter modules
def prime_factors(n: int) -> List[int]:

    """
    Brute force code to find prime factors. Adapted from
    https://stackoverflow.com/questions/15347174/python-finding-prime-factors

    :param n: number
    :return: list of prime factors
    """
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if 1 < n < 13:
        factors.append(n)
    else:
        print('Can not be larger than 13')
    return factors


def decimate_individual_station(sig_wf: np.array,
                                sig_epoch_s: np.array,
                                downsampling_factor: int,
                                filter_order: int,
                                sample_rate_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decimate data and timestamps for an individual station

    :param sig_wf: signal waveform
    :param sig_epoch_s: signal timestamps
    :param downsampling_factor: the downsampling factor
    :param filter_order: the order of the filter
    :param sample_rate_hz: sample rate in Hz
    :return: np.array decimated timestamps, np.array decimated data
    """
    # decimate signal data
    decimate_data = signal.decimate(x=sig_wf,
                                    q=downsampling_factor,
                                    n=filter_order,
                                    ftype='iir',
                                    axis=0,
                                    zero_phase=True)

    # reconstruct signal timestamps from new sample rate hz
    new_sample_rate_hz = sample_rate_hz / downsampling_factor
    reconstruct_time_s = (np.arange(len(decimate_data)) / new_sample_rate_hz) + sig_epoch_s[0]

    return reconstruct_time_s, decimate_data


# Main filter modules
def signal_zero_mean_pandas(df: pd.DataFrame,
                            sig_wf_label: str,
                            new_column_label: str = 'zero_mean') -> pd.DataFrame:
    """
    Eliminate DC offset from all signals in df

    :param df: input pandas data frame
    :param sig_wf_label: string for column name with the waveform data in df
    :param new_column_label: string for new column containing zero mean signal data
    :return: original data frame with extra column containing zero mean signals
    """
    # label new column in df
    new_column_label_sig_data = new_column_label
    list_zero_mean_data = []  # list that will be converted to a column
    for n in df.index:
        if type(df[sig_wf_label][n]) == float:
            list_zero_mean_data.append(float("NaN"))
            continue
        if df[sig_wf_label][n].ndim == 1:
            list_zero_mean_data.append(df[sig_wf_label][n] - np.nanmean(df[sig_wf_label][n]))
        else:
            list_zero_mean_3c_data = []
            for index_dimension, _ in enumerate(df[sig_wf_label][n]):
                list_zero_mean_3c_data.append(df[sig_wf_label][n][index_dimension]
                                              - np.nanmean(df[sig_wf_label][n][index_dimension]))
            list_zero_mean_data.append(np.array(list_zero_mean_3c_data))   # append 3 channels sensor into 'main' list
    df[new_column_label_sig_data] = list_zero_mean_data

    return df


def taper_tukey_pandas(df: pd.DataFrame,
                       sig_wf_label: str,
                       fraction_cosine: float,
                       new_column_label_append: str = 'taper') -> pd.DataFrame:
    """
    Apply taper to all signals in df

    :param df: input pandas data frame
    :param sig_wf_label: string for column name with the waveform data in df
    :param fraction_cosine: fraction of the window inside the cosine tapered window, shared between the head and tail
    :param new_column_label_append: sig_label + string for new column containing signal tapered data
    :return: original data frame with added column for signal values with taper window
    """
    list_taper = []

    for row in df.index:
        if type(df[sig_wf_label][row]) == float:
            list_taper.append(float("NaN"))
            continue
        if df[sig_wf_label][row].ndim == 1:
            sig_data_window = (df[sig_wf_label][row] * signal.windows.tukey(M=len(df[sig_wf_label][row]),
                                                                            alpha=fraction_cosine,
                                                                            sym=True))
            list_taper.append(sig_data_window)
        else:
            list_taper_3c_data = []
            for index_dimension, _ in enumerate(df[sig_wf_label][row]):
                sig_data_window = (df[sig_wf_label][row][index_dimension]
                                   * signal.windows.tukey(M=len(df[sig_wf_label][row][index_dimension]),
                                                          alpha=fraction_cosine,
                                                          sym=True))
                list_taper_3c_data.append(sig_data_window)
            list_taper.append(np.array(list_taper_3c_data))  # append 3 channels sensor into 'main' list
    # label new column in df
    df[f"{sig_wf_label}_{new_column_label_append}"] = list_taper

    return df


def normalize_pandas(df: pd.DataFrame,
                     sig_wf_label: str,
                     scaling: float = 1.0,
                     norm_type: str = 'max',
                     new_column_label: str = 'normalized') -> pd.DataFrame:
    """
    Normalize all signals in df

    :param df: input pandas data frame
    :param sig_wf_label: string for column name with the waveform data in df
    :param scaling: scaling parameter, division
    :param norm_type: {'max', 'l1', 'l2'}, optional
    :param new_column_label:  string for label for new column containing normalized signal data
    :return: original data frame with added column for normalized signal data
    """
    if norm_type == 'max':
        norm_type_utils = rpd_prep.NormType.MAX
    elif norm_type == 'l1':
        norm_type_utils = rpd_prep.NormType.L1
    elif norm_type == 'l2':
        norm_type_utils = rpd_prep.NormType.L2
    else:
        norm_type_utils = rpd_prep.NormType.OTHER

    list_normalized_signals = []  # list that will be converted to a column
    for row in range(len(df)):
        if type(df[sig_wf_label][row]) == float:
            list_normalized_signals.append(float("NaN"))
            continue
        if df[sig_wf_label][row].ndim == 1:
            # use libquantum utils normalize module
            list_normalized_signals.append(rpd_prep.normalize(sig_wf=df[sig_wf_label][row], scaling=scaling,
                                                              norm_type=norm_type_utils))
        else:
            list_3c_normalized_signals = []
            for index_dimension, _ in enumerate(df[sig_wf_label][row]):
                list_3c_normalized_signals.append(rpd_prep.normalize(sig_wf=df[sig_wf_label][row][index_dimension],
                                                                     scaling=scaling,
                                                                     norm_type=norm_type_utils))

            list_normalized_signals.append(np.array(list_3c_normalized_signals))
    df[new_column_label] = list_normalized_signals

    return df


def decimate_signal_pandas(df: pd.DataFrame,
                           downsample_frequency_hz: Union[str, int],
                           sig_id_label: str,
                           sig_wf_label: str,
                           sig_timestamps_label: str,
                           sample_rate_hz_label: str,
                           filter_order: int = 8,
                           new_column_label_decimated_sig: str = 'decimated_sig_data',
                           new_column_label_decimated_sig_timestamps: str = 'decimated_sig_epoch',
                           new_column_label_decimated_sample_rate_hz: str = 'decimated_sample_rate_hz',
                           verbose: bool = False) -> pd.DataFrame:
    """
    Decimate all signal data (via spicy.signal.decimate). Decimates to the smallest sample rate recorded in data frame
    or to custom frequency.

    :param df: input pandas data frame
    :param downsample_frequency_hz: frequency to downsample to in Hz. For minimum frequency found in df, 'Min',
    for other frequencies, integer
    :param sig_id_label: string for column name with station ids in df
    :param sig_wf_label:  string for column name with the waveform data in df
    :param sig_timestamps_label: string for column name with the waveform timestamp data in df
    :param sample_rate_hz_label: string for column name with sample rate in Hz information in df
    :param filter_order: the order of the filter integer. Default is 2
    :param new_column_label_decimated_sig: label for new column containing signal decimated data
    :param new_column_label_decimated_sig_timestamps: label for new column containing signal decimated timestamps
    :param new_column_label_decimated_sample_rate_hz: label for new column containing signal decimated sample rate
    :param verbose: print statements. Default is False

    :return: original data frame with added columns for decimated signal, timestamps, and sample rate
    """
    # rate to downsample to is either the min sample rate in sample rate column or passed value
    min_sample_rate = \
        df[sample_rate_hz_label].min() if downsample_frequency_hz == 'Min' or downsample_frequency_hz == 'min' \
        else int(downsample_frequency_hz)

    if verbose:
        print(f'\nAll signals will de downsampled to (or as close to) {min_sample_rate} Hz \n')

    # list that will be converted to a columns added to the original df
    list_all_decimated_timestamps = []
    list_all_decimated_data = []
    list_all_decimated_sample_rate_hz = []

    for row in range(len(df)):  # for row in df
        if type(df[sig_wf_label][row]) == float:
            list_all_decimated_timestamps.append(float("NaN"))
            list_all_decimated_data.append(float("NaN"))
            list_all_decimated_sample_rate_hz.append(float("NaN"))
            if verbose:
                print(f'No data found for {df[sig_id_label][row]} {sig_wf_label}')
            continue

        if df[sample_rate_hz_label][row] != min_sample_rate:
            # calculate downsampling factor to reach downsampled frequency
            downsampling_factor = int(df[sample_rate_hz_label][row] / min_sample_rate)
            if downsampling_factor <= 1:
                if verbose:
                    print(f'{df[sig_id_label][row]} can not be downsampled to {min_sample_rate} Hz')
                # store the original timestamp/data/sample rate values
                list_all_decimated_timestamps.append(df[sig_timestamps_label][row])
                list_all_decimated_data.append(df[sig_wf_label][row])
                list_all_decimated_sample_rate_hz.append(df[sample_rate_hz_label][row])
            elif downsampling_factor <= 12:  # 13 is the max recommended, if larger, decimate in steps
                decimated_timestamp, decimated_data = \
                    decimate_individual_station(downsampling_factor=downsampling_factor,
                                                filter_order=filter_order,
                                                sig_epoch_s=df[sig_timestamps_label][row],
                                                sig_wf=df[sig_wf_label][row],
                                                sample_rate_hz=df[sample_rate_hz_label][row])
                if verbose:
                    print(f'{df[sig_id_label][row]} data downsampled to '
                          f'{df[sample_rate_hz_label][row] / downsampling_factor} Hz '
                          f'by downsampling factor of {downsampling_factor}')
                # store new decimated timestamp, data and sample rate
                list_all_decimated_timestamps.append(decimated_timestamp)
                list_all_decimated_data.append(decimated_data)
                list_all_decimated_sample_rate_hz.append(df[sample_rate_hz_label][row] / downsampling_factor)
            else:  # if downsampling factor larger than 13, decimate in steps
                list_prime_factors = prime_factors(downsampling_factor)  # find how many decimate 'steps' through primes
                # lists to temporarily store timestamps/data/sample rate decimated in between steps
                # fill temporary list with original timestamp/data/sample rate from df to work with
                # without changing the original
                list_temporary_timestamp_decimate_storage = [df[sig_timestamps_label][row]]
                list_temporary_data_decimate_storage = [df[sig_wf_label][row]]
                list_temporary_sample_rate_hz = [df[sample_rate_hz_label][row]]
                # loop through prime factors aka decimate steps
                for index_list_storage, prime in enumerate(list_prime_factors):
                    decimated_timestamp, decimated_data = \
                        decimate_individual_station(downsampling_factor=int(prime),
                                                    filter_order=filter_order,
                                                    sig_epoch_s=
                                                    list_temporary_timestamp_decimate_storage[index_list_storage],
                                                    sig_wf=list_temporary_data_decimate_storage[index_list_storage],
                                                    sample_rate_hz=list_temporary_sample_rate_hz[index_list_storage])
                    list_temporary_timestamp_decimate_storage.append(decimated_timestamp)
                    list_temporary_data_decimate_storage.append(decimated_data)
                    list_temporary_sample_rate_hz.append(list_temporary_sample_rate_hz[index_list_storage] / prime)
                    if verbose:
                        print(f'{df[sig_id_label][row]} data downsampled to '
                              f'{list_temporary_sample_rate_hz[index_list_storage] / prime} Hz '
                              f'by downsampling factor of {prime}')
                # once timestamps/data/sample rate decimated through all the steps (aka prime factors),
                # store in general list that will be converted to a df column
                # we want the last element of the temporary list aka the last decimated step
                list_all_decimated_timestamps.append(list_temporary_timestamp_decimate_storage[-1])
                list_all_decimated_data.append(list_temporary_data_decimate_storage[-1])
                list_all_decimated_sample_rate_hz.append(list_temporary_sample_rate_hz[-1])
        else:  # if no decimation necessary, store the original timestamp/data/sample rate values
            if verbose:
                print(f'{df[sig_id_label][row]} does not need to be downsampled')
            list_all_decimated_timestamps.append(df[sig_timestamps_label][row])
            list_all_decimated_data.append(df[sig_wf_label][row])
            list_all_decimated_sample_rate_hz.append(df[sample_rate_hz_label][row])
    # convert to columns and add it to df
    df[new_column_label_decimated_sig] = list_all_decimated_data
    df[new_column_label_decimated_sig_timestamps] = list_all_decimated_timestamps
    df[new_column_label_decimated_sample_rate_hz] = list_all_decimated_sample_rate_hz

    return df


def decimate_signal_pandas_audio_rdvx(df: pd.DataFrame,
                                      sig_wf_label: str = "audio_wf",
                                      sig_timestamps_label: str = "audio_epoch_s",
                                      sample_rate_hz_label: str = "audio_sample_rate_nominal_hz"):
    """
    Decimates signal data to 8kHz. Makes columns "decimated_audio_wf", "decimated_audio_epoch_s" and
    "decimated_audio_sample_rate_hz"

    :param df: input pandas dataframe
    :param sig_wf_label: Default is "audio_wf"
    :param sig_timestamps_label: Default is "audio_epoch_s"
    :param sample_rate_hz_label: Default is "audio_sample_rate_nominal_hz"
    :return: original df with new columns with decimated data
    """
    list_all_decimated_data = []
    list_all_decimated_timestamps = []
    list_all_decimated_sample_rate_hz = []
    for row in df.index:
        if df[sample_rate_hz_label][row] == 48000:
            decimated_timestamp, decimated_data = \
                decimate_individual_station(downsampling_factor=6,
                                            filter_order=8,
                                            sig_epoch_s=df[sig_timestamps_label][row],
                                            sig_wf=df[sig_wf_label][row],
                                            sample_rate_hz=df[sample_rate_hz_label][row])
            new_sample_rate_hz = df[sample_rate_hz_label][row] / 6

        elif df[sample_rate_hz_label][row] == 16000:
            decimated_timestamp, decimated_data = decimate_individual_station(
                downsampling_factor=2,
                filter_order=8,
                sig_epoch_s=df[sig_timestamps_label][row],
                sig_wf=df[sig_wf_label][row],
                sample_rate_hz=df[sample_rate_hz_label][row])
            new_sample_rate_hz = df[sample_rate_hz_label][row] / 2
        else:
            decimated_timestamp = df[sig_timestamps_label][row]
            decimated_data = df[sig_wf_label][row]
            new_sample_rate_hz = df[sample_rate_hz_label][row]
        list_all_decimated_data.append(decimated_data)
        list_all_decimated_timestamps.append(decimated_timestamp)
        list_all_decimated_sample_rate_hz.append(new_sample_rate_hz)

    df["decimated_audio_wf"] = list_all_decimated_data
    df["decimated_audio_epoch_s"] = list_all_decimated_timestamps
    df["decimated_audio_sample_rate_hz"] = list_all_decimated_sample_rate_hz

    return df


def bandpass_butter_pandas(df: pd.DataFrame,
                           sig_wf_label: str,
                           sig_sample_rate_label: str,
                           frequency_cut_low_hz: float,
                           frequency_cut_high_hz: float,
                           filter_order: int = 4,
                           tukey_alpha: float = 0.5,
                           new_column_label_sig_bandpass: str = 'bandpass',
                           new_column_label_frequency_low: str = 'frequency_low_hz',
                           new_column_label_frequency_high: str = 'frequency_high_hz') -> pd.DataFrame:
    """
    Apply a taper and a butterworth bandpass filter

    :param df: input pandas data frame
    :param sig_wf_label: string for the waveform column name in df
    :param sig_sample_rate_label: string for the sample rate in Hz column name in df
    :param frequency_cut_low_hz: low cutoff frequency in Hz
    :param frequency_cut_high_hz: high cutoff frequency in Hz
    :param filter_order: filter order is doubled with filtfilt, nominal 4 -> 8
    :param tukey_alpha: 0 = no taper, 1 = Hann taper. 0.25 is flat over 3/4 of sig, good for 75% overlap
    :param new_column_label_sig_bandpass: string for new column with bandpassed signal data
    :param new_column_label_frequency_low: string for new column
    :param new_column_label_frequency_high: string for new column
    :return: original df with added columns for band passed tapered signal, frequency high and low values
    """
    # lists to store arrays to convert to columns in df
    list_all_signal_bandpass_data = []
    list_all_frequency_low_hz = []
    list_all_frequency_high_hz = []

    # Frequencies are scaled by Nyquist, with 1 = Nyquist
    for j in df.index:
        if type(df[sig_wf_label][j]) == float or type(df[sig_sample_rate_label][j]) == float:
            list_all_signal_bandpass_data.append(float("NaN"))
            list_all_frequency_low_hz.append(float("NaN"))
            list_all_frequency_high_hz.append(float("NaN"))
            continue
        nyquist = 0.5 * df[sig_sample_rate_label][j]
        edge_low = frequency_cut_low_hz / nyquist
        edge_high = frequency_cut_high_hz / nyquist
        if edge_high >= 1:
            edge_high = 0.5  # Half of nyquist
        [b, a] = signal.butter(N=filter_order,
                               Wn=[edge_low, edge_high],
                               btype='bandpass')
        if df[sig_wf_label][j].ndim == 1:
            nyquist = 0.5 * df[sig_sample_rate_label][j]
            edge_low = frequency_cut_low_hz / nyquist
            edge_high = frequency_cut_high_hz / nyquist
            if edge_high >= 1:
                edge_high = 0.5  # Half of nyquist
            [b, a] = signal.butter(N=filter_order,
                                   Wn=[edge_low, edge_high],
                                   btype='bandpass')
            sig_taper = np.copy(df[sig_wf_label][j])
            sig_taper = sig_taper * signal.windows.tukey(M=len(sig_taper), alpha=tukey_alpha)
            sig_bandpass = signal.filtfilt(b, a, sig_taper)
            # Append to list
            list_all_signal_bandpass_data.append(sig_bandpass)
        else:
            list_3c_signal_bandpass_data = []
            for index_dimension, _ in enumerate(df[sig_wf_label][j]):
                sig_taper = np.copy(df[sig_wf_label][j][index_dimension])
                sig_taper = sig_taper * signal.windows.tukey(M=len(sig_taper), alpha=tukey_alpha)
                sig_bandpass = signal.filtfilt(b, a, sig_taper)

                list_3c_signal_bandpass_data.append(sig_bandpass)
            # Append to list
            list_all_signal_bandpass_data.append(np.array(list_3c_signal_bandpass_data))
        list_all_frequency_low_hz.append(frequency_cut_low_hz)
        list_all_frequency_high_hz.append(frequency_cut_high_hz)

    # Convert to columns and add it to df
    df[new_column_label_sig_bandpass] = list_all_signal_bandpass_data
    df[new_column_label_frequency_low] = list_all_frequency_low_hz
    df[new_column_label_frequency_high] = list_all_frequency_high_hz

    return df


def highpass_butter_pandas(df: pd.DataFrame,
                           sig_wf_label: str,
                           sig_sample_rate_label: str,
                           frequency_cut_low_hz: float,
                           frequency_cut_high_hz: float,
                           filter_order: int = 4,
                           tukey_alpha: float = 0.5,
                           new_column_label_sig_highpass: str = 'highpass',
                           new_column_label_frequency_low: str = 'frequency_low_hz',
                           new_column_label_frequency_high: str = 'frequency_high_hz') -> pd.DataFrame:
    """
    Apply a taper and a butterworth bandpass filter

    :param df: input pandas data frame
    :param sig_wf_label: string for the waveform column name in df
    :param sig_sample_rate_label: string for the sample rate column name in df
    :param frequency_cut_low_hz: low cutoff frequency in Hz
    :param frequency_cut_high_hz: high cutoff frequency in Hz
    :param filter_order: filter order is doubled with filtfilt, nominal 4 -> 8
    :param tukey_alpha: 0 = no taper, 1 = Hann taper. 0.25 is flat over 3/4 of sig, good for 75% overlap
    :param new_column_label_sig_highpass: string for new column with highpass signal data
    :param new_column_label_frequency_low: string for new column
    :param new_column_label_frequency_high: string for new column
    :return: original df with added columns for band passed tapered signal, frequency high and low values
    """
    # lists to store arrays to convert to columns in df
    list_all_signal_highpass_data = []
    list_all_frequency_low_hz = []
    list_all_frequency_high_hz = []

    # Frequencies are scaled by Nyquist, with 1 = Nyquist
    for j in df.index:
        if type(df[sig_wf_label][j]) == float or type(df[sig_sample_rate_label][j]) == float:
            list_all_signal_highpass_data.append(float("NaN"))
            list_all_frequency_low_hz.append(float("NaN"))
            list_all_frequency_high_hz.append(float("NaN"))
            continue
        nyquist = 0.5 * df[sig_sample_rate_label][j]
        edge_low = frequency_cut_low_hz / nyquist
        [b, a] = signal.butter(N=filter_order,
                               Wn=edge_low,
                               btype='high')
        if df[sig_wf_label][j].ndim == 1:
            sig_taper = np.copy(df[sig_wf_label][j])
            sig_taper = sig_taper * signal.windows.tukey(M=len(sig_taper), alpha=tukey_alpha)
            sig_highpass = signal.filtfilt(b, a, sig_taper)
            # Append to list
            list_all_signal_highpass_data.append(sig_highpass)
        else:
            list_3c_signal_highpass_data = []
            for index_dimension, _ in enumerate(df[sig_wf_label][j]):
                sig_taper = np.copy(df[sig_wf_label][j][index_dimension])
                sig_taper = sig_taper * signal.windows.tukey(M=len(sig_taper), alpha=tukey_alpha)
                sig_highpass = signal.filtfilt(b, a, sig_taper)
                list_3c_signal_highpass_data.append(sig_highpass)
            # Append to list
            list_all_signal_highpass_data.append(np.array(list_3c_signal_highpass_data))
        list_all_frequency_low_hz.append(frequency_cut_low_hz)
        list_all_frequency_high_hz.append(frequency_cut_high_hz)

    # Convert to columns and add it to df
    df[new_column_label_sig_highpass] = list_all_signal_highpass_data
    df[new_column_label_frequency_low] = list_all_frequency_low_hz
    df[new_column_label_frequency_high] = list_all_frequency_high_hz

    return df
