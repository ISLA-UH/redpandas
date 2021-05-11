import numpy as np
import pandas as pd
from scipy import signal
from libquantum import utils

"""
Utils for managing pandas dataframes

Last updated: 31 March 2021
"""
# TODO: to add lowpass and highpass filters
# TODO: Add HP uneven sensor resampling
# TODO: standardize language


# Utils for filter modules
def prime_factors(n: int):  # from https://stackoverflow.com/questions/15347174/python-finding-prime-factors

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


def decimate_individual_station(downsampling_factor: int,
                                filter_order: int,
                                sig_epoch_s: np.array,
                                sig_data: np.array,
                                sample_rate_hz: float):
    """
    Decimate data and timestamps for an individual station

    :param downsampling_factor:
    :param filter_order:
    :param sig_epoch_s:
    :param sig_data:
    :param sample_rate_hz:

    :return: np.array decimated timestamps, np.array decimated data
    """
    # decimate signal data
    decimate_data = signal.decimate(x=sig_data,
                                    q=downsampling_factor,
                                    n=filter_order,
                                    ftype='iir',
                                    axis=0,
                                    zero_phase=True)

    # reconstruct signal timestamps from new sample rate hz
    t0 = sig_epoch_s[0]
    new_sample_rate_hz = sample_rate_hz/downsampling_factor
    reconstruct_time_s = (np.arange(len(decimate_data)) / new_sample_rate_hz) + t0

    return reconstruct_time_s, decimate_data


# Main filter modules
def signal_zero_mean_pandas(df: pd.DataFrame,
                            sig_label: str,
                            new_column_label_append: str = 'zero_mean'):
    """
    Eliminate DC offset from all signals in df

    :param df: input pandas data frame
    :param sig_label: string for column name with the waveform data in df
    :param new_column_label_append: sig_label + string for new column containing zero mean signal data

    :return: original data frame with extra column containing zero mean signals (df['signal_zero_mean'])
    """
    # label new column in df
    new_column_label_sig_data = sig_label + "_" + new_column_label_append

    list_zero_mean_data = []  # list that will be converted to a column
    for row in range(len(df)):
        list_zero_mean_data.append(df[sig_label][row] - np.nanmean(df[sig_label][row]))

    df[new_column_label_sig_data] = list_zero_mean_data

    return df


def taper_tukey_pandas(df: pd.DataFrame,
                       sig_label: str,
                       fraction_cosine: float,
                       new_column_label_append: str = 'taper'):
    """
    Apply taper to all signals in df

    :param df: input pandas data frame
    :param sig_label: string for column name with the waveform data in df
    :param fraction_cosine: fraction of the window inside the cosine tapered window, shared between the head and tail
    :param new_column_label_append: sig_label + string for new column containing signal tapered data

    :return: original data frame with added column for signal values with taper window
    """
    # label new column in df
    new_column_label_taper_data = sig_label + "_" + new_column_label_append

    list_taper = []

    for row in range(len(df)):
        sig_data_window = (df[sig_label][row] * signal.windows.tukey(M=len(df[sig_label][row]),
                                                                     alpha=fraction_cosine,
                                                                     sym=True))

        list_taper.append(sig_data_window)

    df[new_column_label_taper_data] = list_taper

    return df


def normalize_pandas(df: pd.DataFrame,
                     sig_wf_label: str,
                     scaling: float = 1.0,
                     norm_type: str = 'max',
                     new_column_label: str = 'normalized'):
    """
    Normalize all signals in df
    # TODO: Issue with norm class in libquantum, please review (Tyler/Anthony)
    :param df: input pandas data frame
    :param sig_wf_label: string for column name with the waveform data in df
    :param scaling:
    :param norm_type: {'max', 'l1', 'l2'}, optional
    :param new_column_label:  string for label for new column containing normalized signal data
    :return: original data frame with added column for normalized signal data
    """

    if norm_type == 'max':
        norm_type_utils = utils.NormType.MAX
    elif norm_type == 'l1':
        norm_type_utils = utils.NormType.L1
    elif norm_type == 'l2':
        norm_type_utils = utils.NormType.L2
    else:
        norm_type_utils = utils.NormType.OTHER

    list_normalized_signals = []  # list that will be converted to a column
    for row in range(len(df)):
        # use libquantum utils normalize module
        list_normalized_signals.append(utils.normalize(sig=df[sig_wf_label][row], scaling=scaling,
                                                       norm_type=norm_type_utils))

    df[new_column_label] = list_normalized_signals

    return df


def selected_decimate_signal_pandas(df: pd.DataFrame,
                                    list_stations: list,
                                    sig_id_label,
                                    sig_wf_label: str,
                                    sig_timestamps_label: str,
                                    sample_rate_hz_label: str,
                                    downsample_frequency_hz: str or int,
                                    filter_order: int = 2,
                                    new_column_label_decimated_sig: str = 'decimated_sig_data',
                                    new_column_label_decimated_sig_timestamps: str = 'decimated_sig_epoch',
                                    new_column_label_decimated_sample_rate_hz: str = 'decimated_sample_rate_hz',
                                    verbose: bool = True):
    """
    Decimate signal data ( via spicy.signal.decimate) and timestamp data (via calculating timestamps 'bin' where signal
    data is decimated) for specified stations, copies the original timestamp and data for the rest of stations.

    :param df: input pandas data frame
    :param list_stations: list of strings of name of stations as saved in df
    :param sig_id_label: string for column name with station ids in df
    :param sig_wf_label:  string for column name with the waveform data in df
    :param sig_timestamps_label: string for column name with the waveform timestamp data in df
    :param sample_rate_hz_label: string for column name with sample rate in Hz information in df
    :param downsample_frequency_hz: frequency to downsample to in Hz. For minimum frequency found in df, 'Min',
    for other frequencies, integer
    :param filter_order: the order of the filter integer. Default is 2
    :param new_column_label_decimated_sig: string for new column containing signal decimated data
    :param new_column_label_decimated_sig_timestamps: string for new column containing signal decimated timestamps
    :param new_column_label_decimated_sample_rate_hz: string for new column containing signal decimated sample rate
    :param verbose: Default is True

    :return: original data frame with added columns for decimated signal, timestamps, and sample rate
    """
    # select frequency to downsample to
    if downsample_frequency_hz == 'Min' or downsample_frequency_hz == 'min':
        min_sample_rate = df[sample_rate_hz_label].min()  # find min sample rate in sample rate column
    else:
        min_sample_rate = int(downsample_frequency_hz)

    if verbose:
        print(f'\n{list_stations} will de downsampled to (or as close to) {min_sample_rate} Hz \n')

    # lists that will be converted to columns added to the original df
    list_all_decimated_timestamps = []
    list_all_decimated_data = []
    list_all_decimated_sample_rate_hz = []

    # make a list with requested stations the same length as rows in df, easier for looping
    list_zeros = ['no_station'] * (len(df)-len(list_stations))  # zeros list
    list_complete_zeros_stations = list_zeros + list_stations  # complete list

    for row in range(len(df)):

        list_tuples = []  # list to store tuples
        for station in list_complete_zeros_stations:
            # if station list and df match, store as True, otherwise False
            list_tuples.append((str(df[sig_id_label][row]) == str(station)))

        # if any is True, decimate. This assumes only 1 True (aka one match between list and df) per row in df
        if any(list_tuples) is True:

            if df[sample_rate_hz_label][row] != min_sample_rate:
                # calculate downsampling factor to reach downsampled frequency
                downsampling_factor = int(df[sample_rate_hz_label][row]/min_sample_rate)

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
                                                    sig_data=df[sig_wf_label][row],
                                                    sample_rate_hz=df[sample_rate_hz_label][row])

                    if verbose:
                        print(f'{df[sig_id_label][row]} data downsampled to '
                              f'{df[sample_rate_hz_label][row] / downsampling_factor} Hz '
                              f'by downsampling factor of {downsampling_factor}')

                    # store new decimated timestamp, data and sample rate
                    list_all_decimated_timestamps.append(decimated_timestamp)
                    list_all_decimated_data.append(decimated_data)
                    list_all_decimated_sample_rate_hz.append(df[sample_rate_hz_label][row]/downsampling_factor)

                else:  # if downsampling factor larger than 13, decimate in steps
                    # find how many decimate 'steps' through primes
                    list_prime_factors = prime_factors(downsampling_factor)

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
                                                        sig_epoch_s=list_temporary_timestamp_decimate_storage[index_list_storage],
                                                        sig_data=list_temporary_data_decimate_storage[index_list_storage],
                                                        sample_rate_hz=list_temporary_sample_rate_hz[index_list_storage])

                        list_temporary_timestamp_decimate_storage.append(decimated_timestamp)
                        list_temporary_data_decimate_storage.append(decimated_data)
                        list_temporary_sample_rate_hz.append(list_temporary_sample_rate_hz[index_list_storage]/prime)

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

        else:  # if no list station and df station match, store the original signal timestamp and data
            list_all_decimated_timestamps.append(df[sig_timestamps_label][row])
            list_all_decimated_data.append(df[sig_wf_label][row])
            list_all_decimated_sample_rate_hz.append(df[sample_rate_hz_label][row])

    # convert to columns and add it to df
    df[new_column_label_decimated_sig_timestamps] = list_all_decimated_timestamps
    df[new_column_label_decimated_sig] = list_all_decimated_data
    df[new_column_label_decimated_sample_rate_hz] = list_all_decimated_sample_rate_hz

    return df


def decimate_signal_pandas(df: pd.DataFrame,
                           downsample_frequency_hz: str or int,
                           sig_id_label: str,
                           sig_wf_label: str,
                           sig_timestamps_label: str,
                           sample_rate_hz_label: str,
                           filter_order: int = 8,
                           new_column_label_decimated_sig: str = 'decimated_sig_data',
                           new_column_label_decimated_sig_timestamps: str = 'decimated_sig_epoch',
                           new_column_label_decimated_sample_rate_hz: str = 'decimated_sample_rate_hz',
                           verbose: bool = True):
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
    :param verbose: print statements. Default is True

    :return: original data frame with added columns for decimated signal, timestamps, and sample rate
    """
    # select frequency to downsample to
    if downsample_frequency_hz == 'Min' or downsample_frequency_hz == 'min':
        min_sample_rate = df[sample_rate_hz_label].min()  # find min sample rate in sample rate column
    else:
        min_sample_rate = int(downsample_frequency_hz)

    if verbose:
        print(f'\nAll signals will de downsampled to (or as close to) {min_sample_rate} Hz \n')

    # list that will be converted to a columns added to the original df
    list_all_decimated_timestamps = []
    list_all_decimated_data = []
    list_all_decimated_sample_rate_hz = []

    for row in range(len(df)):  # for row in df

        if df[sample_rate_hz_label][row] != min_sample_rate:
            # calculate downsampling factor to reach downsampled frequency
            downsampling_factor = int(df[sample_rate_hz_label][row]/min_sample_rate)

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
                                                sig_data=df[sig_wf_label][row],
                                                sample_rate_hz=df[sample_rate_hz_label][row])

                if verbose:
                    print(f'{df[sig_id_label][row]} data downsampled to '
                          f'{df[sample_rate_hz_label][row] / downsampling_factor} Hz '
                          f'by downsampling factor of {downsampling_factor}')

                # store new decimated timestamp, data and sample rate
                list_all_decimated_timestamps.append(decimated_timestamp)
                list_all_decimated_data.append(decimated_data)
                list_all_decimated_sample_rate_hz.append(df[sample_rate_hz_label][row]/downsampling_factor)

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
                                                    sig_epoch_s=list_temporary_timestamp_decimate_storage[index_list_storage],
                                                    sig_data=list_temporary_data_decimate_storage[index_list_storage],
                                                    sample_rate_hz=list_temporary_sample_rate_hz[index_list_storage])

                    list_temporary_timestamp_decimate_storage.append(decimated_timestamp)
                    list_temporary_data_decimate_storage.append(decimated_data)
                    list_temporary_sample_rate_hz.append(list_temporary_sample_rate_hz[index_list_storage]/prime)

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


def bandpass_butter_pandas(df: pd.DataFrame,
                           sig_wf_label: str,
                           sig_sample_rate_label: str,
                           frequency_cut_low_hz: float,
                           frequency_cut_high_hz: float,
                           filter_order: int = 4,
                           tukey_alpha: float = 0.5,
                           new_column_label_sig_bandpass: str = 'bandpass',
                           new_column_label_frequency_low: str = 'frequency_low_hz',
                           new_column_label_frequency_high: str = 'frequency_high_hz'):
    """
    Apply a taper and a butterworth bandpass filter

    :param df: input pandas data frame
    :param sig_wf_label: string for the waveform name in df
    :param sig_sample_rate_label: string for the sample rate name in df
    :param frequency_cut_low_hz: low cutoff
    :param frequency_cut_high_hz: high cutoff
    :param filter_order: filter order is doubled with filtfilt, nominal 4 -> 8
    :param tukey_alpha: 0 = no taper, 1 = Hann taper. 0.25 is flat over 3/4 of sig, good for 75% overlap
    :param new_column_label_sig_bandpass: sig_label + string for new column with bandpassed signal data
    :param new_column_label_frequency_low: sig_label + string for new column
    :param new_column_label_frequency_high: sig_label + string for new column

    :return: original df with added columns for band passed tapered signal, frequency high and low values
    """

    # lists to store arrays to convert to columns in df
    list_all_signal_bandpass_data = []
    list_all_frequency_low_hz = []
    list_all_frequency_high_hz = []

    # Frequencies are scaled by Nyquist, with 1 = Nyquist
    for j in df.index:
        # TODO: check for j out of sorts
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
                           new_column_label_frequency_high: str = 'frequency_high_hz'):
    """
    Apply a taper and a butterworth bandpass filter

    :param df: input pandas data frame
    :param sig_wf_label: string for the waveform name in df
    :param sig_sample_rate_label: string for the sample rate name in df
    :param frequency_cut_low_hz: low cutoff
    :param frequency_cut_high_hz: high cutoff
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
        nyquist = 0.5 * df[sig_sample_rate_label][j]
        edge_low = frequency_cut_low_hz / nyquist
        [b, a] = signal.butter(N=filter_order,
                               Wn=edge_low,
                               btype='high')
        sig_taper = np.copy(df[sig_wf_label][j])
        sig_taper = sig_taper * signal.windows.tukey(M=len(sig_taper), alpha=tukey_alpha)
        sig_highpass = signal.filtfilt(b, a, sig_taper)

        # Append to list
        list_all_signal_highpass_data.append(sig_highpass)
        list_all_frequency_low_hz.append(frequency_cut_low_hz)
        list_all_frequency_high_hz.append(frequency_cut_high_hz)

    # Convert to columns and add it to df
    df[new_column_label_sig_highpass] = list_all_signal_highpass_data
    df[new_column_label_frequency_low] = list_all_frequency_low_hz
    df[new_column_label_frequency_high] = list_all_frequency_high_hz

    return df
