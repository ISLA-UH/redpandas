"""
Calculate Time Representation Frequency.
"""
from typing import Optional

import numpy as np
import pandas as pd
from libquantum import atoms, spectra, utils
import redpandas.redpd_preprocess as rpd_prep
from redpandas.redpd_preprocess import find_nearest_idx


def frame_panda_no_offset(df: pd.DataFrame,
                          sig_wf_label: str,
                          sig_epoch_s_label: str,
                          sig_epoch_s_start: float,
                          sig_epoch_s_end: float,
                          new_column_aligned_wf: str = 'sig_aligned_wf',
                          new_column_aligned_epoch: str = 'sig_aligned_epoch_s') -> pd.DataFrame:
    """
    Align signals in dataframe (no seconds offset)

    :param df: input pandas data frame
    :param sig_wf_label: string for the waveform column name in df
    :param sig_epoch_s_label: string for column name with the waveform timestamp (in epoch s) in df
    :param sig_epoch_s_start: first timestamp in epoch s
    :param sig_epoch_s_end: last timestamp in epoch s
    :param new_column_aligned_wf: label for new column containing aligned waveform
    :param new_column_aligned_epoch: label for new column containing aligned timestamps in epoch s
    :return: input df with new columns
    """
    aligned_wf = []
    aligned_epoch_s = []
    for n in df.index:
        if sig_wf_label not in df.columns or type(df[sig_wf_label][n]) == float:
            aligned_wf.append(float("NaN"))
            aligned_epoch_s.append(float("NaN"))
            continue
        if df[sig_wf_label][n].ndim == 1:
            sig_wf, sig_epoch_s = \
                utils.sig_frame(sig=df[sig_wf_label][n],
                                time_epoch_s=df[sig_epoch_s_label][n],
                                epoch_s_start=sig_epoch_s_start,
                                epoch_s_stop=sig_epoch_s_end)
            aligned_wf.append(sig_wf)
            aligned_epoch_s.append(sig_epoch_s)
        else:
            aligned_wf_3c = []
            sig_epoch_s = 0  # set just in case
            for index_sensor_array, _ in enumerate(df[sig_wf_label][n]):
                sig_wf, sig_epoch_s = \
                    utils.sig_frame(sig=df[sig_wf_label][n][index_sensor_array],
                                    time_epoch_s=df[sig_epoch_s_label][n],
                                    epoch_s_start=sig_epoch_s_start,
                                    epoch_s_stop=sig_epoch_s_end)
                aligned_wf_3c.append(sig_wf)
            aligned_wf.append(np.array(aligned_wf_3c))
            aligned_epoch_s.append(sig_epoch_s)
    df[new_column_aligned_wf] = aligned_wf
    df[new_column_aligned_epoch] = aligned_epoch_s

    return df


def frame_panda(df: pd.DataFrame,
                sig_wf_label: str,
                sig_epoch_s_label: str,
                sig_epoch_s_start: float,
                sig_epoch_s_end: float,
                offset_seconds_label: str = "xcorr_offset_seconds",
                new_column_aligned_wf: str = 'sig_aligned_wf',
                new_column_aligned_epoch: str = 'sig_aligned_epoch_s') -> pd.DataFrame:
    """
    Align signals in dataframe (with seconds offset)

    :param df: input pandas data frame
    :param sig_wf_label: string for the waveform column name in df
    :param sig_epoch_s_label: string for column name with the waveform timestamp (in epoch s) in df
    :param sig_epoch_s_start: first timestamp in epoch s
    :param sig_epoch_s_end: last timestamp in epoch s
    :param offset_seconds_label: time offset correction in seconds
    :param new_column_aligned_wf: label for new column containing aligned waveform
    :param new_column_aligned_epoch: label for new column containing aligned timestamps in epoch s
    :return: input df with new columns
    """
    aligned_wf = []
    aligned_epoch_s = []
    for n in df.index:
        if sig_wf_label not in df.columns or type(df[sig_wf_label][n]) == float:
            aligned_wf.append(float("NaN"))
            aligned_epoch_s.append(float("NaN"))
            continue
        sig_wf, sig_epoch_s = \
            utils.sig_frame(sig=df[sig_wf_label][n],
                            time_epoch_s=df[sig_epoch_s_label][n] + df[offset_seconds_label][n],
                            epoch_s_start=sig_epoch_s_start,
                            epoch_s_stop=sig_epoch_s_end)
        aligned_wf.append(sig_wf)
        aligned_epoch_s.append(sig_epoch_s)
    df[new_column_aligned_wf] = aligned_wf
    df[new_column_aligned_epoch] = aligned_epoch_s

    return df


# todo: don't repeat functions that can't be reused easily.  This function is a pointless as long as the window
# function exists
# INPUT ALIGNED DATA
def tfr_bits_panda(df: pd.DataFrame,
                   sig_wf_label: str,
                   sig_sample_rate_label: str,
                   order_number_input: float = 3,
                   tfr_type: str = 'cwt',
                   new_column_tfr_bits: str = 'tfr_bits',
                   new_column_tfr_time_s: str = 'tfr_time_s',
                   new_column_tfr_frequency_hz: str = 'tfr_frequency_hz') -> pd.DataFrame:
    """
    Calculate Time Frequency Representation for a signal

    :param df: input pandas data frame
    :param sig_wf_label: string for the waveform column name in df
    :param sig_sample_rate_label: string for column name with sample rate in Hz information in df
    :param order_number_input: band order Nth
    :param tfr_type: 'cwt' or 'stft'.  if not either option, uses 'cwt'.
    :param new_column_tfr_bits: label for new column containing tfr in bits
    :param new_column_tfr_time_s: label for new column containing tfr timestamps in epoch s
    :param new_column_tfr_frequency_hz: label for new column containing tfr frequency in Hz
    :return: input dataframe with new columns
    """
    tfr_bits = []
    tfr_time_s = []
    tfr_frequency_hz = []
    for n in df.index:
        if sig_wf_label not in df.columns or type(df[sig_wf_label][n]) == float:
            tfr_bits.append(float("NaN"))
            tfr_time_s.append(float("NaN"))
            tfr_frequency_hz.append(float("NaN"))
            continue
        if df[sig_wf_label][n].ndim == 1:  # audio basically
            sig_wf_n = np.copy(df[sig_wf_label][n])
            sig_wf_n *= rpd_prep.taper_tukey(sig_wf_or_time=sig_wf_n, fraction_cosine=0.1)
            if tfr_type == "stft":
                # Compute complex wavelet transform (cwt) from signal duration
                _, sig_bits, sig_time_s, sig_frequency_hz = \
                    spectra.stft_from_sig(sig_wf=sig_wf_n,
                                          frequency_sample_rate_hz=df[sig_sample_rate_label][n],
                                          band_order_Nth=order_number_input)
            else:
                # Compute complex wavelet transform (cwt) from signal duration
                _, sig_bits, sig_time_s, sig_frequency_hz = \
                    atoms.cwt_chirp_from_sig(sig_wf=sig_wf_n,
                                             frequency_sample_rate_hz=df[sig_sample_rate_label][n],
                                             band_order_Nth=order_number_input)
            tfr_bits.append(sig_bits)
            tfr_time_s.append(sig_time_s)
            tfr_frequency_hz.append(sig_frequency_hz)
        else:  # sensor that is acceleration/gyroscope/magnetometer/barometer
            tfr_3c_bits = []
            tfr_3c_time = []
            tfr_3c_frequency = []
            for index_dimension, _ in enumerate(df[sig_wf_label][n]):
                sig_wf_n = np.copy(df[sig_wf_label][n][index_dimension])
                sig_wf_n *= rpd_prep.taper_tukey(sig_wf_or_time=sig_wf_n, fraction_cosine=0.1)
                if tfr_type == "stft":
                    # Compute complex wavelet transform (cwt) from signal duration
                    _, sig_bits, sig_time_s, sig_frequency_hz = \
                        spectra.stft_from_sig(sig_wf=sig_wf_n,
                                              frequency_sample_rate_hz=df[sig_sample_rate_label][n],
                                              band_order_Nth=order_number_input)
                else:
                    # Compute complex wavelet transform (cwt) from signal duration
                    _, sig_bits, sig_time_s, sig_frequency_hz = \
                        atoms.cwt_chirp_from_sig(sig_wf=sig_wf_n,
                                                 frequency_sample_rate_hz=df[sig_sample_rate_label][n],
                                                 band_order_Nth=order_number_input)
                tfr_3c_bits.append(sig_bits)
                tfr_3c_time.append(sig_time_s)
                tfr_3c_frequency.append(sig_frequency_hz)
            # append 3c tfr into 'main' list
            tfr_bits.append(np.array(tfr_3c_bits))
            tfr_time_s.append(np.array(tfr_3c_time))
            tfr_frequency_hz.append(np.array(tfr_3c_frequency))
    df[new_column_tfr_bits] = tfr_bits
    df[new_column_tfr_time_s] = tfr_time_s
    df[new_column_tfr_frequency_hz] = tfr_frequency_hz

    return df


# INPUT ALIGNED DATA
def tfr_bits_panda_window(df: pd.DataFrame,
                          sig_wf_label: str,
                          sig_sample_rate_label: str,
                          sig_timestamps_label: str,
                          order_number_input: float = 3,
                          tfr_type: str = 'cwt',
                          new_column_tfr_bits: str = 'tfr_bits',
                          new_column_tfr_time_s: str = 'tfr_time_s',
                          new_column_tfr_frequency_hz: str = 'tfr_frequency_hz',
                          start_time_window: Optional[float] = 0.0,
                          end_time_window: Optional[float] = 0.0) -> pd.DataFrame:
    """
    Calculate Time Frequency Representation for a signal within a time window

    :param df: input pandas data frame
    :param sig_wf_label: string for the waveform column name in df
    :param sig_sample_rate_label: string for column name with sample rate in Hz information in df
    :param sig_timestamps_label: string for timestamp column name in df
    :param order_number_input: band order Nth
    :param tfr_type: 'cwt' or 'stft'  if not either option, uses 'cwt'.
    :param new_column_tfr_bits: label for new column containing tfr in bits
    :param new_column_tfr_time_s: label for new column containing tfr timestamps in epoch s
    :param new_column_tfr_frequency_hz: label for new column containing tfr frequency in Hz
    :param start_time_window: float, start time window (within sig_timestamps_label)
    :param end_time_window: float, end time window (within sig_timestamps_label)
    :return: input dataframe with new columns
    """
    tfr_bits = []
    tfr_time_s = []
    tfr_frequency_hz = []
    # Check zooming window
    if start_time_window > 0.0 and end_time_window > 0.0:
        if end_time_window <= start_time_window:
            raise ValueError(f"end_time_window parameter ('{end_time_window}') "
                             f"cannot be smaller than start_time_window parameter ('{start_time_window}')")
    for n in df.index:
        if sig_wf_label not in df.columns or type(df[sig_wf_label][n]) == float:
            tfr_bits.append(float("NaN"))
            tfr_time_s.append(float("NaN"))
            tfr_frequency_hz.append(float("NaN"))
            continue
        if df[sig_wf_label][n].ndim == 1:  # audio basically
            timestamps = df[sig_timestamps_label][n]
            if start_time_window > 0.0 and end_time_window > 0.0:
                idx_time_start = find_nearest_idx(timestamps, start_time_window)
                idx_time_end = find_nearest_idx(timestamps, end_time_window)
            elif start_time_window > 0.0 and end_time_window == 0.0:
                idx_time_start = find_nearest_idx(timestamps, start_time_window)
                idx_time_end = -1
            elif end_time_window > 0.0 and start_time_window == 0.0:
                idx_time_start = 0
                idx_time_end = find_nearest_idx(timestamps, end_time_window)
            else:
                idx_time_start = 0
                idx_time_end = -1
            sig_wf = df[sig_wf_label][n][idx_time_start:idx_time_end]
            sig_wf_n = np.copy(sig_wf)
            sig_wf_n *= rpd_prep.taper_tukey(sig_wf_or_time=sig_wf_n, fraction_cosine=0.1)
            if tfr_type == "stft":
                # Compute complex wavelet transform (cwt) from signal duration
                _, sig_bits, sig_time_s, sig_frequency_hz = \
                    spectra.stft_from_sig(sig_wf=sig_wf_n,
                                          frequency_sample_rate_hz=df[sig_sample_rate_label][n],
                                          band_order_Nth=order_number_input)
            else:
                # Compute complex wavelet transform (cwt) from signal duration
                _, sig_bits, sig_time_s, sig_frequency_hz = \
                    atoms.cwt_chirp_from_sig(sig_wf=sig_wf_n,
                                             frequency_sample_rate_hz=df[sig_sample_rate_label][n],
                                             band_order_Nth=order_number_input)
            tfr_bits.append(sig_bits)
            tfr_time_s.append(sig_time_s)
            tfr_frequency_hz.append(sig_frequency_hz)
        else:  # sensor that is acceleration/gyroscope/magnetometer/barometer
            tfr_3c_bits = []
            tfr_3c_time = []
            tfr_3c_frequency = []
            for index_dimension, _ in enumerate(df[sig_wf_label][n]):
                sig_wf_n = np.copy(df[sig_wf_label][n][index_dimension])
                sig_wf_n *= rpd_prep.taper_tukey(sig_wf_or_time=sig_wf_n, fraction_cosine=0.1)
                if tfr_type == "stft":
                    # Compute complex wavelet transform (cwt) from signal duration
                    _, sig_bits, sig_time_s, sig_frequency_hz = \
                        spectra.stft_from_sig(sig_wf=sig_wf_n,
                                              frequency_sample_rate_hz=df[sig_sample_rate_label][n],
                                              band_order_Nth=order_number_input)
                else:
                    # Compute complex wavelet transform (cwt) from signal duration
                    _, sig_bits, sig_time_s, sig_frequency_hz = \
                        atoms.cwt_chirp_from_sig(sig_wf=sig_wf_n,
                                                 frequency_sample_rate_hz=df[sig_sample_rate_label][n],
                                                 band_order_Nth=order_number_input)
                tfr_3c_bits.append(sig_bits)
                tfr_3c_time.append(sig_time_s)
                tfr_3c_frequency.append(sig_frequency_hz)
            # append 3c tfr into 'main' list
            tfr_bits.append(np.array(tfr_3c_bits))
            tfr_time_s.append(np.array(tfr_3c_time))
            tfr_frequency_hz.append(np.array(tfr_3c_frequency))
    df[new_column_tfr_bits] = tfr_bits
    df[new_column_tfr_time_s] = tfr_time_s
    df[new_column_tfr_frequency_hz] = tfr_frequency_hz

    return df
