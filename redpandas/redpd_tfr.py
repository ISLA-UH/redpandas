"""
This module contains functions to calculate Time Representation Frequency.
"""

import numpy as np
import pandas as pd
from libquantum import atoms, spectra, utils
import redpandas.redpd_preprocess as rpd_prep


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
        sig_wf, sig_epoch_s = \
            utils.sig_frame(sig=df[sig_wf_label][n],
                            time_epoch_s=df[sig_epoch_s_label][n],
                            epoch_s_start=sig_epoch_s_start,
                            epoch_s_stop=sig_epoch_s_end)
        aligned_wf.append(sig_wf)
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
    :param tfr_type: 'cwt' or 'stft'
    :param new_column_tfr_bits: label for new column containing tfr in bits
    :param new_column_tfr_time_s: label for new column containing tfr timestamps in epoch s
    :param new_column_tfr_frequency_hz: label for new column containing tfr frequency in Hz
    :return: input dataframe with new columns
    """

    tfr_bits = []
    tfr_time_s = []
    tfr_frequency_hz = []

    for n in df.index:

        if df[sig_wf_label][n].ndim == 1:  # audio basically

            sig_wf_n = np.copy(df[sig_wf_label][n])
            sig_wf_n *= rpd_prep.taper_tukey(sig_wf_or_time=sig_wf_n, fraction_cosine=0.1)

            if tfr_type == "cwt":
                # Compute complex wavelet transform (cwt) from signal duration
                sig_cwt, sig_cwt_bits, sig_cwt_time_s, sig_cwt_frequency_hz = \
                    atoms.cwt_chirp_from_sig(sig_wf=sig_wf_n,
                                             frequency_sample_rate_hz=df[sig_sample_rate_label][n],
                                             band_order_Nth=order_number_input)

                tfr_bits.append(sig_cwt_bits)
                tfr_time_s.append(sig_cwt_time_s)
                tfr_frequency_hz.append(sig_cwt_frequency_hz)

            if tfr_type == "stft":
                # Compute complex wavelet transform (cwt) from signal duration
                sig_stft, sig_stft_bits, sig_stft_time_s, sig_stft_frequency_hz = \
                    spectra.stft_from_sig(sig_wf=sig_wf_n,
                                          frequency_sample_rate_hz=df[sig_sample_rate_label][n],
                                          band_order_Nth=order_number_input)

                tfr_bits.append(sig_stft_bits)
                tfr_time_s.append(sig_stft_time_s)
                tfr_frequency_hz.append(sig_stft_frequency_hz)

        else:  # sensor that is acceleration/gyroscope/magnetometer/barometer
            tfr_3c_bits = []
            tfr_3c_time = []
            tfr_3c_frequency = []
            for index_dimension, _ in enumerate(df[sig_wf_label][n]):

                sig_wf_n = np.copy(df[sig_wf_label][n][index_dimension])
                sig_wf_n *= rpd_prep.taper_tukey(sig_wf_or_time=sig_wf_n, fraction_cosine=0.1)

                if tfr_type == "cwt":
                    # Compute complex wavelet transform (cwt) from signal duration
                    sig_cwt, sig_cwt_bits, sig_cwt_time_s, sig_cwt_frequency_hz = \
                        atoms.cwt_chirp_from_sig(sig_wf=sig_wf_n,
                                                 frequency_sample_rate_hz=df[sig_sample_rate_label][n],
                                                 band_order_Nth=order_number_input)
                    tfr_3c_bits.append(sig_cwt_bits)
                    tfr_3c_time.append(sig_cwt_time_s)
                    tfr_3c_frequency.append(sig_cwt_frequency_hz)

                if tfr_type == "stft":
                    # Compute complex wavelet transform (cwt) from signal duration
                    sig_stft, sig_stft_bits, sig_stft_time_s, sig_stft_frequency_hz = \
                        spectra.stft_from_sig(sig_wf=sig_wf_n,
                                              frequency_sample_rate_hz=df[sig_sample_rate_label][n],
                                              band_order_Nth=order_number_input)
                    tfr_3c_bits.append(sig_stft_bits)
                    tfr_3c_time.append(sig_stft_time_s)
                    tfr_3c_frequency.append(sig_stft_frequency_hz)

            # append 3c tfr into 'main' list
            tfr_bits.append(np.array(tfr_3c_bits))
            tfr_time_s.append(np.array(tfr_3c_time))
            tfr_frequency_hz.append(np.array(tfr_3c_frequency))

    df[new_column_tfr_bits] = tfr_bits
    df[new_column_tfr_time_s] = tfr_time_s
    df[new_column_tfr_frequency_hz] = tfr_frequency_hz

    return df
