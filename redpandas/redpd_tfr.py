"""
This module contains panda versions of libquantum.
"""

import numpy as np
import pandas as pd
from libquantum import atoms, entropy, scales, spectra, utils, synthetics
import redpandas.redpd_preprocess as rpd_prep

def frame_panda(df: pd.DataFrame,
                sig_wf_label: str,
                sig_epoch_s_label: str,
                sig_epoch_s_start: float,
                sig_epoch_s_end: float,
                offset_seconds_label: str = "xcorr_offset_seconds",
                new_column_aligned_wf: str = 'sig_aligned_wf',
                new_column_aligned_epoch: str = 'sig_aligned_epoch_s'):

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
                   sig_epoch_s_label: str,
                   sig_sample_rate_label: str,
                   order_number_input: float = 3,
                   tfr_type: str = 'cwt',
                   new_column_tfr_bits: str = 'tfr_bits',
                   new_column_tfr_time_s: str = 'tfr_time_s',
                   new_column_tfr_frequency_hz: str = 'tfr_frequency_hz'):

    # TODO: Export epoch?
    tfr_bits = []
    tfr_time_s = []
    tfr_frequency_hz = []
    
    for n in df.index:
        sig_wf_n = np.copy(df[sig_wf_label][n])
        sig_wf_n *= rpd_prep.taper_tukey(sig_or_time=sig_wf_n, fraction_cosine=0.1)
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

    df[new_column_tfr_bits] = tfr_bits
    df[new_column_tfr_time_s] = tfr_time_s
    df[new_column_tfr_frequency_hz] = tfr_frequency_hz

    return df
