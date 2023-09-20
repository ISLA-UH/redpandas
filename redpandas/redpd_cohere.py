"""
Calculate coherence.
"""

import numpy as np
import pandas as pd
from scipy import signal
from libquantum import utils
import redpandas.redpd_plot.coherence as rpd_plt


def coherence_numpy(sig_in: np.ndarray,
                    sig_in_ref: np.ndarray,
                    sig_sample_rate_hz: int,
                    sig_ref_sample_rate_hz: int,
                    window_seconds: float = 2.,
                    window_overlap_fractional: float = 0.5,
                    frequency_ref_hz: float = 40.,
                    frequency_min_hz: float = 1.,
                    frequency_max_hz: float = 320.,
                    sig_calib: float = 1.,
                    sig_ref_calib: float = 1.):
    """
    Find coherence between a signal and a reference signal, plot results

    :param sig_in: signal
    :param sig_in_ref: reference signal
    :param sig_sample_rate_hz: sample rate of signal in Hz
    :param sig_ref_sample_rate_hz: sample rate of reference signal in Hz
    :param window_seconds: seconds duration of window. Default is 2.0
    :param window_overlap_fractional: number of points to overlap between segments in window. Default is 0.5
    :param frequency_ref_hz: reference frequency in Hz. Default is 40.0
    :param frequency_min_hz: minimum frequency to plot in Hz (x min limit). Default is 1.0
    :param frequency_max_hz: maximum frequency to plot in Hz (x max limit). Default is 320.
    :param sig_calib: calibration of signal. Default is 1.0
    :param sig_ref_calib: calibration of reference signal. Default is 1.0
    :return: plots
    """
    # Stated with WACT IMS ref code, increased consistency.
    # Core computation is standard scipy.signal.

    # Compute PSDs and response - /4 calib divider removed
    sig_ref = np.copy(sig_in_ref) * sig_ref_calib
    sig = np.copy(sig_in) * sig_calib

    window_points = int(window_seconds*sig_sample_rate_hz)
    window_overlap_points = int(window_overlap_fractional*window_points)

    # Compute PSDs for each and coherence between the two
    f, pxx_ref = signal.welch(x=sig_ref,
                              fs=sig_ref_sample_rate_hz,
                              nperseg=window_points,
                              noverlap=window_overlap_points)
    f, pxx_sig = signal.welch(x=sig,
                              fs=sig_sample_rate_hz,
                              nperseg=window_points,
                              noverlap=window_overlap_points)

    # dB, absolute -  add EPSILON
    psd_ref_bits = 0.5 * np.log2(abs(pxx_ref))
    psd_sig_bits = 0.5 * np.log2(abs(pxx_sig))

    # Compute cross-power spectral density with ref sample rate
    # Original code had no overlap - fixed

    f, pxy = signal.csd(x=sig,
                        y=sig_ref,
                        fs=sig_ref_sample_rate_hz,
                        nperseg=window_points,
                        noverlap=window_overlap_points)
    cross_spectrum_bits = 0.5 * np.log2(abs(pxy))

    # Coherence, same as from PSD
    f, cxy = signal.coherence(x=sig,
                              y=sig_ref,
                              fs=sig_ref_sample_rate_hz,
                              nperseg=window_points,
                              noverlap=window_overlap_points)

    # Compute response assuming incoherent comp in ref.
    # Ref sensor response already removed
    h_x = pxx_sig / pxy

    # compute magnitude and phase in deg
    mag = np.abs(h_x)
    ph = np.unwrap(180 / np.pi * np.angle(h_x))

    # get new mag and phase values at frequency closest to ref frequency
    frequency_ref_index = np.argmin(np.abs(f - frequency_ref_hz))
    frequency_coherence_max_index = np.argmax(cxy)

    calmag = mag[frequency_ref_index]
    calph = ph[frequency_ref_index]
    calcoh = cxy[frequency_ref_index]
    maxcoh_f = f[frequency_coherence_max_index]
    maxcoh = cxy[frequency_coherence_max_index]

    calflab = '%s, %.2f Hz' % ('Ref frequency', frequency_ref_hz)
    calmaglab = 'Mag=%.2f' % calmag
    calphlab = 'Phase=%.2f' % calph
    calcohlab = 'Coherence=%.2f' % calcoh

    print(calflab)
    print(calmaglab)
    print(calphlab)
    print(calcohlab)
    print('Max coherence frequency, level:')
    print(maxcoh_f, maxcoh)

    rpd_plt.plot_psd_coh(psd_sig=psd_sig_bits, psd_ref=psd_ref_bits,
                         coherence_sig_ref=cxy,
                         f_hz=f,
                         f_min_hz=frequency_min_hz,
                         f_max_hz=frequency_max_hz,
                         f_scale='linear')

    rpd_plt.plot_psd_coh(psd_sig=cross_spectrum_bits, psd_ref=psd_ref_bits,
                         coherence_sig_ref=cxy,
                         f_hz=f,
                         f_min_hz=frequency_min_hz,
                         f_max_hz=frequency_max_hz,
                         f_scale='linear',
                         sig_label='Cross spectrum')

    rpd_plt.plot_response_scatter(h_magnitude=mag,
                                  h_phase_deg=ph,
                                  color_guide=cxy,
                                  f_hz=f,
                                  f_min_hz=frequency_min_hz,
                                  f_max_hz=frequency_max_hz,
                                  f_scale='linear')


def coherence_re_ref_pandas(df: pd.DataFrame,
                            ref_id: str,
                            sig_id_label: str,
                            sig_wf_label: str,
                            sig_sample_rate_label: str,
                            fs_fractional_tolerance: float = 0.02,
                            window_seconds: float = 2.,
                            window_overlap_fractional: float = 0.5,
                            frequency_ref_hz: float = 40.,
                            frequency_min_hz: float = 1,
                            frequency_max_hz: float = 320.,
                            sig_calib: float = 1.,
                            sig_ref_calib: float = 1.,
                            export_option: str = 'max_coherence',
                            plot_response: bool = False,
                            new_column_label_cohere_frequency: str = 'coherence_frequency',
                            new_column_label_cohere_value: str = 'coherence_value',
                            new_column_label_cohere_response_magnitude_bits: str = 'coherence_response_magnitude_bits',
                            new_column_label_cohere_response_phase_degrees: str = 'coherence_response_phase_degrees'
                            ) -> pd.DataFrame:
    """
    Find coherence between signals stored in dataframe, plot results

    :param df: input pandas DataFrame
    :param ref_id: name of reference signal in sig_id_label column in df
    :param sig_id_label: string for column name with station ids in df
    :param sig_wf_label: string for column name with waveform data in df
    :param sig_sample_rate_label: string for column name with sample rate in Hz information in df
    :param fs_fractional_tolerance: difference in sample rate (in Hz) tolerated. Default is 0.02
    :param window_seconds: seconds duration of window. Default is 2.0
    :param window_overlap_fractional: number of points to overlap between segments in window. Default is 0.5
    :param frequency_ref_hz: reference frequency in Hz. Default is 40.0
    :param frequency_min_hz: minimum frequency to plot in Hz (x min limit). Default is 1.0
    :param frequency_max_hz: maximum frequency to plot in Hz (x max limit). Default is 320.
    :param sig_calib: calibration of signal. Default is 1.0
    :param sig_ref_calib: sample rate of reference signal in Hz
    :param export_option: 'max_coherence' or 'ref_frequency'. Default is 'max_coherenece'
    :param plot_response: plot results. Default is False
    :param new_column_label_cohere_frequency: string for new column containing coherence frequency
    :param new_column_label_cohere_value: string for new column containing coherence values
    :param new_column_label_cohere_response_magnitude_bits: string for new column containing coherence response in bits
    :param new_column_label_cohere_response_phase_degrees: string for new column containing coherence phase in degrees
    :return: input pandas dataframe with new columns
    """
    number_sig = len(df.index)
    print("Coherence, number of signals excluding reference:", number_sig-1)
    print("Reference station: ", ref_id)

    # Is there a better way?
    m_list = df.index[df[sig_id_label] == ref_id]
    m = m_list[0]
    if len(m_list) > 1:
        raise Warning("More than one station meets the id spec. Picked first instance")

    # Initialize
    coherence_frequency = []
    coherence_value = []
    coherence_response_magnitude_bits = []
    coherence_response_phase_degrees = []

    if m is not None:
        print("Coherence Reference station ", df[sig_id_label][m])
        sig_m = np.copy(df[sig_wf_label][m]) * sig_ref_calib

        for n in df.index:
            sample_rate_condition = np.abs(df[sig_sample_rate_label][m] - df[sig_sample_rate_label][n]) \
                                    > fs_fractional_tolerance*df[sig_sample_rate_label][m]
            if sample_rate_condition:
                print("Sample rates out of tolerance")
                continue
            else:
                # Generalized sensor cross correlations, including unequal lengths
                sig_n = np.copy(df[sig_wf_label][n]) * sig_calib

            # Compute PSDs for each and coherence between the two
            window_points = int(window_seconds * df[sig_sample_rate_label][m])
            window_overlap_points = int(window_overlap_fractional*window_points)

            frequency_auto, auto_spectrum_sig = signal.welch(x=sig_n,
                                                             fs=df[sig_sample_rate_label][n],
                                                             nperseg=window_points,
                                                             noverlap=window_overlap_points)
            _, auto_spectrum_ref = signal.welch(x=sig_m,
                                                fs=df[sig_sample_rate_label][m],
                                                nperseg=window_points,
                                                noverlap=window_overlap_points)

            # Compute cross-power spectral density with ref sample rate
            frequency_cross, cross_spectrum = signal.csd(x=sig_n,
                                                         y=sig_m,
                                                         fs=df[sig_sample_rate_label][m],
                                                         nperseg=window_points,
                                                         noverlap=window_overlap_points)

            psd_ref_bits = 0.5 * utils.log2epsilon(abs(auto_spectrum_ref))
            psd_sig_bits = 0.5 * utils.log2epsilon(abs(auto_spectrum_sig))
            cross_spectrum_bits = 0.5 * utils.log2epsilon(abs(cross_spectrum))

            # Coherence, same as coherence from PSD
            frequency_coherence, coherence_welch = signal.coherence(x=sig_n,
                                                                    y=sig_m,
                                                                    fs=df[sig_sample_rate_label][m],
                                                                    nperseg=window_points,
                                                                    noverlap=window_overlap_points)

            # Compute response
            h_complex_response_sig = auto_spectrum_sig / cross_spectrum

            # Compute magnitude and phase in degrees
            magnitude_norm = np.abs(h_complex_response_sig)
            phase_degrees = np.unwrap(180 / np.pi * np.angle(h_complex_response_sig))

            # Assumes all the frequencies are the same - must verify
            frequency_ref_index = np.argmin(np.abs(frequency_coherence - frequency_ref_hz))
            frequency_coherence_max_index = np.argmax(coherence_welch)

            if 'max_coherence' == export_option:
                # Return max coherence values
                coherence_frequency.append(np.nan if n == m else frequency_coherence[frequency_coherence_max_index])
                coherence_value.append(np.max(coherence_welch))
                coherence_response_magnitude_bits.append(
                    0.5 * utils.log2epsilon(magnitude_norm[frequency_coherence_max_index]))
                coherence_response_phase_degrees.append(phase_degrees[frequency_coherence_max_index])
            if 'ref_frequency' == export_option:
                # New magnitude_norm and phase values at coherence frequency closest to ref frequency
                coherence_frequency.append(frequency_coherence[frequency_coherence_max_index])
                coherence_value.append(coherence_welch[frequency_ref_index])
                coherence_response_magnitude_bits.append(0.5 * utils.log2epsilon(magnitude_norm[frequency_ref_index]))
                coherence_response_phase_degrees.append(phase_degrees[frequency_ref_index])

            if plot_response:
                rpd_plt.plot_psd_coh(psd_sig=psd_sig_bits, psd_ref=psd_ref_bits,
                                     coherence_sig_ref=coherence_welch,
                                     f_hz=frequency_coherence,
                                     f_min_hz=frequency_min_hz,
                                     f_max_hz=frequency_max_hz,
                                     f_scale='linear')

                rpd_plt.plot_psd_coh(psd_sig=cross_spectrum_bits, psd_ref=psd_ref_bits,
                                     coherence_sig_ref=coherence_welch,
                                     f_hz=frequency_coherence,
                                     f_min_hz=frequency_min_hz,
                                     f_max_hz=frequency_max_hz,
                                     f_scale='linear',
                                     sig_label='Cross spectrum')

                rpd_plt.plot_response_scatter(h_magnitude=magnitude_norm,
                                              h_phase_deg=phase_degrees,
                                              color_guide=coherence_welch,
                                              f_hz=frequency_coherence,
                                              f_min_hz=frequency_min_hz,
                                              f_max_hz=frequency_max_hz,
                                              f_scale='linear')

        df[new_column_label_cohere_frequency] = coherence_frequency
        df[new_column_label_cohere_value] = coherence_value
        df[new_column_label_cohere_response_magnitude_bits] = coherence_response_magnitude_bits
        df[new_column_label_cohere_response_phase_degrees] = coherence_response_phase_degrees

    return df
