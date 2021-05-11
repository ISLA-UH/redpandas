import numpy as np
import pandas as pd
from scipy import signal
from libquantum import utils
import redpandas.redpd_plot as rpd_plt


def coherence_numpy(sig_in,
                    sig_in_ref,
                    sig_sample_rate_hz,
                    sig_ref_sample_rate_hz,
                    window_seconds: float = 2.,
                    window_overlap_fractional: float = 0.5,
                    frequency_ref_hz: float = 40.,
                    frequency_min_hz: float = 1,
                    frequency_max_hz: float = 320.,
                    sig_calib: float = 1.,
                    sig_ref_calib:float = 1.):

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

    f, Pxy = signal.csd(x=sig,
                        y=sig_ref,
                        fs=sig_ref_sample_rate_hz,
                        nperseg=window_points,
                        noverlap=window_overlap_points)
    cross_spectrum_bits = 0.5 * np.log2(abs(Pxy))

    # Coherence
    # Original code had no overlap - fixed
    # Same as coherence from PSD
    # Cxy_psd = np.abs(Pxy)**2/(pxx_ref*pxx_sig)
    # TODO: Compute coherence from spectral products with other representations
    f, Cxy = signal.coherence(x=sig,
                              y=sig_ref,
                              fs=sig_ref_sample_rate_hz,
                              nperseg=window_points,
                              noverlap=window_overlap_points)

    # Compute response assuming incoherent comp in ref.
    # Ref sensor response already removed
    H_x = pxx_sig / Pxy

    # compute magnitude and phase in deg
    mag = np.abs(H_x)
    ph = np.unwrap(180 / np.pi * np.angle(H_x))

    # get new mag and phase values at frequency closest to ref frequency
    frequency_ref_index = np.argmin(np.abs(f - frequency_ref_hz))
    frequency_coherence_max_index = np.argmax(Cxy)

    calmag = mag[frequency_ref_index]
    calph = ph[frequency_ref_index]
    calcoh = Cxy[frequency_ref_index]
    maxcoh_f = f[frequency_coherence_max_index]
    maxcoh = Cxy[frequency_coherence_max_index]

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
                         coherence_sig_ref=Cxy,
                         f_hz=f,
                         f_min_hz=frequency_min_hz,
                         f_max_hz=frequency_max_hz,
                         f_scale='linear')

    rpd_plt.plot_psd_coh(psd_sig=cross_spectrum_bits, psd_ref=psd_ref_bits,
                         coherence_sig_ref=Cxy,
                         f_hz=f,
                         f_min_hz=frequency_min_hz,
                         f_max_hz=frequency_max_hz,
                         f_scale='linear',
                         sig_label='Cross spectrum')

    rpd_plt.plot_response_scatter(h_magnitude=mag,
                                  h_phase_deg=ph,
                                  color_guide=Cxy,
                                  f_hz=f,
                                  f_min_hz=frequency_min_hz,
                                  f_max_hz=frequency_max_hz,
                                  f_scale='linear')


def coherence_re_ref_pandas(df: pd.DataFrame,
                            ref_id: str,
                            sig_id_name: str,
                            sig_name: str,
                            sig_sample_rate_name: str,
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
                            new_column_label_cohere_response_phase_degrees: str = 'coherence_response_phase_degrees'):

    number_sig = len(df.index)
    print("Coherence, number of signals excluding reference:", number_sig-1)
    print("Reference station: ", ref_id)
    # exit()

    # Is there a better way?
    m_list = df.index[df[sig_id_name] == ref_id]
    m = m_list[0]
    # print("m", m)
    if len(m_list) > 1:
        raise Warning("More than one station meets the id spec. Picked first instance")

    # Initialize
    coherence_frequency = []
    coherence_value = []
    coherence_response_magnitude_bits = []
    coherence_response_phase_degrees = []

    if m is not None:
        print("Coherence Reference station ", df[sig_id_name][m])
        sig_m = np.copy(df[sig_name][m]) * sig_ref_calib
        m_points = len(sig_m)

        for n in df.index:
            # if n == m:
            #     # Skip itself
            #     continue

            sample_rate_condition = np.abs(df[sig_sample_rate_name][m] - df[sig_sample_rate_name][n]) \
                                    > fs_fractional_tolerance*df[sig_sample_rate_name][m]
            if sample_rate_condition:
                print("Sample rates out of tolerance")
                return
            else:
                # Generalized sensor cross correlations, including unequal lengths
                sig_n = np.copy(df[sig_name][n]) * sig_calib
                n_points = len(sig_n)

            # TODO: Can incorrectly feed in different length windows.
            #  Inherit length/sample rate checks from xcorr and spectcorr.
            # Compute PSDs for each and coherence between the two
            window_points = int(window_seconds*df[sig_sample_rate_name][m])
            window_overlap_points = int(window_overlap_fractional*window_points)

            frequency_auto, auto_spectrum_sig = signal.welch(x=sig_n,
                                                             fs=df[sig_sample_rate_name][n],
                                                             nperseg=window_points,
                                                             noverlap=window_overlap_points)
            _, auto_spectrum_ref = signal.welch(x=sig_m,
                                                fs=df[sig_sample_rate_name][m],
                                                nperseg=window_points,
                                                noverlap=window_overlap_points)


            # Compute cross-power spectral density with ref sample rate
            # Original code had no overlap - fixed
            frequency_cross, cross_spectrum = signal.csd(x=sig_n,
                                                         y=sig_m,
                                                         fs=df[sig_sample_rate_name][m],
                                                         nperseg=window_points,
                                                         noverlap=window_overlap_points)

            psd_ref_bits = 0.5 * utils.log2epsilon(abs(auto_spectrum_ref))
            psd_sig_bits = 0.5 * utils.log2epsilon(abs(auto_spectrum_sig))
            cross_spectrum_bits = 0.5 * utils.log2epsilon(abs(cross_spectrum))

            # Coherence, same as coherence from PSD
            # Cxy_psd = np.abs(cross_spectrum)**2/(pxx_ref*auto_spectrum_sig)
            frequency_coherence, coherence_welch = signal.coherence(x=sig_n,
                                                                    y=sig_m,
                                                                    fs=df[sig_sample_rate_name][m],
                                                                    nperseg=window_points,
                                                                    noverlap=window_overlap_points)

            # Compute response
            h_complex_response_sig = auto_spectrum_sig / cross_spectrum

            # Compute magnitude and phase in degrees
            magnitude_norm = np.abs(h_complex_response_sig)
            phase_degrees = np.unwrap(180 / np.pi * np.angle(h_complex_response_sig))

            # Assumes all the frequencies are the same - must verify
            # TODO: use function for closest
            frequency_ref_index = np.argmin(np.abs(frequency_coherence - frequency_ref_hz))
            frequency_coherence_max_index = np.argmax(coherence_welch)

            # New magnitude_norm and phase values at coherence frequency closest to ref frequency
            ref_frequency_hz = frequency_coherence[frequency_coherence_max_index]
            ref_frequency_coherence = coherence_welch[frequency_ref_index]
            ref_frequency_response_magnitude_bits = 0.5*utils.log2epsilon(magnitude_norm[frequency_ref_index])
            ref_frequency_response_phase_degrees = phase_degrees[frequency_ref_index]

            # Return max coherence values
            max_coherence_frequency_hz = frequency_coherence[frequency_coherence_max_index]
            max_coherence = np.max(coherence_welch)
            max_coherence_response_magnitude_bits = 0.5*utils.log2epsilon(magnitude_norm[frequency_coherence_max_index])
            max_coherence_response_phase_degrees = phase_degrees[frequency_coherence_max_index]

            if n == m:
                max_coherence_frequency_hz = np.nan

            if 'max_coherence' == export_option:
                coherence_frequency.append(max_coherence_frequency_hz)
                coherence_value.append(max_coherence)
                coherence_response_magnitude_bits.append(max_coherence_response_magnitude_bits)
                coherence_response_phase_degrees.append(max_coherence_response_phase_degrees)
            if 'ref_frequency' == export_option:
                coherence_frequency.append(ref_frequency_hz)
                coherence_value.append(ref_frequency_coherence)
                coherence_response_magnitude_bits.append(ref_frequency_response_magnitude_bits)
                coherence_response_phase_degrees.append(ref_frequency_response_phase_degrees)

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
                # Must plt.show() on main

        df[new_column_label_cohere_frequency] = coherence_frequency
        df[new_column_label_cohere_value] = coherence_value
        df[new_column_label_cohere_response_magnitude_bits] = coherence_response_magnitude_bits
        df[new_column_label_cohere_response_phase_degrees] = coherence_response_phase_degrees

    return df
