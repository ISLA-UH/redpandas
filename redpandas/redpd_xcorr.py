import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt


def find_nearest(array, value):
    # https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    xi = np.argmin(np.abs(np.ceil(array[None].T - value)), axis=0)
    return xi


def plot_square(xnorm_max, xoffset_s, xoffset_points, sig_descriptor: str = "Signal"):
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


def most_similar_station_index(xnorm_max):
    """
    Sums over column, subtract self xcorr (1), divides by number of stations - 1
    :param xnorm_max: normalized cross correlation
    :return: index of most self-similar station to the ensemble
    """
    xnorm_max_sum = np.sum(xnorm_max, axis=1)
    xnorm_stats = (xnorm_max_sum - 1)/(len(xnorm_max_sum) - 1)

    xcorr_ref_index = int(np.argmax(xnorm_stats))
    xcorr_mean_max = xnorm_stats[xcorr_ref_index]

    return xcorr_ref_index, xcorr_mean_max


# TODO: Repeat with spectrogram, return time, frequency, and max cross-correlation
# Sort out time first: time gate input, refer to shared datum, correct times
def xcorr_pandas(df: pd.DataFrame,
                 sig_label: str,
                 sig_sample_rate_label: str,
                 fs_fractional_tolerance: float = 0.02,
                 abs_xcorr: bool = True):
    """
    Returns square matrix, a concise snapshot of the self-similarity of the input data set.
    :param df:
    :param sig_label:
    :param sig_sample_rate_label:
    :param fs_fractional_tolerance:
    :param abs_xcorr:
    :return:
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
                print("Sample rates out of tolerance for index m,n =" + str(m) + "," + str(n))
                continue
            else:
                sig_n = np.copy(df[sig_label][n])
                sig_m = np.copy(df[sig_label][m])
                # Generalized sensor cross correlations, including unequal lengths
                n_points = len(sig_n)
                m_points = len(sig_m)
                # Faster as floats

                if n_points > m_points:
                    """Cross Correlation 'full' sums over the dimension of sig_n"""
                    xcorr_indexes = np.arange(1-n_points, m_points)
                    # Ensure it is a float
                    xcorr = signal.correlate(sig_m, sig_n, mode='full')
                    # Normalize
                    xcorr /= np.sqrt(n_points*m_points) * sig_n.std() * sig_m.std()
                    if abs_xcorr:
                        # Allows negative peak in cross correlation (pi phase shift)
                        xcorr_offset_index = np.argmax(np.abs(xcorr))
                    else:
                        # Must be in phase -  for array processing
                        xcorr_offset_index = np.argmax(xcorr)
                    xcorr_offset_samples = xcorr_indexes[xcorr_offset_index]
                elif n_points < m_points:
                    """Cross Correlation 'full' sums over the dimension of sig_m"""
                    xcorr_indexes = np.arange(1-m_points, n_points)
                    xcorr = signal.correlate(sig_n, sig_m, mode='full')
                    # Normalize
                    xcorr /= np.sqrt(n_points*m_points) * sig_n.std() * sig_m.std()
                    if abs_xcorr:
                        # Allows negative peak in cross correlation (pi phase shift)
                        xcorr_offset_index = np.argmax(np.abs(xcorr))
                    else:
                        # Must be in phase -  for array processing
                        xcorr_offset_index = np.argmax(xcorr)
                    # Flip sign
                    xcorr_offset_samples = -xcorr_indexes[xcorr_offset_index]
                elif n_points == m_points:
                    """Cross correlation is centered in the middle of the record and has length n_points"""
                    # Fastest, o(NX) and can use FFT solution
                    if n_points % 2 == 0:
                        xcorr_indexes = np.arange(-int(n_points/2), int(n_points/2))
                    else:
                        xcorr_indexes = np.arange(-int(n_points/2), int(n_points/2)+1)
                    xcorr = signal.correlate(sig_m, sig_n, mode='same')
                    # Normalize
                    xcorr /= np.sqrt(n_points*m_points) * sig_n.std() * sig_m.std()
                    if abs_xcorr:
                        # Allows negative peak in cross correlation (pi phase shift)
                        xcorr_offset_index = np.argmax(np.abs(xcorr))
                    else:
                        # Must be in phase -  for array processing
                        xcorr_offset_index = np.argmax(xcorr)
                    xcorr_offset_samples = xcorr_indexes[xcorr_offset_index]
                else:
                    print('One of the waveforms is broken')
                    continue

                xcorr_normalized_max[m, n] = xcorr[xcorr_offset_index]
                xcorr_offset_seconds[m, n] = xcorr_offset_samples/df[sig_sample_rate_label][n]
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
    Returns new pandas columns per station with xcorr results relative to a reference station
    :param df:
    :param ref_id_label:
    :param sig_id_label:
    :param sig_wf_label:
    :param sig_sample_rate_label:
    :param fs_fractional_tolerance:
    :param abs_xcorr:
    :param return_xcorr_full:
    :param new_column_label_xcorr_offset_points:
    :param new_column_label_xcorr_offset_seconds:
    :param new_column_label_xcorr_normalized_max:
    :param new_column_label_xcorr_full_array:
    :return:
    """

    number_sig = len(df.index)
    print("XCORR Nmber of signals:", number_sig)
    # exit()

    m_list = df.index[df[sig_id_label] == ref_id_label]
    m = m_list[0]
    # print("m", m)
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
                    xcorr_indexes = np.arange(1-n_points, m_points)
                    # Ensure it is a float
                    xcorr = signal.correlate(sig_m, sig_n, mode='full')
                    # Normalize
                    xcorr /= np.sqrt(n_points*m_points) * sig_n.std() * sig_m.std()
                    if abs_xcorr:
                        # Allows negative peak in cross correlation (pi phase shift)
                        xcorr_offset_index = np.argmax(np.abs(xcorr))
                    else:
                        # Must be in phase -  for array processing
                        xcorr_offset_index = np.argmax(xcorr)
                    xcorr_offset_samples = xcorr_indexes[xcorr_offset_index]
                elif n_points < m_points:
                    """Cross Correlation 'full' sums over the dimension of sig_m"""
                    xcorr_indexes = np.arange(1-m_points, n_points)
                    xcorr = signal.correlate(sig_n, sig_m, mode='full')
                    # Normalize
                    xcorr /= np.sqrt(n_points*m_points) * sig_n.std() * sig_m.std()
                    if abs_xcorr:
                        # Allows negative peak in cross correlation (pi phase shift)
                        xcorr_offset_index = np.argmax(np.abs(xcorr))
                    else:
                        # Must be in phase -  for array processing
                        xcorr_offset_index = np.argmax(xcorr)
                    # Flip sign
                    xcorr_offset_samples = -xcorr_indexes[xcorr_offset_index]
                elif n_points == m_points:
                    """Cross correlation is centered in the middle of the record and has length n_points"""
                    # Fastest, o(NX) and can use FFT solution
                    if n_points % 2 == 0:
                        xcorr_indexes = np.arange(-int(n_points/2), int(n_points/2))
                    else:
                        xcorr_indexes = np.arange(-int(n_points/2), int(n_points/2)+1)
                    xcorr = signal.correlate(sig_m, sig_n, mode='same')
                    # Normalize
                    xcorr /= np.sqrt(n_points*m_points) * sig_n.std() * sig_m.std()
                    if abs_xcorr:
                        # Allows negative peak in cross correlation (pi phase shift)
                        xcorr_offset_index = np.argmax(np.abs(xcorr))
                    else:
                        # Must be in phase -  for array processing
                        xcorr_offset_index = np.argmax(xcorr)
                    xcorr_offset_samples = xcorr_indexes[xcorr_offset_index]
                else:
                    print('One of the waveforms is broken')
                    continue

                # Main export parameters
                # Allows negative peak in cross correlation (pi phase shift) in raw waveform, unless the input is power
                xcorr_normalized_max.append(xcorr[xcorr_offset_index])
                xcorr_offset_points.append(xcorr_offset_samples)
                xcorr_offset_seconds.append(xcorr_offset_samples/df[sig_sample_rate_label][n])
                if return_xcorr_full:
                    xcorr_full.append(xcorr)

        # Convert to columns and add it to df
        df[new_column_label_xcorr_normalized_max] = xcorr_normalized_max
        df[new_column_label_xcorr_offset_points] = xcorr_offset_points
        df[new_column_label_xcorr_offset_seconds] = xcorr_offset_seconds
        if return_xcorr_full:
            df[new_column_label_xcorr_full_array] = xcorr_full

    else:
        print('ERROR: Incorrect reference station id')
        exit()

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

    # TODO: Refer to quantum/examples/blast_spectcorr_check, test with redshift
    # Have to learn how to use/validate correlate2D
    number_sig = len(df.index)
    print("SPECTCORR number of signals:", number_sig)
    # exit()

    # M is the reference station
    m_list = df.index[df[sig_id_label] == ref_id_label]
    m = m_list[0]
    # print("m", m)
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
        if ref_columns % 2 == 0:
            xcorr_index = np.arange(-int(ref_columns/2), int(ref_columns/2))
        else:
            xcorr_index = np.arange(-int(ref_columns/2), int(ref_columns/2)+1)
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
        print('ERROR: Incorrect reference station id')
        exit()

    return df
