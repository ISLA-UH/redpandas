import numpy as np
import pandas as pd
import scipy.io.wavfile
import scipy.signal as signal
from scipy.fft import rfft, fftfreq
from libquantum import synthetics
import matplotlib.pyplot as plt

"""These methods make assumptions about the RedVox sensor sample rates, which seldom exceed 500 Hz.
I also assume the highest audio sample rate we'll use is 8 kHz.
M. Garces, last updated 20210419"""

# Supported wav sample rates
permitted_wav_fs_values = 8000., 16000., 48000., 96000.
exception_str = "Wav sample rate must be 8000, 16000, or 48000 Hz"
lowest_wav_fs_value = 8000.
# TODO: Roll out lowest_fs
# TODO: Build tests in doc string


def stretch_factor_str(sig_sample_rate_hz: float,
                       wav_sample_rate_hz: float) -> str:
    """
    Compute file string for speedup and slowdown options
    :param sig_sample_rate_hz: input signal sample rate
    :param wav_sample_rate_hz: wav sample rate; supports permitted_wav_fs_values
    :return:
    """
    stretch_factor = wav_sample_rate_hz / sig_sample_rate_hz
    # If stretch factor is unity, no change
    stretch_str = '_preserve'
    if stretch_factor > 1:
        stretch_str = '_speedup_' + str(int(10.*stretch_factor)/10) + 'x_to'
    elif 1 > stretch_factor > 0:
        stretch_str = '_slowdown_' + str(int(10./stretch_factor)/10) + 'x_to'
    else:
        print("Stretch factor is zero or negative, address")
    return stretch_str


def resample_factor_str(sig_sample_rate_hz: float,
                        wav_sample_rate_hz: float) -> str:
    """
    Compute file string for oversample and downsample options
    :param sig_sample_rate_hz: input signal sample rate
    :param wav_sample_rate_hz: wav sample rate; supports permitted_wav_fs_values
    :return:
    """
    resample_factor = wav_sample_rate_hz / sig_sample_rate_hz
    # If resample factor is unity, no change
    resample_str = '_preserve'
    if resample_factor > 1:
        resample_str = '_upsample_' + str(int(10.*resample_factor)/10) + 'x_to'
    elif 1 > resample_factor > 0:
        resample_str = '_decimate_' + str(int(10./resample_factor)/10) + 'x_to'
    else:
        print("Resample factor is zero or negative, address")
    return resample_str


def sample_rate_str(wav_sample_rate_hz: float) -> str:
    """
    Generate the sample rate string for the exported sound file
    :param wav_sample_rate_hz: target wav sample rate
    :return: string with sample rate in kHz
    """
    wav_fs_str = '_' + str(int(wav_sample_rate_hz / 1000)) + 'khz.wav'
    return wav_fs_str


def resample_fourier(sig_wf: np.ndarray,
                     sig_sample_rate_hz: float,
                     new_sample_rate_hz: float = 8000.) -> np.ndarray:
    """
    Resample the Fourier way; can upsample or downsample. Downsample will be aliased, so use decimate in that case.
    :param sig_wf: input signal waveform, reasonably well preprocessed
    :param sig_sample_rate_hz: signal sample rate
    :param new_sample_rate_hz: resampling sample rate
    :return: resampled signal
    """
    sig_len = len(sig_wf)
    new_len = int(sig_len * new_sample_rate_hz / sig_sample_rate_hz)
    sig_resampled = signal.resample(x=sig_wf, num=new_len)
    return sig_resampled


def decimate_to_aud(sig_wf: np.ndarray,
                    sig_sample_rate_hz: float,
                    new_sample_rate_hz: float = 8000.) -> np.ndarray:
    """
    Decimate with AA, min of 8 kHz. Assumed preprocessed for gaps, DC offset, slope, etc.
    :param sig_wf: input signal waveform, reasonably well preprocessed
    :param sig_sample_rate_hz: signal sample rate
    :param new_sample_rate_hz: target wav sample rate
    :return: decimated signal
    """

    decimation_factor = int(np.round(sig_sample_rate_hz/new_sample_rate_hz))

    if decimation_factor >= 2:
        sig_resampled = signal.decimate(x=sig_wf, q=decimation_factor, zero_phase=True)
        return sig_resampled
    else:
        print("Should not have gotten this far, check code")
        exit()


def save_to_elastic_wav(sig_wf: np.ndarray,
                        sig_sample_rate_hz: float,
                        wav_filename: str,
                        wav_sample_rate_hz: float = 8000.):
    """
    :param sig_wf: input signal waveform, reasonably well preprocessed
    :param sig_sample_rate_hz: input signal sample rate
    :param wav_filename: wav file name, with directory path
    :param wav_sample_rate_hz: wav sample rate; supports permitted_wav_fs_values
    :return: Export to wav file
    """

    if int(wav_sample_rate_hz) in permitted_wav_fs_values:
        stretch_str = stretch_factor_str(sig_sample_rate_hz=sig_sample_rate_hz,
                                         wav_sample_rate_hz=wav_sample_rate_hz)
        khz_str = sample_rate_str(wav_sample_rate_hz=wav_sample_rate_hz)
        export_filename = wav_filename + stretch_str + khz_str
        synth_wav = 0.9 * np.real(sig_wf) / np.max(np.abs((np.real(sig_wf))))
        scipy.io.wavfile.write(export_filename, int(wav_sample_rate_hz), synth_wav)
    else:
        print(exception_str)


def save_to_resampled_wav(sig_wf: np.ndarray,
                          sig_sample_rate_hz: float,
                          wav_filename: str,
                          wav_sample_rate_hz: float = 8000.):
    """
    :param sig_wf: input signal waveform, reasonably well preprocessed
    :param sig_sample_rate_hz: input signal sample rate
    :param wav_filename: wav file name, with directory path
    :param wav_sample_rate_hz: wav sample rate; only 8kHz, 16kHz, and 48kHz
    :return: Export to wav file
    """

    # Export to wav directory
    if int(wav_sample_rate_hz) in permitted_wav_fs_values:
        resample_str = resample_factor_str(sig_sample_rate_hz=sig_sample_rate_hz,
                                           wav_sample_rate_hz=wav_sample_rate_hz)
        khz_str = sample_rate_str(wav_sample_rate_hz=wav_sample_rate_hz)
        export_filename = wav_filename + resample_str + khz_str
        synth_wav = 0.9 * np.real(sig_wf) / np.max(np.abs((np.real(sig_wf))))
        scipy.io.wavfile.write(export_filename, int(wav_sample_rate_hz), synth_wav)
    else:
        print(exception_str)


def pandas_to_resampled_wav(df: pd.DataFrame,
                            sig_wf_label: str,
                            sig_sample_rate_hz_label: str,
                            sig_id_label: str = "index",
                            wav_dir_prefix: str = "aud_",
                            wav_sample_rate_hz: float = 8000.,
                            sample_rate_tolerance_percent: float = 1.):
    """
    Ensonify a pandas data frame
    :param df: data frame
    :param sig_wf_label: label of signal to be ensonified
    :param sig_sample_rate_hz_label: label of sample rate
    :param sig_id_label: label to be used to id the signal
    :param wav_dir_prefix: export directory, and prefix for file
    :param wav_sample_rate_hz: nominal wav sample rate, default of 8 kHz
    :param sample_rate_tolerance_percent: percent of permitted difference in sig and wav sample rates
    :return: export to .wav
    """

    for n in df.index:
        if sig_id_label == "index":
            sig_id_str = str(df.index[n])
        else:
            sig_id_str = df[sig_id_label][n]

        wav_pd_filename = wav_dir_prefix + sig_id_str
        sig_sample_rate_hz = df[sig_sample_rate_hz_label][n]

        # Criteria to decimate to downsample or resample to upsample
        decimation_factor = int(np.round(sig_sample_rate_hz/wav_sample_rate_hz))

        # Some variability is expected; don't resample if difference is less than tolerance
        threshold = sample_rate_tolerance_percent/100.*sig_sample_rate_hz
        if np.abs(sig_sample_rate_hz - wav_sample_rate_hz) > threshold:
            if decimation_factor >= 2:
                # Decimate
                sig_resampled = \
                    decimate_to_aud(sig_wf=df[sig_wf_label][n],
                                    sig_sample_rate_hz=df[sig_sample_rate_hz_label][n],
                                    new_sample_rate_hz=wav_sample_rate_hz)
            else:
                # Resample
                sig_resampled = \
                    resample_fourier(sig_wf=df[sig_wf_label][n],
                                     sig_sample_rate_hz=df[sig_sample_rate_hz_label][n],
                                     new_sample_rate_hz=wav_sample_rate_hz)

            save_to_resampled_wav(sig_wf=sig_resampled,
                                  sig_sample_rate_hz=df[sig_sample_rate_hz_label][n],
                                  wav_filename=wav_pd_filename,
                                  wav_sample_rate_hz=wav_sample_rate_hz)
        else:
            # Save unchanged waveform
            save_to_resampled_wav(sig_wf=df[sig_wf_label][n],
                                  sig_sample_rate_hz=df[sig_sample_rate_hz_label][n],
                                  wav_filename=wav_pd_filename,
                                  wav_sample_rate_hz=wav_sample_rate_hz)


def pandas_to_elastic_wav(df: pd.DataFrame,
                          sig_wf_label: str,
                          sig_sample_rate_hz_label: str,
                          sig_id_label: str = "index",
                          wav_dir_prefix: str = "aud_",
                          wav_sample_rate_hz: float = 8000.):
    """
    Ensonify a pandas data frame
    :param df: data frame
    :param sig_wf_label: label of signal to be ensonified
    :param sig_sample_rate_hz_label: label of sample rate
    :param sig_id_label: label to be used to id the signal
    :param wav_dir_prefix: export directory, and prefix for file
    :param wav_sample_rate_hz: nominal wav sample rate, default of 8 kHz
    :param sample_rate_tolerance_percent: percent of permitted difference in sig and wav sample rates
    :return: export to .wav
    """

    for n in df.index:
        if sig_id_label == "index":
            sig_id_str = str(df.index[n])
        else:
            sig_id_str = df[sig_id_label][n]

        wav_pd_filename = wav_dir_prefix + sig_id_str
        save_to_elastic_wav(sig_wf=df[sig_wf_label][n],
                            sig_sample_rate_hz=df[sig_sample_rate_hz_label][n],
                            wav_filename=wav_pd_filename,
                            wav_sample_rate_hz=wav_sample_rate_hz)


def dual_tone_test():
    dir_filename = "./test"
    # Test tone
    sample_rate = 48000.
    new_rate = 8000.
    duration_s = 1.
    center_frequency = np.min([sample_rate, new_rate]) / 8.
    t = np.arange(0, duration_s, 1 / sample_rate)
    peak_amp = np.sqrt(2)
    y = peak_amp * np.sin(2 * np.pi * center_frequency * t) + \
        peak_amp * np.sin(2 * np.pi * new_rate/2. * t)
    z = synthetics.antialias_halfNyquist(y)
    lz = len(z)
    print('Original Number of Points: ', lz)
    fz = 2 * rfft(z) / lz
    fz_f = fftfreq(lz, 1 / sample_rate)

    z_rs = resample_fourier(sig_wf=z, sig_sample_rate_hz=sample_rate, new_sample_rate_hz=new_rate)
    lz_rs = len(z_rs)
    print('Resampled Number of Points: ', lz_rs)
    t_rs = np.arange(lz_rs) / new_rate
    fz_rs = 2 * rfft(z_rs) / lz_rs
    fz_rs_f = fftfreq(lz_rs, 1 / new_rate)

    plt.figure()
    plt.subplot(211), plt.plot(t, z)
    plt.title('Unit rms test tone, fc = ' + str(int(center_frequency)) + ' Hz')
    plt.subplot(212), plt.loglog(fz_f[1:lz // 2], np.abs(fz[1:lz // 2]))

    plt.figure()
    plt.subplot(211), plt.plot(t_rs, z_rs)
    plt.title('Resampled test tone, fc = ' + str(int(center_frequency)) + ' Hz + Nyquist at new rate')
    plt.subplot(212), plt.loglog(fz_rs_f[1:lz_rs // 2], np.abs(fz_rs[1:lz_rs // 2]))

    save_to_resampled_wav(sig_wf=z_rs,
                          sig_sample_rate_hz=sample_rate,
                          wav_filename=dir_filename,
                          wav_sample_rate_hz=new_rate)
    save_to_elastic_wav(sig_wf=z,
                        sig_sample_rate_hz=sample_rate,
                        wav_filename=dir_filename,
                        wav_sample_rate_hz=new_rate)

    plt.show()


if __name__ == "__main__":
    dual_tone_test()
