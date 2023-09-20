"""
Module to facilitate reports
"""
from typing import Tuple

import pandas as pd
from matplotlib.figure import Figure

import redpandas.redpd_tfr as rpd_tfr
from redpandas import redpd_scales as rpd_scales
from redpandas.redpd_plot.mesh import plot_mesh_pandas
from redpandas.redpd_plot.wiggles import plot_wiggles_pandas, plot_wiggles_3c_pandas


def plot_wiggles_pandas_audio(df: pd.DataFrame,
                              start_time_window: float = 0.0,
                              end_time_window: float = 0.0) -> Figure:
    """
    Plot wiggles, set for Audio

    :param df: input pandas dataframe
    :param start_time_window: float, epoch s
    :param end_time_window: float, epoch s
    :return: matplotlib figure instance
    """
    return plot_wiggles_pandas(df=df,
                               sig_wf_label="audio_wf",
                               sig_timestamps_label="audio_epoch_s",
                               sig_id_label="station_id",
                               show_figure=False,
                               fig_title="Normalized Signals for Audio",
                               start_time_window=start_time_window,
                               end_time_window=end_time_window)


def plot_wiggles_pandas_bar(df: pd.DataFrame,
                            start_time_window: float = 0.0,
                            end_time_window: float = 0.0) -> Figure:
    """
    Plot wiggles, set for Barometer

    :param df: input pandas dataframe
    :param start_time_window: float, epoch s
    :param end_time_window: float, epoch s
    :return: matplotlib figure instance
    """
    return plot_wiggles_pandas(df=df,
                               sig_wf_label="barometer_wf_highpass",
                               sig_timestamps_label="barometer_epoch_s",
                               sig_id_label="station_id",
                               show_figure=False,
                               fig_title="Normalized Signals for Barometer",
                               start_time_window=start_time_window,
                               end_time_window=end_time_window)


def plot_wiggles_pandas_acc(df: pd.DataFrame,
                            start_time_window: float = 0.0,
                            end_time_window: float = 0.0) -> Tuple[Figure, Figure, Figure]:
    """
    Plot wiggles, set for Accelerometer

    :param df: input pandas dataframe
    :param start_time_window: float, epoch s
    :param end_time_window: float, epoch s
    :return: matplotlib figure instance x3
    """
    figs = plot_wiggles_3c_pandas(df=df,
                                  sig_wf_label='accelerometer_wf_highpass',
                                  sig_timestamps_label='accelerometer_epoch_s',
                                  fig_title="Normalized Signals for Accelerometer",
                                  show_figure=False,
                                  start_time_window=start_time_window,
                                  end_time_window=end_time_window)
    return figs[0], figs[1], figs[2]


def plot_wiggles_pandas_gyr(df: pd.DataFrame,
                            start_time_window: float = 0.0,
                            end_time_window: float = 0.0) -> Tuple[Figure, Figure, Figure]:
    """
    Plot wiggles, set for Gyroscope

    :param df: input pandas dataframe
    :param start_time_window: float, epoch s
    :param end_time_window: float, epoch s
    :return: matplotlib figure instance x3
    """
    figs = plot_wiggles_3c_pandas(df=df,
                                  sig_wf_label='gyroscope_wf_highpass',
                                  sig_timestamps_label='gyroscope_epoch_s',
                                  fig_title="Normalized Signals for Gyroscope",
                                  show_figure=False,
                                  start_time_window=start_time_window,
                                  end_time_window=end_time_window)
    return figs[0], figs[1], figs[2]


def plot_wiggles_pandas_mag(df: pd.DataFrame,
                            start_time_window: float = 0.0,
                            end_time_window: float = 0.0) -> Tuple[Figure, Figure, Figure]:
    """
    Plot wiggles, set for Magnetometer

    :param df: input pandas dataframe
    :param start_time_window: float, epoch s
    :param end_time_window: float, epoch s
    :return: matplotlib figure instance x3
    """
    figs = plot_wiggles_3c_pandas(df=df,
                                  sig_wf_label='magnetometer_wf_highpass',
                                  sig_timestamps_label='magnetometer_epoch_s',
                                  fig_title="Normalized Signals for Magnetometer",
                                  show_figure=False,
                                  start_time_window=start_time_window,
                                  end_time_window=end_time_window)

    return figs[0], figs[1], figs[2]


def tfr_bits_panda_audio(df: pd.DataFrame,
                         start_time_window: float = 0.0,
                         end_time_window: float = 0.0,
                         tfr_type: str = 'stft',
                         order_number_input: int = 12):
    """
    Calculate TFR, set for Audio

    :param df: input pandas dataframe
    :param start_time_window: float, epoch s
    :param end_time_window: float, epoch s
    :param tfr_type: 'stft' or 'cwt'. Default is 'stft'
    :param order_number_input: default is 12
    :return: matplotlib figure of audio
    """
    # Audio TFR
    return rpd_tfr.tfr_bits_panda_window(df=df,
                                         sig_wf_label="audio_wf",
                                         sig_sample_rate_label="audio_sample_rate_nominal_hz",
                                         sig_timestamps_label="audio_epoch_s",
                                         order_number_input=order_number_input,
                                         tfr_type=tfr_type,
                                         new_column_tfr_bits="audio_tfr_bits",
                                         new_column_tfr_frequency_hz="audio_tfr_frequency_hz",
                                         new_column_tfr_time_s="audio_tfr_time_s",
                                         start_time_window=start_time_window,
                                         end_time_window=end_time_window)


def plot_mesh_pandas_audio(df: pd.DataFrame,
                           frequency_hz_ymin: float = 1.0,
                           frequency_hz_ymax: float = 0.0,
                           frequency_scaling: str = "log",
                           start_time_window: float = 0.0,):
    """
    Plot mesh, set for Audio

    :param df: input pandas dataframe
    :param frequency_hz_ymin: float, default is 1.0
    :param frequency_hz_ymax: float, sets to Nyquist if 0.0
    :param frequency_scaling: 'log' or 'lin', default is 'log'
    :param start_time_window: float, epoch s
    :return: matplotlib figure instance
    """
    if frequency_hz_ymax == 0.0:
        frequency_hz_ymax = rpd_scales.Slice.F0
    if frequency_hz_ymin == 0.0:
        frequency_hz_ymin = rpd_scales.Slice.FU
    if frequency_hz_ymin >= frequency_hz_ymax and frequency_hz_ymin != 0.0:
        frequency_hz_ymin = 1.0
        frequency_hz_ymax = rpd_scales.Slice.F0
    return plot_mesh_pandas(df=df,
                            mesh_time_label="audio_tfr_time_s",
                            mesh_frequency_label="audio_tfr_frequency_hz",
                            mesh_tfr_label="audio_tfr_bits",
                            sig_id_label='station_id',
                            t0_sig_epoch_s=df['audio_epoch_s'][0][0] if start_time_window == 0.0 else start_time_window,
                            fig_title="STFT Audio",
                            frequency_scaling=frequency_scaling,
                            frequency_hz_ymin=frequency_hz_ymin,
                            frequency_hz_ymax=frequency_hz_ymax,
                            mesh_color_scaling="range",
                            mesh_color_range=16.0,
                            show_figure=False)
