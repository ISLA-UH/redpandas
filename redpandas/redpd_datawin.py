"""
RedVox DataWindow related modules.
"""

# Python libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Tuple


# RedVox modules
from redvox.common.data_window import DataWindow, DataWindowConfig
from redvox.common.station import Station
from redvox.common.date_time_utils import MICROSECONDS_IN_SECOND
import redvox.common.date_time_utils as dt_utils

# RedPandas config
from redpandas.redpd_config import RedpdConfig


def dw_from_redpd_config(config: RedpdConfig) -> DataWindow:
    """
    Create RedVox DataWindow object from RedPandas configuration file with start/end times in epoch s

    :param config: RedpdConfig. REQUIRED
    :return: RedVox DataWindow object
    """
    dwa_config = DataWindowConfig(input_dir=config.input_dir,
                                  station_ids=config.station_ids,
                                  start_datetime=dt_utils.datetime_from_epoch_seconds_utc(config.event_start_epoch_s),
                                  end_datetime=dt_utils.datetime_from_epoch_seconds_utc(config.event_end_epoch_s),
                                  structured_layout=True,
                                  apply_correction=True,
                                  start_buffer_td=dt_utils.timedelta(minutes=config.start_buffer_minutes),
                                  end_buffer_td=dt_utils.timedelta(minutes=config.end_buffer_minutes))
    # Load RedVox Datawindow
    rdvx_data: DataWindow = DataWindow(event_name=config.event_name, config=dwa_config, debug=False)
    return rdvx_data


def plot_dw_mic(data_window: DataWindow, figsize: Tuple[int, int] = (10, 8)) -> Figure:
    """
    Plot audio data for all stations in RedVox DataWindow

    :param data_window: RedVox DataWindow object. REQUIRED
    :param figsize: Size of figure, defaults to (10, 8)
    :return: matplotlib figure instance
    """
    station: Station
    f1, ax1 = plt.subplots(figsize=figsize)  # Adjust to your screen
    for k, station in enumerate(data_window.stations()):
        if station.has_audio_data():
            mic_wf_raw = station.audio_sensor().get_data_channel("microphone")
            mic_epoch_s = station.audio_sensor().data_timestamps() / MICROSECONDS_IN_SECOND
            ax1.plot(mic_epoch_s - mic_epoch_s[0], mic_wf_raw / np.nanmax(np.abs(mic_wf_raw)), label=station.id())
            ax1.legend(loc='upper right')
            ax1.set_title("Audio raw normalized waveforms")
            ax1.set_xlabel("Time from record start, s")
    return f1


def plot_dw_baro(data_window: DataWindow, figsize: Tuple[int, int] = (10, 8)) -> Figure:
    """
    Plot barometer data for all stations in RedVox DataWindow

    :param data_window: RedVox DataWindow object. REQUIRED
    :param figsize: Size of figure, defaults to (10, 8)
    :return: matplotlib figure instance
    """
    station: Station
    f1, ax1 = plt.subplots(figsize=figsize)  # Adjust to your screen
    for k, station in enumerate(data_window.stations()):
        if station.has_barometer_data():
            baro_wf_raw = station.barometer_sensor().get_data_channel("pressure")
            baro_epoch_s = station.barometer_sensor().data_timestamps() / MICROSECONDS_IN_SECOND
            baro_wf = baro_wf_raw - np.nanmean(baro_wf_raw)
            ax1.plot(baro_epoch_s-baro_epoch_s[0], baro_wf / np.nanmax(np.abs(baro_wf)), label=station.id())
            ax1.legend(loc='upper right')
            ax1.set_title("Pressure raw normalized waveforms")
            ax1.set_xlabel("Time from record start, s")
    return f1
