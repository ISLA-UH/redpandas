"""
RedVox DataWindow related modules.
"""

# Python libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List, Optional
from datetime import datetime
import os

# RedVox modules
from redvox.common.data_window import DataWindow
from redvox.common.station import Station
from redvox.common.date_time_utils import MICROSECONDS_IN_SECOND
import redvox.common.data_window_configuration as dwc
import redvox.common.date_time_utils as dt_utils

# RedPandas config
from redpandas.redpd_config import RedpdConfig


def dw_from_redpd_config(config: RedpdConfig) -> DataWindow:
    """
    Create RedVox DataWindow object from RedPandas configuration file with start/end times in epoch s

    :param config: RedpdConfig. REQUIRED

    :return: RedVox DataWindow object
    """

    api_input_directory: str = config.input_dir
    redvox_station_ids: List[str] = config.station_ids
    start_epoch_s: float = config.event_start_epoch_s
    end_epoch_s: float = config.event_end_epoch_s
    start_buffer_minutes: int = config.start_buffer_minutes
    end_buffer_minutes: int = config.end_buffer_minutes

    # Load RedVox Datawindow
    rdvx_data: DataWindow = DataWindow(input_dir=api_input_directory,
                                       structured_layout=True,
                                       start_datetime=dt_utils.datetime_from_epoch_seconds_utc(start_epoch_s),
                                       end_datetime=dt_utils.datetime_from_epoch_seconds_utc(end_epoch_s),
                                       station_ids=redvox_station_ids,
                                       start_buffer_td=dt_utils.timedelta(minutes=start_buffer_minutes),
                                       end_buffer_td=dt_utils.timedelta(minutes=end_buffer_minutes),
                                       apply_correction=True,
                                       debug=False)
    return rdvx_data


def export_dw_to_pickle(dw: DataWindow,
                        output_directory: str,
                        output_filename: Optional[str] = None,
                        event_name: Optional[str] = "Redvox") -> str:
    """
    Export DataWindow to pickle file

    :param dw: RedVox Datawindow object. REQUIRED
    :param output_directory: optional string, output directory to save pickle. Default is None
    :param output_filename: optional string, pickle filename. Default is None
    :param event_name: optional string, name of event. Default is "Redvox"

    :return: string with path of pickle
    """

    # if output_directory is None:  # set output directory
    #     output_directory = os.path.join(".", "rpd_files")
    #     if not os.path.exists(output_directory):  # make output directory if first time running build function
    #         print(f"Creating output directory: {output_directory}...")
    #         os.mkdir(output_directory)

    # else:  # check output directory exists
    if not os.path.exists(output_directory):  # make output directory if it doesn't exist
        print(f"Creating output directory: {output_directory}...")
        os.mkdir(output_directory)

    if output_filename is None:  # set output filename
        output_filename_dw: str = event_name
    # make sure the .pkl does not repeat when we save the DataWindow
    elif output_filename.find(".pickle") != -1:
        output_filename_dw = output_filename.replace(".pickle", "")
    elif output_filename.find(".pkl") != -1:
        output_filename_dw = output_filename.replace(".pkl", "")
    else:
        output_filename_dw = output_filename

    print(output_filename_dw)

    print("Exporting RedVox DataWindow JSON and Pickle...", end=" ")
    dw.to_json_file(base_dir=output_directory,
                    file_name=output_filename_dw)

    if output_filename_dw.find(".pkl") == -1 and output_filename_dw.find(".pickle") == -1:
        path_dw_pickle = os.path.join(output_directory, output_filename_dw + '.pkl')
        print(f"Done. Path:{path_dw_pickle}")

    else:
        path_dw_pickle = os.path.join(output_directory, output_filename_dw)
        print(f"Done. Path:{path_dw_pickle}")

    return path_dw_pickle


def plot_dw_mic(data_window: DataWindow) -> Figure:
    """
    Plot audio data for all stations in RedVox DataWindow

    :param data_window: RedVox DataWindow object. REQUIRED

    :return: matplotlib figure instance
    """
    station: Station
    f1, ax1 = plt.subplots(figsize=(10, 8))  # Adjust to your screen
    for k, station in enumerate(data_window.stations):
        if station.has_audio_data():
            mic_wf_raw = station.audio_sensor().get_data_channel("microphone")
            mic_epoch_s = station.audio_sensor().data_timestamps() / MICROSECONDS_IN_SECOND

            ax1.plot(mic_epoch_s-mic_epoch_s[0],
                     mic_wf_raw/np.nanmax(np.abs(mic_wf_raw)),
                     label=station.id)
            ax1.legend(loc='upper right')
            ax1.set_title("Audio raw normalized waveforms")
            ax1.set_xlabel("Time from record start, s")

    return f1


def plot_dw_baro(data_window: DataWindow) -> Figure:
    """
    Plot barometer data for all stations in RedVox DataWindow

    :param data_window: RedVox DataWindow object. REQUIRED

    :return: matplotlib figure instance
    """
    station: Station
    f1, ax1 = plt.subplots(figsize=(10, 8))  # Adjust to your screen
    for k, station in enumerate(data_window.stations):
        if station.has_barometer_data():
            baro_wf_raw = station.barometer_sensor().get_data_channel("pressure")
            baro_epoch_s = station.barometer_sensor().data_timestamps() / MICROSECONDS_IN_SECOND
            baro_wf = baro_wf_raw - np.nanmean(baro_wf_raw)
            ax1.plot(baro_epoch_s-baro_epoch_s[0],
                     baro_wf/np.nanmax(np.abs(baro_wf)),
                     label=station.id)
            ax1.legend(loc='upper right')
            ax1.set_title("Pressure raw normalized waveforms")
            ax1.set_xlabel("Time from record start, s")

    return f1


def create_dw_config_from_epoch(api_input_directory: str,
                                start_epoch_s: Optional[float] = None,
                                end_epoch_s: Optional[float] = None,
                                redvox_station_ids: Optional[List[str]] = None,
                                start_buffer_minutes: Optional[int] = 3,
                                end_buffer_minutes: Optional[int] = 3,
                                debug: Optional[bool] = False,
                                structured_layout: Optional[bool] = True) -> dwc.DataWindowConfig:
    """
    Create a RedVox DataWindow Configuration from epoch s

    :param api_input_directory: Redvox DataWindow, or string with directory that contains the files to read data from. REQUIRED
    :param start_epoch_s: optional float, start time in epoch s. Default is None
    :param end_epoch_s: optional float, end time in epoch s. Default is None
    :param redvox_station_ids: optional list of strings, list of station ids to filter on. Default is None, so all stations
        included
    :param start_buffer_minutes: float representing the amount of minutes to include before the start datetime
        when filtering data. Default is 3
    :param end_buffer_minutes: float representing the amount of minutes to include after the end datetime
        when filtering data. Default is 3
    :param debug: optional bool, print debug for DataWindow if True. Default is False
    :param structured_layout: if True, the input_directory contains specially named and organized directories of data. Default
        True

    :return: DataWindowConfig
    """

    # DataWindow Config Time defaults
    start_year_dw_config = None
    start_month_dw_config = None
    start_day_dw_config = None
    start_hour_dw_config = None
    start_minute_dw_config = None
    start_second_dw_config = None

    end_year_dw_config = None
    end_month_dw_config = None
    end_day_dw_config = None
    end_hour_dw_config = None
    end_minute_dw_config = None
    end_second_dw_config = None

    # Convert epoch time to year/month/day/hour/minute/second for DataWindowConfig
    if start_epoch_s is not None:
        start_datetime_object = datetime.utcfromtimestamp(start_epoch_s)

        start_year_dw_config = start_datetime_object.year
        start_month_dw_config = start_datetime_object.month
        start_day_dw_config = start_datetime_object.day
        start_hour_dw_config = start_datetime_object.hour
        start_minute_dw_config = start_datetime_object.minute
        start_second_dw_config = start_datetime_object.second

    if end_epoch_s is not None:
        end_datetime_object = datetime.utcfromtimestamp(end_epoch_s)

        end_year_dw_config = end_datetime_object.year
        end_month_dw_config = end_datetime_object.month
        end_day_dw_config = end_datetime_object.day
        end_hour_dw_config = end_datetime_object.hour
        end_minute_dw_config = end_datetime_object.minute
        end_second_dw_config = end_datetime_object.second

    dw_config = dwc.DataWindowConfig(input_directory=api_input_directory,
                                     structured_layout=structured_layout,
                                     apply_correction=True,
                                     station_ids=redvox_station_ids,
                                     start_year=start_year_dw_config,
                                     start_month=start_month_dw_config,
                                     start_day=start_day_dw_config,
                                     start_hour=start_hour_dw_config,
                                     start_minute=start_minute_dw_config,
                                     start_second=start_second_dw_config,
                                     end_year=end_year_dw_config,
                                     end_month=end_month_dw_config,
                                     end_day=end_day_dw_config,
                                     end_hour=end_hour_dw_config,
                                     end_minute=end_minute_dw_config,
                                     end_second=end_second_dw_config,
                                     start_padding_seconds=start_buffer_minutes * 60,
                                     end_padding_seconds=end_buffer_minutes * 60,
                                     debug=debug)
    return dw_config
