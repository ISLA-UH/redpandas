"""
DataWindow related modules.

Last updated: 24 June 2021
"""

# Python libraries
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
from datetime import datetime
import os

# RedVox modules
from redvox.common.data_window import DataWindow
from redvox.common.station import Station
from redvox.common.date_time_utils import MICROSECONDS_IN_SECOND
import redvox.common.data_window_configuration as dwc

# RedPandas config
from redpandas.redpd_config import RedpdConfig


def dw_from_config_epoch(config: RedpdConfig) -> DataWindow:
    """
    Create RedVox DataWindow object from configuration file with start/end times in epoch s

    :param config: RedpdConfig
    :return: RedVox DataWindow object
    """

    api_input_directory: str = config.input_dir
    redvox_station_ids: List[str] = config.station_ids
    start_epoch_s: float = config.event_start_epoch_s
    end_epoch_s: float = config.event_end_epoch_s
    start_buffer_minutes: int = config.start_buffer_minutes
    end_buffer_minutes: int = config.end_buffer_minutes

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
                                     structured_layout=True,
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
                                     end_padding_seconds=end_buffer_minutes * 60)

    rdvx_data: DataWindow = DataWindow.from_config(dw_config)

    return rdvx_data


def build_from_config(config: RedpdConfig) -> None:
    """
    Load data, construct data window, export as pickle using a configuration file

    :param config: RedpdConfig
    :return: RedVox DataWindow saved in pickle file and corresponding JSON file
    """
    build(api_input_directory=config.input_dir,
          event_name=config.event_name,
          output_directory=config.output_dir,
          output_filename=config.output_filename_pkl_pqt,
          redvox_station_ids=config.station_ids,
          start_epoch_s=config.event_start_epoch_s,
          end_epoch_s=config.event_end_epoch_s,
          start_buffer_minutes=config.start_buffer_minutes,
          end_buffer_minutes=config.end_buffer_minutes,
          debug=False)


def build(api_input_directory: str,
          event_name: Optional[str] = "Redvox",
          output_directory: Optional[str] = None,
          output_filename: Optional[str] = None,
          redvox_station_ids: Optional[List[str]] = None,
          start_epoch_s: Optional[float] = None,
          end_epoch_s: Optional[float] = None,
          start_buffer_minutes: Optional[int] = 3,
          end_buffer_minutes: Optional[int] = 3,
          debug: bool = False) -> None:

    """
    Load data, construct data window, export as pickle

    :param api_input_directory: directory where data is located
    :param event_name: name of event
    :param output_directory: directory where pickle and JSON files are saved. Default is api_input_directory
    :param output_filename: name for pickle and JSON files. Default is event_name + .pkl
    :param redvox_station_ids: ID of RedVox stations
    :param start_epoch_s: start time in epoch time in seconds
    :param end_epoch_s: end time in epoch time in seconds
    :param start_buffer_minutes: the amount of time to include before the start_epoch_s when filtering data.
    Default is 3.
    :param end_buffer_minutes: the amount of time to include after the end_epoch_s when filtering data. Default is 3.
    :param debug: Toggle DataWindow debug_dw on/off. Default is False.
    :return: RedVox DataWindow saved in pickle file and corresponding JSON file
    """

    print(f"Loading data and constructing RedVox DataWindow for: {event_name}...", end=" ")

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
                                     structured_layout=True,
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

    # Load signals
    rdvx_data = DataWindow.from_config(dw_config)
    print(f"Done.")
    print(f"RedVox SDK version: {rdvx_data.sdk_version}")

    if output_directory is None:  # set output directory
        output_directory = os.path.join(api_input_directory, "rpd_files")
        if not os.path.exists(output_directory):  # make output directory if first time running build function
            print(f"Creating output directory: {output_directory}...")
            os.mkdir(output_directory)

    else:  # check output directory exists
        if not os.path.exists(output_directory):  # make output directory if it doesn't exist
            print(f"Creating output directory: {output_directory}...")
            os.mkdir(output_directory)

    if output_filename is None:  # set output filename
        output_filename: str = event_name
    else:
        output_filename = output_filename.replace(".pkl", "")

    print("Exporting RedVox DataWindow JSON and Pickle...", end=" ")
    rdvx_data.to_json_file(base_dir=output_directory,
                           file_name=output_filename)

    if output_filename.find(".pkl") == -1:
        print(f"Done. Path:{os.path.join(output_directory,output_filename + '.pkl')}")
    else:
        print(f"Done. Path:{os.path.join(output_directory,output_filename)}")


def plot_dw_mic(data_window: DataWindow) -> None:
    """
    Plot audio data for all stations in RedVox DataWindow
    :param data_window: RedVox DataWindow object
    :return: plot
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


def plot_dw_baro(data_window: DataWindow) -> None:
    """
    Plot barometer data for all stations in RedVox DataWindow
    :param data_window: RedVox DataWindow object
    :return: plot
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
