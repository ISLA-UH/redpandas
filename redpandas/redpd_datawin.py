# Python libraries
import os.path
import pickle

import numpy as np
import matplotlib.pyplot as plt
# RedVox modules
from redvox.common.data_window import DataWindow
from redvox.common.io import serialize_data_window
from redvox.common.station import Station
import redvox.common.date_time_utils as dt
from redvox.common.date_time_utils import MICROSECONDS_IN_SECOND


def build(api_input_directory: str,
          start_epoch_s: float,
          end_epoch_s: float,
          redvox_station_ids,
          event_name: str,
          output_directory: str,
          output_filename: str,
          start_buffer_minutes: float = 3,
          end_buffer_minutes: float = 3,
          debug: bool = False):
    """
    Load data, construct data window, export as pickle

    :param api_input_directory: directory where data is located
    :param start_epoch_s: start time in epoch time in seconds
    :param end_epoch_s: end time in epoch time in seconds
    :param redvox_station_ids: ID of RedVox stations
    :param event_name: name of event
    :param output_filename: name for pickle and JSON files
    :param output_directory: directory where pickle and JSON files are saved
    :param start_buffer_minutes: the amount of time to include before the start_epoch_s when filtering data.
     Default is 3.
    :param end_buffer_minutes: the amount of time to include after the end_epoch_s when filtering data. Default is 3.
    :param debug: Toggle DataWindow debug on/off. Default is False.
    :return: RedVox DataWindow saved in pickle file and corresponding JSON file
    """

    print(f"Loading data and constructing RedVox DataWindow for {event_name}.", end=" ")
    # Load signals
    rdvx_data = DataWindow(input_dir=api_input_directory,
                           station_ids=redvox_station_ids,
                           start_datetime=dt.datetime_from_epoch_seconds_utc(start_epoch_s),
                           end_datetime=dt.datetime_from_epoch_seconds_utc(end_epoch_s),
                           apply_correction=True,
                           structured_layout=True,
                           start_buffer_td=dt.timedelta(minutes=start_buffer_minutes),
                           end_buffer_td=dt.timedelta(minutes=end_buffer_minutes),
                           debug=debug)

    print("Exporting RedVox DataWindow JSON and Pickle...", end=" ")
    rdvx_data.to_json_file(base_dir=output_directory,
                           file_name=output_filename)

    print("Done.")


def build_ez(api_input_directory, pickle_filename: str = "DataWindow.pickle",
             is_serialized: bool = True):
    """
    Load data, construct data window, export as pickle with minimal inputs
    :param api_input_directory: directory where data is located
    :param pickle_filename: name for pickle file
    :param is_serialized: is pickle file serialized. Default is True.
    :return: RedVox DataWindow saved in pickle file
    """

    print("Load data and construct DataWindow for data in ", api_input_directory)
    # Load signals
    rdvx_data = DataWindow(input_dir=api_input_directory)

    if is_serialized:
        # Save serialized lz4 data window
        serialize_data_window(data_window=rdvx_data,
                              base_dir=api_input_directory,
                              file_name=pickle_filename)
        print("Exported serialized data window as pickle")
    else:
        with open(api_input_directory + pickle_filename, 'wb') as f:
            pickle.dump(rdvx_data, f)
        print("Exported data window as pickle")


def plot_dw_mic(data_window):
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


def plot_dw_baro(data_window):
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
