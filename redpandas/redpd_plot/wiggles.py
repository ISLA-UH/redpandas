"""
Plot waveforms in dataframe
"""
import datetime as dt
from typing import List, Union, Optional
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from redpandas.redpd_plot.parameters import FigureParameters as FigParam


# PLOT_WIGGLES AUXILIARY FUNCTIONS
def find_wiggle_num(df: pd.DataFrame,
                    sig_wf_label: Union[List[str], str] = "audio_wf",
                    sig_timestamps_label: Union[List[str], str] = "audio_epoch_s",
                    sig_id_label: str = "station_id",
                    station_id_str: str = None) -> int:
    """
    Determine number of wiggles to plot

    :param df: input pandas dataframe. REQUIRED
    :param sig_wf_label: single string or list of strings for the waveform column name in df. Default is "audio_wf". For example, for
        multiple sensor waveforms: sig_wf_label = ["audio_wf", "barometer_wf_highpass", "accelerometer_wf_highpass"]
    :param sig_timestamps_label: optional string or list of strings for column label in df with epoch time. Default is "audio_epoch_s"
    :param sig_id_label: string for the station id column name in df. Default is "station_id"
    :param station_id_str: optional string with name of one station to plot. Default is None
    :return: int, number of wiggles
    """
    # Assumes accelerometer, gyrsocope and magnetometer will have three channels
    dict_wiggle_num = {"audio_wf": 1,
                       'sig_aligned_wf': 1,
                       "barometer_wf_raw": 1,
                       "barometer_wf_highpass": 1,
                       "accelerometer_wf_raw": 3,
                       "accelerometer_wf_highpass": 3,
                       "gyroscope_wf_raw": 3,
                       "gyroscope_wf_highpass": 3,
                       "magnetometer_wf_raw": 3,
                       "magnetometer_wf_highpass": 3}

    wiggle_num_list = []  # number of wiggles

    for index_sensor_in_list, sensor_in_list in enumerate(sig_wf_label):
        for index_n in df.index:
            if station_id_str is None or df[sig_id_label][index_n].find(station_id_str) != -1:
                # check column exists and not empty
                if sensor_in_list in df.columns and type(df[sensor_in_list][index_n]) != float and \
                        df[sig_timestamps_label[index_sensor_in_list]][index_n] is not None:

                    if dict_wiggle_num.get(sensor_in_list) is not None:
                        # Get number of wiggles per sensor
                        wiggle_num_list.append(dict_wiggle_num.get(sensor_in_list))
                    else:
                        # Get number of wiggles per sensor
                        if df[sensor_in_list][index_n].ndim == 1:
                            wiggle_num_list.append(1)
                        elif df[sig_id_label][index_n].find("pressure") == 0 or \
                                df[sig_id_label][index_n].find("bar") == 0 or \
                                sensor_in_list.find("bar") == 0 or \
                                sensor_in_list.find("pressure") == 0:
                            wiggle_num_list.append(1)
                        else:
                            # Assume if not audio related wf, it is 3c sensors
                            wiggle_num_list.append(3)
                else:
                    continue

    wiggle_num = sum(wiggle_num_list)  # total number of signal that will be displayed
    return wiggle_num


def find_ylabel(df: pd.DataFrame,
                sig_wf_label: Union[List[str], str] = "audio_wf",
                sig_timestamps_label: Union[List[str], str] = "audio_epoch_s",
                sig_id_label: str = "station_id",
                station_id_str: Optional[str] = None,
                custom_yticks: Optional[Union[List[str], str]] = None) -> List:
    """
    Determine ylabels that will be used

    :param df: input pandas dataframe. REQUIRED
    :param sig_wf_label: single string or list of strings for the waveform column name in df. Default is "audio_wf". For example, for
        multiple sensor waveforms: sig_wf_label = ["audio_wf", "barometer_wf_highpass", "accelerometer_wf_highpass"]
    :param sig_timestamps_label: optional string or list of strings for column label in df with epoch time. Default is "audio_epoch_s"
    :param sig_id_label: optional string for the station id column name in df. Default is "station_id"
    :param station_id_str: optional string with name of one station to plot. Default is None
    :param custom_yticks:

    :return: list of y tick labels
    """

    dict_yticks = {"audio_wf": ["aud"],
                   'sig_aligned_wf': ["sig"],
                   "barometer_wf_raw": ["bar raw"],
                   "barometer_wf_highpass": ["bar hp"],
                   "accelerometer_wf_raw": ["acc X raw", "acc Y raw", "acc Z raw"],
                   "accelerometer_wf_highpass": ["acc X hp", "acc Y hp", "acc Z hp"],
                   "gyroscope_wf_raw": ["gyr X raw", "gyr Y raw", "gyr Z raw"],
                   "gyroscope_wf_highpass": ["gyr X hp", "gyr Y hp", "gyr Z hp"],
                   "magnetometer_wf_raw": ["mag X raw", "mag Y raw", "mag Z raw"],
                   "magnetometer_wf_highpass": ["mag X hp", "mag Y hp", "mag Z hp"]}

    # if custom_yticks provided, make that the yticks
    if custom_yticks is not None and custom_yticks != "index":
        wiggle_yticklabel = custom_yticks

    else:  # construct the yticks:
        wiggle_yticklabel = []  # name/y label of wiggles

        for index_sensor_in_list, sensor_in_list in enumerate(sig_wf_label):  # loop for every sensor
            for index_n in df.index:  # loop for every station
                if station_id_str is None or df[sig_id_label][index_n].find(station_id_str) != -1:
                    # check column exists and not empty
                    if sensor_in_list in df.columns and type(df[sensor_in_list][index_n]) != float and \
                            df[sig_timestamps_label[index_sensor_in_list]][index_n] is not None:
                        # Get yticks
                        if custom_yticks == "index":  # if ylabel for wiggle is index station
                            list_index = [df.index[index_n]] * len(dict_yticks.get(sensor_in_list))
                            wiggle_yticklabel += list_index

                        elif custom_yticks is None:
                            # try:
                            sensor_short = dict_yticks.get(sensor_in_list)

                            if sensor_short is not None:
                                # if sensor_short is not None:
                                if station_id_str is not None:
                                    # if only doing one station, yticks just name sensors
                                    wiggle_yticklabel += sensor_short
                                # If only one 3c sensor, skip, but if only doing one 1c sensor, yticks just name station
                                elif len(sig_wf_label) == 1 and sensor_in_list.find("acc") == -1 \
                                        and sensor_in_list.find("gyr") == -1 \
                                        and sensor_in_list.find("mag") == -1:
                                    wiggle_yticklabel.append(f"{df[sig_id_label][index_n]}")
                                else:  # if multiple sensors and stations, yticks both station and sensors
                                    station_and_sensor = [f"{df[sig_id_label][index_n]} " + element for element in sensor_short]
                                    wiggle_yticklabel += station_and_sensor
                            else:
                                if len(sig_wf_label) == 1:
                                    wiggle_yticklabel.append(f"{df[sig_id_label][index_n]}")
                                else:
                                    wiggle_yticklabel.append(f"{df[sig_id_label][index_n]} {sensor_in_list}")
                    else:
                        continue

    return wiggle_yticklabel


def determine_time_epoch_origin(df: pd.DataFrame,
                                sig_id_label: str = "station_id",
                                sig_timestamps_label: Union[List[str], str] = "audio_epoch_s",
                                station_id_str: Optional[str] = None) -> float:
    """
    Get time epoch origin for all sensors for all stations to establish the earliest timestamp

    :param df: input pandas dataframe. REQUIRED
    :param sig_id_label: optional string for the station id column name in df. Default is "station_id"
    :param sig_timestamps_label: optional string or list of strings for column label in df with epoch time. Default is "audio_epoch_s"
    :param station_id_str: optional string with name of one station to plot. Default is None

    :return: time_epoch_origin
    """

    if type(sig_timestamps_label) == str:
        sig_timestamps_label = [sig_timestamps_label]

    # Establish min xlim aka min time
    epoch_j = []

    # loop though each sensor in station
    for index_time_label, sensor_time_label in enumerate(sig_timestamps_label):
        for index_station in df.index:
            # No station indicated, or station indicated and found
            if station_id_str is None or df[sig_id_label][index_station].find(station_id_str) != -1:
                # check that the time column exists first
                if sensor_time_label not in df.columns:  # check column exists
                    raise ValueError(f"the column name {sensor_time_label} was not found in the dataframe")

                elif type(df[sensor_time_label][index_station]) == float or df[sensor_time_label][index_station] is None:  # not an array, so a Nan
                    continue  # skip cause entry for this station is empty
                else:
                    epoch_j.append(df[sensor_time_label][index_station].min())

    epoch_j = np.array(epoch_j)
    try:
        time_epoch_origin = np.min(epoch_j[np.nonzero(epoch_j)])
    except ValueError:  # unless it so happens all min values are 0
        time_epoch_origin = 0.0

    return time_epoch_origin


# PLOT_WIGGLES
def plot_wiggles_pandas(df: pd.DataFrame,
                        sig_wf_label: Union[List[str], str] = "audio_wf",
                        sig_timestamps_label: Union[List[str], str] = "audio_epoch_s",
                        sig_id_label: str = "station_id",
                        station_id_str: Optional[str] = None,
                        fig_title_show: bool = True,
                        fig_title: str = 'Signals',
                        custom_yticks: Optional[Union[List[str], str]] = None,
                        show_figure: bool = True) -> Figure:

    """
    More nuanced plots with minimal distraction. Optimized for pandas input.
    Defualt is audio, to plot other sensors add the relevant column labels in sig_wf_label and sig_timestamps_label parameters.
    For more information on available columns in dataframe, visit:
    https://github.com/RedVoxInc/redpandas/blob/master/docs/redpandas/columns_name.md#redpandas-dataframe-columns

    :param df: input pandas data frame. REQUIRED
    :param sig_wf_label: single string or list of strings for the waveform column name in df. Default is "audio_wf". For example, for
        multiple sensor waveforms: sig_wf_label = ["audio_wf", "barometer_wf_highpass", "accelerometer_wf_highpass"]
    :param sig_timestamps_label: string or list of strings for column label in df with epoch time. Default is "audio_epoch_s". For example, for
        multiple sensor timestamps: sig_timestamps_label = ["audio_epoch_s", "barometer_epoch_s", "accelerometer_epoch_s"]
    :param sig_id_label: string for the station id column name in df. Default is "station_id"
    :param station_id_str: string with name of one station to plot only that station. Default is None
    :param fig_title_show: optional bool, include a title in the figure if True. Default is True
    :param fig_title: optional string, 'Normalized' + title label. Default is "signals"
    :param custom_yticks: optional, provide custom names for yticks, list of strings (one label per channel component) or
        "index" for station index in dataframe. For example, for multiple sensors: custom_y_ticks = ["audio", "bar", "acc X", "acc Y", "acc Z"].
         Another example, for multiple stations with 1 sensor
    :param show_figure: optional bool, show figure if True. Default is True

    :return: matplotlib figure instance
    """
    # Create List of signal channels to loop through later
    # If given only one, aka a sting, make it a list of length 1
    if type(sig_timestamps_label) == str:
        sig_timestamps_label = [sig_timestamps_label]
    if type(sig_wf_label) == str:
        sig_wf_label = [sig_wf_label]

    # Check same length of waveform columns and timestamps
    if len(sig_wf_label) != len(sig_timestamps_label):
        raise ValueError(f"The number of waveform columns provided in sig_wf_label ({len(sig_wf_label)}) must be the "
                         f"same as the number of timestamps columns provided in sig_timestamps_label "
                         f"({len(sig_timestamps_label)})")

    if station_id_str is not None:  # check station input exists
        if station_id_str in df[sig_id_label].unique() is False:
            raise ValueError(f"station_id_str parameter provided ('{station_id_str}') "
                             f"was not found in dataframe")

    # Get wiggle number, yticks label
    wiggle_num = find_wiggle_num(df=df,
                                 sig_wf_label=sig_wf_label,
                                 sig_timestamps_label=sig_timestamps_label,
                                 sig_id_label=sig_id_label,
                                 station_id_str=station_id_str)
    wiggle_yticklabel = find_ylabel(df=df,
                                    sig_wf_label=sig_wf_label,
                                    sig_timestamps_label=sig_timestamps_label,
                                    sig_id_label=sig_id_label,
                                    station_id_str=station_id_str,
                                    custom_yticks=custom_yticks)
    # # For debugging
    # print("Wiggle num:", wiggle_num)
    # print("Wiggle ylabel:", wiggle_yticklabel)

    # Check wiggle_num and # of ylabels match
    if len(wiggle_yticklabel) != wiggle_num:
        raise ValueError(f"The number of labels provided in the custom_yticks parameter ({len(wiggle_yticklabel)}) "
                         f"does not match the number of signal channels provided in sig_wf_label "
                         f"or the number of stations in dataframe ({wiggle_num})."
                         f"\nDo not forget that accelerometer, gyroscope, and magnetometer have X, Y and Z components "
                         f"so a label is required for each component.")

    # Wiggle scaling
    offset_scaling = 2**(np.log2(wiggle_num)+1.0)/wiggle_num
    wiggle_offset = np.arange(0, wiggle_num)*offset_scaling
    wiggle_yticks = wiggle_offset

    # Set up figure
    if show_figure:
        fig, ax1 = plt.subplots(figsize=(FigParam().figure_size_x, FigParam().figure_size_y))
    else:
        fig: Figure = Figure(figsize=(FigParam().figure_size_x, FigParam().figure_size_y))
        ax1 = fig.subplots()
    ax1.set_yticks(wiggle_yticks)
    ax1.set_yticklabels(wiggle_yticklabel)
    ax1.set_ylim(wiggle_offset[0]-offset_scaling, wiggle_offset[-1]+offset_scaling)
    ax1.tick_params(axis='both', which='both', labelsize=FigParam().text_size)

    # Get first timestamps out of all the sensors for all stations to establish xlim min
    time_epoch_origin = determine_time_epoch_origin(df=df,
                                                    sig_timestamps_label=sig_timestamps_label,
                                                    station_id_str=station_id_str,
                                                    sig_id_label=sig_id_label)
    # Set up xlim min and max arrays.
    xlim_min = np.empty(wiggle_num)
    xlim_max = np.empty(wiggle_num)

    index_sensor_label_ticklabels_list = 0
    # for index_sensor_label_ticklabels_list, index_station in enumerate(df.index):  # loop per station

    for index_sensor_in_list, label in enumerate(sig_wf_label):  # loop per sensor
        for index_station in df.index:  # loop per station
            # first things first, check if column with data exists and if there is data in it:
            if label not in df.columns or type(df[label][index_station]) == float or \
                    df[sig_timestamps_label[index_sensor_in_list]][index_station] is None:
                print(f"SensorMissingException: The column {label} was not found in DataFrame or no data available in "
                      f"{label} for station {df[sig_id_label][index_station]}")
                continue  # if not, skip this iteration

            if station_id_str is None or df[sig_id_label][index_station].find(station_id_str) != -1:
                sensor_timestamps_label = sig_timestamps_label[index_sensor_in_list]  # timestamps
                time_s = df[sensor_timestamps_label][index_station] - time_epoch_origin  # scrubbed clean time

                for sensor_array in df[label][index_station]:  # Make a regular loop
                    if label == "audio_wf":  # or "sig_aligned_wf":
                        sig_j = df[label][index_station] / np.max(df[label][index_station])
                    elif label == "sig_aligned_wf":
                        sig_j = df[label][index_station] / np.max(df[label][index_station])
                    elif df[label][index_station].ndim == 1:
                        sig_j = df[label][index_station] / np.max(df[label][index_station])
                    else:
                        sig_j = sensor_array / np.max(sensor_array)

                    ax1.plot(time_s, sig_j + wiggle_offset[index_sensor_label_ticklabels_list],
                             color='midnightblue')
                    xlim_min[index_sensor_label_ticklabels_list] = np.min(time_s)
                    xlim_max[index_sensor_label_ticklabels_list] = np.max(time_s)

                    index_sensor_label_ticklabels_list += 1

                    if label == "audio_wf":
                        break
                    if label == "sig_aligned_wf":
                        break
                    if df[label][index_station].ndim == 1:
                        break

    ax1.set_xlim(np.min(xlim_min), np.max(xlim_max))  # Set xlim min and max
    ax1.grid(True)
    if fig_title_show:  # Set title
        if station_id_str is None and len(sig_wf_label) > 1:
            ax1.set_title(f'Normalized {fig_title}', size=FigParam().text_size)
        elif station_id_str is None and len(sig_wf_label) == 1:
            ax1.set_title(f'Normalized {fig_title} for {sig_wf_label[0]}', size=FigParam().text_size)
        else:
            ax1.set_title(f'Normalized {fig_title} for Station {station_id_str}', size=FigParam().text_size)

    # Set ylabel
    if station_id_str is None:
        ax1.set_ylabel("Signals", size=FigParam().text_size)
    else:
        ax1.set_ylabel("Sensors", size=FigParam().text_size)

    x_label = "Time (s)"
    if time_epoch_origin > 0:
        x_label += " relative to UTC " + dt.datetime.utcfromtimestamp(time_epoch_origin).strftime('%Y-%m-%d %H:%M:%S')
    ax1.set_xlabel(x_label, size=FigParam().text_size)
    fig.tight_layout()

    if show_figure is True:
        plt.show()

    return fig
