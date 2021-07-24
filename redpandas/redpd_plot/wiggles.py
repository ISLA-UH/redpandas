"""

"""
import datetime as dt
from typing import List, Union, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

# Wiggle plot scaling
scale = 1.25*1080/8
figure_size_x = int(1920/scale)
figure_size_y = int(1080/scale)
text_size = int(2.9*1080/scale)

# Colormap/
color_map = "inferno"  # 'hot_r'  # 'afmhot_r' #colormap for plotting


# PLOT_WIGGLES AUXILIARY FUNCTIONS
def find_wiggle_num_yticks(df: pd.DataFrame,
                           sig_wf_label: Union[List[str], str] = "audio_wf",
                           sig_id_label: str = "station_id",
                           station_id_str: Optional[str] = None,
                           custom_yticks: Optional[Union[List[str], str]] = None) -> Tuple[int, List]:
    """
    Determine number of wiggles and ylabels that will be used

    :param df: input pandas dataframe. REQUIRED
    :param sig_wf_label: single string or list of strings for the waveform column name in df. Default is "audio_wf". For example, for
        multiple sensor waveforms: sig_wf_label = ["audio_wf", "barometer_wf_highpass", "accelerometer_wf_highpass"]
    :param sig_id_label: optional string for the station id column name in df. Default is "station_id"
    :param station_id_str: optional string with name of one station to plot. Default is None
    :param custom_yticks:

    :return: number of wiggles, list of y tick labels
    """
    dict_yticks = {"audio_wf": "aud",
                   "barometer_wf_raw[0]": "bar raw",
                   "barometer_wf_highpass[0]": "bar hp",
                   "accelerometer_wf_raw[0]": "acc X raw",
                   "accelerometer_wf_raw[1]": "acc Y raw",
                   "accelerometer_wf_raw[2]": "acc Z raw",
                   "accelerometer_wf_highpass[0]": "acc X hp",
                   "accelerometer_wf_highpass[1]": "acc Y hp",
                   "accelerometer_wf_highpass[2]": "acc Z hp",
                   "gyroscope_wf_raw[0]": "gyr X raw",
                   "gyroscope_wf_raw[1]": "gyr Y raw",
                   "gyroscope_wf_raw[2]": "gyr Z raw",
                   "gyroscope_wf_highpass[0]": "gyr X hp",
                   "gyroscope_wf_highpass[1]": "gyr Y hp",
                   "gyroscope_wf_highpass[2]": "gyr Z hp",
                   "magnetometer_wf_raw[0]": "mag X raw",
                   "magnetometer_wf_raw[1]": "mag Y raw",
                   "magnetometer_wf_raw[2]": "mag Z raw",
                   "magnetomer_wf_highpass[0]": "mag X hp",
                   "magnetomer_wf_highpass[1]": "mag Y hp",
                   "magnetomer_wf_highpass[2]": "mag Z hp"}

    wiggle_num_list = []  # number of wiggles
    wiggle_yticklabel = []  # name/y label of wiggles

    if station_id_str is not None:  # check station input exists
        if check_if_station_exists_in_df(df=df,
                                         station_id_str=station_id_str,
                                         sig_id_label=sig_id_label) is False:

            raise ValueError(f"station_id_str parameter provided ('{station_id_str}') "
                             f"was not found in dataframe")

    for index_sensor_in_list, sensor_in_list in enumerate(sig_wf_label):
        for index_n in df.index:
            # first things first, check if column exists:
            if check_if_column_exists_in_df(df=df, label=sensor_in_list) is False:
                continue  # if not, skip this iteration
            # check if empty
            if type(df[sensor_in_list][index_n]) == float:  # not an array, so a Nan
                continue  # skip cause entry for this station is empty

            if station_id_str is None or df[sig_id_label][index_n].find(station_id_str) != -1:
                if df[sensor_in_list][index_n].ndim == 1:  # aka audio
                    wiggle_num_list.append(1)  # append 1 wiggle cause it will only be one panel

                    # Establish ylabel for wiggle
                    if custom_yticks == "index":  # if ylabel for wiggle is index station
                        wiggle_yticklabel.append(df.index[index_n])

                    elif custom_yticks is None:
                        sensor_short = dict_yticks.get(sensor_in_list)
                        if station_id_str is not None:  # if only doing one station, yticks just name sensors
                            wiggle_yticklabel.append(f"{sensor_short}")
                        elif len(sig_wf_label) == 1 and len(sig_wf_label) == 1:  # if only doing one sensor, yticks just name station
                            wiggle_yticklabel.append(f"{df[sig_id_label][index_n]}")
                        else:  # if multiple sensors and stations, yticks both station and sensors
                            wiggle_yticklabel.append(f"{df[sig_id_label][index_n]} {sensor_short}")

                else:
                    # Check if barometer, cause then only 1 wiggle
                    if sensor_in_list.find("pressure") == 0 or sensor_in_list.find("bar") == 0:
                        wiggle_num_list.append(1)
                    else:  # if not barometer, its is a 3c sensors aka gyroscope/accelerometer/magnetometer
                        wiggle_num_list.append(3)

                    for index_dimension, _ in enumerate(df[sensor_in_list][index_n]):
                        # Establish ylabel for wiggle
                        if custom_yticks == "index":  # if ylabel for wiggle is index station
                            wiggle_yticklabel.append(df.index[index_n])

                        elif custom_yticks is None:
                            sensor_short = dict_yticks.get(f"{sensor_in_list}[{index_dimension}]")
                            if station_id_str is not None:  # if only doing one station, yticks just name sensors
                                wiggle_yticklabel.append(f"{sensor_short}")
                            elif len(sig_wf_label) == 1 and len(sig_wf_label) == 1:  # if only doing one sensor, yticks just name station
                                wiggle_yticklabel.append(f"{df[sig_id_label][index_n]}")
                            else:  # if multiple sensors and stations, yticks both station and sensors
                                wiggle_yticklabel.append(f"{df[sig_id_label][index_n]} {sensor_short}")

    # if custom_yticks provided, make that the yticks
    if custom_yticks is not None and custom_yticks != "index":
        wiggle_yticklabel = custom_yticks

    wiggle_num = sum(wiggle_num_list)  # total number of signal that will be displayed

    # Make sure wiggle_num and # of ylabels match to avoid problems later on
    if len(wiggle_yticklabel) != wiggle_num:
        raise ValueError(f"The number of labels provided in the custom_yticks parameter ({len(wiggle_yticklabel)}) "
                         f"does not match the number of signal channels provided in sig_wf_label "
                         f"or the number of stations in dataframe ({wiggle_num})."
                         f"\nDo not forget that accelerometer, gyroscope, and magnetometer have X, Y and Z components "
                         f"so a label is required for each component.")
    else:
        return wiggle_num, wiggle_yticklabel


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
    for index_station in df.index:
        # No station indicated, or station indicated and found
        if station_id_str is None or df[sig_id_label][index_station].find(station_id_str) != -1:

            # loop though each sensor in station
            for index_time_label, sensor_time_label in enumerate(sig_timestamps_label):
                # check that the time column exists first
                if check_if_column_exists_in_df(df=df, label=sensor_time_label) is False:
                    raise ValueError(f"the column name ({sensor_time_label}) was not found in the dataframe")

                elif type(df[sensor_time_label][index_station]) == float:  # not an array, so a Nan
                    continue  # skip cause entry for this station is empty

                else:
                    epoch_j.append(df[sensor_time_label][index_station].min())

    epoch_j = np.array(epoch_j)
    try:
        time_epoch_origin = np.min(epoch_j[np.nonzero(epoch_j)])
    except ValueError:  # unless it so happens all min values are 0
        time_epoch_origin = 0.0

    return time_epoch_origin


def check_if_column_exists_in_df(df: pd.DataFrame,
                                 label: str):
    """
    Check if column is in dataframe.
    Based on the assumption that there is data in the column if it exists.

    :param df: input pandas dataframe. REQUIRED
    :param label: string with column name. REQUIRED

    :return: False if label not in df
    """
    return label in df.columns


def check_if_station_exists_in_df(df: pd.DataFrame,
                                  station_id_str: Union[str, None],
                                  sig_id_label: str = "station_id"):
    """
    Check if station is in dataframe.

    :param df: input pandas dataframe. REQUIRED
    :param station_id_str: string with name of one station to plot only that station. REQUIRED
    :param sig_id_label: string for the station id column name in df. Default is "station_id"

    :return: False if label not in df
    """

    return station_id_str in df[sig_id_label].unique()


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

    # Check same amount of waveform columns and timestamps provided
    if len(sig_wf_label) != len(sig_timestamps_label):
        raise ValueError(f"The number of waveform columns provided in sig_wf_label ({len(sig_wf_label)}) must be the "
                         f"same as the number of timestamps columns provided in sig_timestamps_label "
                         f"({len(sig_timestamps_label)})")

    # Get wiggle number, yticks label
    wiggle_num, wiggle_yticklabel = find_wiggle_num_yticks(df=df,
                                                           sig_wf_label=sig_wf_label,
                                                           sig_id_label=sig_id_label,
                                                           station_id_str=station_id_str,
                                                           custom_yticks=custom_yticks)

    # Wiggle scaling
    offset_scaling = 2**(np.log2(wiggle_num)+1.0)/wiggle_num
    wiggle_offset = np.arange(0, wiggle_num)*offset_scaling
    wiggle_yticks = wiggle_offset

    # Set up figure
    fig, ax1 = plt.subplots(figsize=(figure_size_x, figure_size_y))
    ax1.set_yticks(wiggle_yticks)
    ax1.set_yticklabels(wiggle_yticklabel)
    ax1.set_ylim(wiggle_offset[0]-offset_scaling, wiggle_offset[-1]+offset_scaling)
    ax1.tick_params(axis='both', which='both', labelsize=text_size)

    # Get t0 out of all the sensors for all stations
    time_epoch_origin = determine_time_epoch_origin(df=df,
                                                    sig_timestamps_label=sig_timestamps_label,
                                                    station_id_str=station_id_str,
                                                    sig_id_label=sig_id_label)
    # Set up xlim min and max arrays.
    xlim_min = np.empty(wiggle_num)
    xlim_max = np.empty(wiggle_num)

    index_sensor_label_ticklabels_list = 0  # keep track of total sensor wf including x/y/z per station

    for index_station in df.index:  # loop per station
        for index_sensor_in_list, label in enumerate(sig_wf_label):  # loop per sensor

            # first things first, check if column with data exists and if there is data in it:
            if check_if_column_exists_in_df(df=df, label=label) is False or type(df[label][index_station]) == float:
                print(f"SensorMissingException: The column {label} was not found in DataFrame or no data available in "
                      f"{label} for station {df[sig_id_label][index_station]}")
                continue  # if not, skip this iteration

            if station_id_str is None or df[sig_id_label][index_station].find(station_id_str) != -1:

                sensor_wf_df = df[label][index_station]
                sensor_timestamps_label = sig_timestamps_label[index_sensor_in_list]
                time_s = df[sensor_timestamps_label][index_station] - time_epoch_origin

                if sensor_wf_df.ndim == 1:  # sensor that is NOT acceleration/gyroscope/magnetometer

                    sig_j = df[label][index_station] / np.max(df[label][index_station])
                    ax1.plot(time_s, sig_j + wiggle_offset[index_sensor_label_ticklabels_list], color='midnightblue')
                    xlim_min[index_sensor_label_ticklabels_list] = np.min(time_s)
                    xlim_max[index_sensor_label_ticklabels_list] = np.max(time_s)

                    index_sensor_label_ticklabels_list += 1

                else:  # sensor that is acceleration/gyroscope/magnetometer/barometer
                    for index_dimension, sensor_array in enumerate(sensor_wf_df):

                        sig_j = sensor_array / np.max(sensor_array)
                        ax1.plot(time_s, sig_j + wiggle_offset[index_sensor_label_ticklabels_list], color='midnightblue')
                        xlim_min[index_sensor_label_ticklabels_list] = np.min(time_s)
                        xlim_max[index_sensor_label_ticklabels_list] = np.max(time_s)

                        index_sensor_label_ticklabels_list += 1

    ax1.set_xlim(np.min(xlim_min), np.max(xlim_max))
    ax1.grid(True)
    if fig_title_show:
        if station_id_str is None and len(sig_wf_label) > 1:
            ax1.set_title(f'Normalized {fig_title}', size=text_size)
        elif station_id_str is None and len(sig_wf_label) == 1:
            ax1.set_title(f'Normalized {fig_title} for {sig_wf_label[0]}', size=text_size)
        else:
            ax1.set_title(f'Normalized {fig_title} for Station {station_id_str}', size=text_size)

    # Set ylabel
    if station_id_str is None:
        ax1.set_ylabel("Signals", size=text_size)
    else:
        ax1.set_ylabel("Sensors", size=text_size)

    x_label = "Time (s)"
    if time_epoch_origin > 0:
        x_label += " relative to " + dt.datetime.utcfromtimestamp(time_epoch_origin).strftime('%Y-%m-%d %H:%M:%S')
    ax1.set_xlabel(x_label, size=text_size)
    fig.tight_layout()

    if show_figure is True:
        plt.show()

    return fig
