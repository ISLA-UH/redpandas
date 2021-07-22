"""
This module contains main utils for plotting RedPandas DataFrames.
"""

import datetime as dt
from typing import List, Union, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from libquantum.plot_templates import plot_time_frequency_reps as pnl

import redpandas.redpd_scales as rpd_scales

# Wiggle plot scaling
scale = 1.25*1080/8
figure_size_x = int(1920/scale)
figure_size_y = int(1080/scale)
text_size = int(2.9*1080/scale)

# Colormap/
color_map = "inferno"  # 'hot_r'  # 'afmhot_r' #colormap for plotting


def plot_mesh_pandas(df: pd.DataFrame,
                     mesh_time_label: Union[str, List[str]],
                     mesh_frequency_label: Union[str, List[str]],
                     mesh_tfr_label: Union[str, List[str]],
                     sig_id_label: Union[str, List[str]],
                     t0_sig_epoch_s: float = None,
                     fig_title_show: bool = True,
                     fig_title: str = "STFT",
                     frequency_scaling: str = "log",
                     frequency_hz_ymin: float = rpd_scales.Slice.FU,
                     frequency_hz_ymax: float = rpd_scales.Slice.F0,
                     common_colorbar: bool = True,
                     mesh_color_scaling: Union[List[str], str] = 'auto',
                     mesh_color_range: Union[List[float], float] = 15,
                     show_figure: bool = True) -> Figure:

    """
     Plots spectrogram for all signals in df

     :param df: input pandas data frame
     :param mesh_time_label: string for the mesh time column name in df. List of strings for multiple
     :param mesh_frequency_label: string for the mesh frequency column name in df. List of strings for multiple
     :param mesh_tfr_label: string for the mesh tfr column name in df. List of strings for multiple
     :param sig_id_label: string for column name with station ids in df. You can also provide a list of custom labels
      :param t0_sig_epoch_s: epoch time in seconds of first timestamp. Default is None
     :param fig_title_show: include a title in the figure. Default is True
     :param fig_title: figure title label. Default is "STFT"
     :param frequency_scaling: "log" or "lin". Default is "log"
     :param frequency_hz_ymin: y axis min lim
     :param frequency_hz_ymax: y axis max lim
     :param common_colorbar: display a colorbar for all mesh panels. Default is True
     :param mesh_color_scaling: colorbar scaling, "auto" or "range". Default is 'auto'
     :param mesh_color_range: Default is 15
     :param show_figure: show figure is True. Default is True
     :return: matplotlib figure instance
     """

    # Create List of mesh tfr to loop through later
    # If given only one, aka a sting, make it a list of length 1
    if type(mesh_tfr_label) == str:
        mesh_tfr_label = [mesh_tfr_label]
    if type(mesh_time_label) == str:
        mesh_time_label = [mesh_time_label]
    if type(mesh_frequency_label) == str:
        mesh_frequency_label = [mesh_frequency_label]

    # Check mesh, time and frequency are the same length:
    if len(mesh_tfr_label) != len(mesh_time_label) or len(mesh_tfr_label) != len(mesh_frequency_label) or \
            len(mesh_time_label) != len(mesh_frequency_label):
        print("mesh_time_label, mesh_tfr_label, or mesh_frequency_label do not have the same length. Please check.")
        exit()

    # Determine overall number of mesh panels in fig
    wiggle_num_list = []  # number of wiggles
    wiggle_yticklabel = []  # name/y label of wiggles
    for mesh_n in range(len(mesh_tfr_label)):
        mesh_tfr_label_individual = mesh_tfr_label[mesh_n]  # individual mesh label from list

        for n in df.index:
            if df[mesh_tfr_label_individual][n].ndim == 2:  # aka audio
                wiggle_num_list.append(1)  # append 1 wiggle cause it will only be one tfr panel

                # Establish ylabel for wiggle
                if type(sig_id_label) == str:
                    if sig_id_label == "index":  # if ylabel for wiggle is index station
                        wiggle_yticklabel.append(df.index[n])
                    else:
                        wiggle_yticklabel.append(df[sig_id_label][n])  # if ylabel for wiggles is custom list

            else:
                # Check if barometer, cause then only 1 wiggle
                if mesh_tfr_label_individual.find("pressure") == 0 or mesh_tfr_label_individual.find("bar") == 0:
                    wiggle_num_list.append(1)
                else:  # if not barometer, its is a 3c sensors aka gyroscope/accelerometer/magnetometer
                    wiggle_num_list.append(3)

                for index_dimension, _ in enumerate(df[mesh_tfr_label_individual][n]):

                    # Establish ylabel for wiggle
                    if type(sig_id_label) == str:
                        if sig_id_label == "index":  # if ylabel for wiggle is index station
                            wiggle_yticklabel.append(df.index[n])
                        else:
                            wiggle_yticklabel.append(df[sig_id_label][n])  # if ylabel for wiggles is custom list

    # if wiggle_yticklabel is not index or a custom list of names, just take the column label name provided
    if len(wiggle_yticklabel) == 0:
        wiggle_yticklabel = sig_id_label

    wiggle_num = sum(wiggle_num_list)  # total number of signal that will be displayed

    # loop to find xlim and tfr global max/min
    x_lim_min = np.empty(wiggle_num)
    x_lim_max = np.empty(wiggle_num)

    if common_colorbar is True:
        tfr_min = np.empty(wiggle_num)
        tfr_max = np.empty(wiggle_num)

    index_wiggle_num_total = 0  # index to keep track of which wiggle
    for mesh_n in range(len(mesh_tfr_label)):

        mesh_tfr_label_individual = mesh_tfr_label[mesh_n]  # individual mesh label from list
        mesh_time_label_individual = mesh_time_label[mesh_n]  # individual mesh label from list

        for index_element in df.index:

            if df[mesh_tfr_label_individual][index_element].ndim == 2:  # aka audio
                # Extract max/min x limit for each wiggle that will be plotted
                x_lim_min[index_wiggle_num_total] = np.min(df[mesh_time_label_individual][index_element])
                x_lim_max[index_wiggle_num_total] = np.max(df[mesh_time_label_individual][index_element])

                if common_colorbar is True:
                    # Extract max/min mesh tfr value for each wiggle that will be plotted
                    tfr_min[index_wiggle_num_total] = np.min(df[mesh_tfr_label_individual][index_element])
                    tfr_max[index_wiggle_num_total] = np.max(df[mesh_tfr_label_individual][index_element])

                index_wiggle_num_total += 1

            else:

                for index_dimension, _ in enumerate(df[mesh_tfr_label_individual][index_element]):
                    # Extract max/min x limit for each wiggle that will be plotted
                    x_lim_min[index_wiggle_num_total] = np.min(df[mesh_time_label_individual][index_element][index_dimension])
                    x_lim_max[index_wiggle_num_total] = np.max(df[mesh_time_label_individual][index_element][index_dimension])

                    if common_colorbar is True:
                        # Extract max/min mesh tfr value for each wiggle that will be plotted
                        tfr_min[index_wiggle_num_total] = np.min(df[mesh_tfr_label_individual][index_element][index_dimension])
                        tfr_max[index_wiggle_num_total] = np.max(df[mesh_tfr_label_individual][index_element][index_dimension])

                    index_wiggle_num_total += 1

    # Determine global min/max x limits
    x_lim_min_total = np.min(x_lim_min)
    x_lim_max_total = np.max(x_lim_max)

    if common_colorbar is True:  # Determine global min/max mesh tfr limits
        # global min/max limits tfr
        # tfr_min_total = np.min(tfr_min)
        tfr_max_total = np.max(tfr_max) - 3
        tfr_min_total = tfr_max_total - 18

    # start of figure
    fig = plt.figure(figsize=(figure_size_x, figure_size_y))
    if common_colorbar is True:  # for colorbar, two columns in fig
        gs = fig.add_gridspec(nrows=wiggle_num, ncols=2, figure=fig, width_ratios=[10., 0.1], wspace=0.03)
    else:
        gs = fig.add_gridspec(nrows=wiggle_num, ncols=1, figure=fig)

    # Start plotting each sensor/station
    index_wiggle_yticklabels = 0  # index to keep track of which wiggle y label to apply
    index_panel_order = wiggle_num - 1  # index to keep track of which wiggle is being plotted
    index_mesh_color_scale_panel = 0  # index to keep track of which mesh tfr color scale to apply if provided

    for mesh_n in range(len(mesh_tfr_label)):  # for each column label provided

        mesh_tfr_label_individual = mesh_tfr_label[mesh_n]  # individual mesh label from list
        mesh_time_label_individual = mesh_time_label[mesh_n]  # individual mesh label from list
        mesh_frequency_label_individual = mesh_frequency_label[mesh_n]  # individual mesh label from list

        # loop to plot column label provided per station, reversed to match plot_wiggles
        for _, index_signal in enumerate(reversed(df.index)):

            if df[mesh_tfr_label_individual][index_signal].ndim == 2:  # aka audio wiggle

                if common_colorbar is True:
                    ax = fig.add_subplot(gs[index_panel_order, 0])
                    plotted = ax.pcolormesh(df[mesh_time_label_individual][index_signal],
                                            df[mesh_frequency_label_individual][index_signal],
                                            df[mesh_tfr_label_individual][index_signal],
                                            vmin=tfr_min_total,
                                            vmax=tfr_max_total,
                                            cmap=color_map,
                                            edgecolor='face',
                                            shading="auto",
                                            snap=True)
                else:
                    # Color scaling calculation if colorbar False
                    if type(mesh_color_scaling) == str:
                        mesh_color_min, mesh_color_max = pnl.mesh_colormap_limits(df[mesh_tfr_label_individual][index_signal],
                                                                                  mesh_color_scaling,
                                                                                  mesh_color_range)
                    else:
                        mesh_color_min, mesh_color_max = pnl.mesh_colormap_limits(df[mesh_tfr_label_individual][index_signal],
                                                                                  mesh_color_scaling[index_mesh_color_scale_panel],
                                                                                  mesh_color_range[index_mesh_color_scale_panel])

                    ax = fig.add_subplot(gs[index_panel_order])
                    plotted = ax.pcolormesh(df[mesh_time_label_individual][index_signal],
                                            df[mesh_frequency_label_individual][index_signal],
                                            df[mesh_tfr_label_individual][index_signal],
                                            vmin=mesh_color_min,
                                            vmax=mesh_color_max,
                                            cmap=color_map,
                                            edgecolor='face',
                                            shading="auto",
                                            snap=True)

                # set ax limits
                plt.xlim(x_lim_min_total, x_lim_max_total)

                # This is a very useful bit of code
                _, _, frequency_fix_ymin, frequency_fix_ymax = \
                    pnl.mesh_time_frequency_edges(frequency=df[mesh_frequency_label_individual][index_signal],
                                                  time=df[mesh_time_label_individual][index_signal],
                                                  frequency_ymin=frequency_hz_ymin,
                                                  frequency_ymax=frequency_hz_ymax,
                                                  frequency_scaling=frequency_scaling)

                plt.ylim((frequency_fix_ymin, frequency_fix_ymax))

                # set ytick labels and y scale
                if frequency_scaling == "log":
                    ax.set_yscale("log", subs=None)
                    middle_point_diff = np.sqrt(frequency_fix_ymax*frequency_fix_ymin)
                    ax.minorticks_off()

                else:
                    middle_point_diff = (frequency_fix_ymax-frequency_fix_ymin)/2

                # Station Labels
                ax.set_yticks([middle_point_diff])  # set station label in the middle of the yaxis
                ax.set_yticklabels([wiggle_yticklabel[index_signal]], size=text_size)

                if index_panel_order < (wiggle_num - 1):  # plot x ticks for only last subplot
                    ax.set_xticks([])

                ax.tick_params(axis='both', which='major', labelsize=text_size)

                index_wiggle_yticklabels += 1
                index_panel_order -= 1
                index_mesh_color_scale_panel += 1

            else:
                # plot 3c sensors
                for index_dimension, _ in enumerate(df[mesh_tfr_label_individual][index_signal]):

                    if common_colorbar is True:
                        ax = fig.add_subplot(gs[index_panel_order, 0])
                        plotted = ax.pcolormesh(df[mesh_time_label_individual][index_signal][index_dimension],
                                                df[mesh_frequency_label_individual][index_signal][index_dimension],
                                                df[mesh_tfr_label_individual][index_signal][index_dimension],
                                                vmin=tfr_min_total,
                                                vmax=tfr_max_total,
                                                cmap=color_map,
                                                edgecolor='face',
                                                shading="auto",
                                                snap=True)
                    else:
                        # Color scaling calculation if colorbar False
                        if type(mesh_color_scaling) == str:
                            mesh_color_min, mesh_color_max = pnl.mesh_colormap_limits(df[mesh_tfr_label_individual][index_signal][index_dimension],
                                                                                      mesh_color_scaling,
                                                                                      mesh_color_range)
                        else:
                            mesh_color_min, mesh_color_max = pnl.mesh_colormap_limits(df[mesh_tfr_label_individual][index_signal][index_dimension],
                                                                                      mesh_color_scaling[index_mesh_color_scale_panel],
                                                                                      mesh_color_range[index_mesh_color_scale_panel])

                        ax = fig.add_subplot(gs[index_panel_order])
                        plotted = ax.pcolormesh(df[mesh_time_label_individual][index_signal][index_dimension],
                                                df[mesh_frequency_label_individual][index_signal][index_dimension],
                                                df[mesh_tfr_label_individual][index_signal][index_dimension],
                                                vmin=mesh_color_min,
                                                vmax=mesh_color_max,
                                                cmap=color_map,
                                                edgecolor='face',
                                                shading="auto",
                                                snap=True)

                    # set ax limits
                    plt.xlim(x_lim_min_total, x_lim_max_total)

                    # This is a very useful bit of code
                    _, _, frequency_fix_ymin, frequency_fix_ymax = \
                        pnl.mesh_time_frequency_edges(frequency=df[mesh_frequency_label_individual][index_signal][index_dimension],
                                                      time=df[mesh_time_label_individual][index_signal][index_dimension],
                                                      frequency_ymin=frequency_hz_ymin,
                                                      frequency_ymax=frequency_hz_ymax,
                                                      frequency_scaling=frequency_scaling)

                    plt.ylim((frequency_fix_ymin, frequency_fix_ymax))

                    # set ytick labels and y scale
                    if frequency_scaling == "log":
                        ax.set_yscale("log", subs=None)
                        middle_point_diff = np.sqrt(frequency_fix_ymax*frequency_fix_ymin)
                        ax.minorticks_off()

                    else:
                        middle_point_diff = (frequency_fix_ymax-frequency_fix_ymin)/2

                    # Station Labels
                    ax.set_yticks([middle_point_diff])  # set station label in the middle of the yaxis
                    ax.set_yticklabels([wiggle_yticklabel[index_wiggle_yticklabels]], size=text_size)

                    if index_panel_order < (wiggle_num - 1):  # plot x ticks for only last subplot
                        ax.set_xticks([])

                    ax.tick_params(axis='both', which='major', labelsize=text_size)

                    index_panel_order -= 1
                    index_mesh_color_scale_panel += 1
                    index_wiggle_yticklabels += 1

    # Find limits of axes subplot to create macro axes
    x0 = min([ax.get_position().x0 for ax in fig.axes])
    y0 = min([ax.get_position().y0 for ax in fig.axes])
    x1 = max([ax.get_position().x1 for ax in fig.axes])
    y1 = max([ax.get_position().y1 for ax in fig.axes])

    # Hide axes for common x and y labels
    plt.axes([x0, y0, x1 - x0, y1 - y0], frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    # Common x and y labels
    plt.xlabel("Time (s) relative to " + dt.datetime.utcfromtimestamp(t0_sig_epoch_s).strftime('%Y-%m-%d %H:%M:%S'),
               size=text_size, labelpad=10)
    if fig_title_show:
        plt.title(fig_title, size=text_size + 2, y=1.05)
        # Adjust overall plot to maximize figure space for press if title on
        if common_colorbar is False:
            plt.subplots_adjust(left=0.1, right=0.97)
        else:
            plt.subplots_adjust(top=0.92)

    else:
        # Adjust overall plot to maximize figure space for press if title off
        if common_colorbar is False:
            plt.subplots_adjust(left=0.1, top=0.95, right=0.97)
        else:
            plt.subplots_adjust(top=0.95)

    # Format colorbar
    if common_colorbar is True:
        cax = fig.add_subplot(gs[:, 1])
        mesh_panel_cbar: Colorbar = fig.colorbar(mappable=plotted, cax=cax)
        mesh_panel_cbar.ax.tick_params(labelsize=text_size-2)
        mesh_panel_cbar.set_label('bits relative to max', rotation=270, size=text_size, labelpad=25)

    if show_figure is True:
        plt.show()

    return fig

# TODO: docstring
# PLOT_WIGGLES AUXILIARY FUNCTIONS
def find_wiggle_num_yticks(df: pd.DataFrame,
                           sig_wf_label: Union[List[str], str] = "audio_wf",
                           sig_id_label: str = "station_id",
                           station_id_str: Optional[str] = None,
                           custom_yticks: Optional[Union[List[str], str]] = None) -> Tuple[int, List]:
    """
    Determine number of wiggles and ylabels that will be used

    :param df:
    :param sig_wf_label:
    :param sig_id_label:
    :param station_id_str:
    :param custom_yticks:
    :return:
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
                print(f'No data in column {sensor_in_list} for station {df[sig_id_label][index_n]}')
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
    Get time epoch origin for all sensors for all stations to establish the earliest timestamp if sig_timestamps_label provided

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
                    raise ValueError(f"the column name ({sensor_time_label}) was not found in dataframe")

                if type(df[sensor_time_label][index_station]) == float:  # not an array, so a Nan
                    continue  # skip cause entry for this station is empty

                else:
                    epoch_j.append(df[sensor_time_label][index_station].min())

    epoch_j = np.array(epoch_j)
    time_epoch_origin = np.min(epoch_j[np.nonzero(epoch_j)])
    return time_epoch_origin


def check_if_column_exists_in_df(df: pd.DataFrame,
                                 label: str):
    """
    Check if column is in dataframe.
    Based on the assumption that there is data in the column if it exists.

    :param df: input pandas dataframe
    :param label: string with column name

    :return: False if label not in df
    """
    return label in df.columns


def check_if_station_exists_in_df(df: pd.DataFrame,
                                  station_id_str: Union[str, None],
                                  sig_id_label: str = "station_id"):
    """
    Check if station is in dataframe.

    :param df: input pandas dataframe
    :param station_id_str: string with name of one station to plot only that station. Default is None
    :param sig_id_label: string for the station id column name in df. Default is "station_id"

    :return: False if label not in df
    """

    return station_id_str in df[sig_id_label].unique()


# PLOT_WIGGLES
# TODO: improve docstring
def plot_wiggles_pandas(df: pd.DataFrame,
                        sig_wf_label: Union[List[str], str] = "audio_wf",
                        sig_timestamps_label: Union[List[str], str] = "audio_epoch_s",
                        sig_id_label: str = "station_id",
                        station_id_str: Optional[str] = None,
                        x_label: str = "Time (s)",
                        y_label: str = "Signals",
                        fig_title_show: bool = True,
                        fig_title: str = 'Signals',
                        wf_color: str = 'midnightblue',
                        custom_yticks: Optional[Union[List[str], str]] = None,
                        show_figure: bool = True) -> Figure:

    """
    More nuanced plots with minimal distraction. Optimized for pandas input.
    Add signal timestamps to sig_timestamps_label for more accurate representation.
    For more information on available columns in dataframe, visit
    https://github.com/RedVoxInc/redpandas/blob/master/docs/redpandas/columns_name.md#redpandas-dataframe-columns

    :param df: input pandas data frame. REQUIRED
    :param sig_wf_label: single string or list of strings for the waveform column name in df. Default is "audio_wf". For example, for
        multiple sensor waveforms: sig_wf_label = ["audio_wf", "barometer_wf_highpass", "accelerometer_wf_highpass"]
    :param sig_timestamps_label: string or list of strings for column label in df with epoch time. Default is "audio_epoch_s". For example, for
        multiple sensor timestamps: sig_timestamps_label = ["audio_epoch_s", "barometer_epoch_s", "accelerometer_epoch_s"]
    :param sig_id_label: string for the station id column name in df. Default is "station_id"
    :param station_id_str: string with name of one station to plot only that station. Default is None
    :param x_label: optional x label. Default is "Time (s)"
    :param y_label: optional y label. Default is "Signals"
    :param fig_title_show: optional bool, include a title in the figure if True. Default is True
    :param fig_title: optional string, 'Normalized' + title label. Default is "signals"
    :param wf_color: optional string for waveform color. Default is midnightblue
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

    # set up figure
    fig, ax1 = plt.subplots(figsize=(figure_size_x, figure_size_y))
    ax1.set_yticks(wiggle_yticks)
    ax1.set_yticklabels(wiggle_yticklabel)
    ax1.set_ylim(wiggle_offset[0]-offset_scaling, wiggle_offset[-1]+offset_scaling)
    ax1.tick_params(axis='both', which='both', labelsize=text_size)

    # get t0 out of all the sensors for all stations
    time_epoch_origin = determine_time_epoch_origin(df=df,
                                                    sig_timestamps_label=sig_timestamps_label,
                                                    station_id_str=station_id_str,
                                                    sig_id_label=sig_id_label)
    # set up xlim min and max arrays.
    xlim_min = np.empty(wiggle_num)
    xlim_max = np.empty(wiggle_num)

    index_sensor_label_ticklabels_list = 0  # keep track of total sensor wf including x/y/z per station

    for index_station in df.index:  # loop per station
        for index_sensor_in_list, label in enumerate(sig_wf_label):  # loop per sensor

            # first things first, check if column with data exists and if there is data in it:
            if check_if_column_exists_in_df(df=df, label=label) is False or type(df[label][index_station]) == float:
                print(f"SensorMissingException: the column {label} is not in input DataFrame")
                continue  # if not, skip this iteration

            if station_id_str is None or df[sig_id_label][index_station].find(station_id_str) != -1:

                sensor_wf_df = df[label][index_station]
                sensor_timestamps_label = sig_timestamps_label[index_sensor_in_list]
                time_s = df[sensor_timestamps_label][index_station] - time_epoch_origin

                if sensor_wf_df.ndim == 1:  # sensor that is NOT acceleration/gyroscope/magnetometer

                    sig_j = df[label][index_station] / np.max(df[label][index_station])
                    ax1.plot(time_s, sig_j + wiggle_offset[index_sensor_label_ticklabels_list], color=wf_color)
                    xlim_min[index_sensor_label_ticklabels_list] = np.min(time_s)
                    xlim_max[index_sensor_label_ticklabels_list] = np.max(time_s)

                    index_sensor_label_ticklabels_list += 1

                else:  # sensor that is acceleration/gyroscope/magnetometer/barometer
                    for index_dimension, sensor_array in enumerate(sensor_wf_df):

                        sig_j = sensor_array / np.max(sensor_array)
                        ax1.plot(time_s, sig_j + wiggle_offset[index_sensor_label_ticklabels_list], color=wf_color)
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

    if y_label == "Signals":
        if station_id_str is None:
            ax1.set_ylabel(y_label, size=text_size)
        else:
            ax1.set_ylabel("Sensors", size=text_size)
    else:
        ax1.set_ylabel(y_label, size=text_size)

    if time_epoch_origin > 0:
        x_label += " relative to " + dt.datetime.utcfromtimestamp(time_epoch_origin).strftime('%Y-%m-%d %H:%M:%S')
    ax1.set_xlabel(x_label, size=text_size)
    fig.tight_layout()

    if show_figure is True:
        plt.show()

    return fig


def plot_psd_coh(psd_sig,
                 psd_ref,
                 coherence_sig_ref,
                 f_hz,
                 f_min_hz,
                 f_max_hz,
                 f_scale: str = "log",
                 sig_label: str = "PSD Sig",
                 ref_label: str = "PSD Ref",
                 psd_label: str = "PSD (bits)",
                 coh_label: str = "Coherence",
                 f_label: str = "Frequency (Hz)",
                 fig_title: str = "Power spectral density and coherence",
                 show_figure: bool = True) -> Figure:
    """
    Plot coherence and power spectral density

    :param psd_sig: Power spectral density of signal
    :param psd_ref: Power spectral density of reference signal
    :param coherence_sig_ref:  magnitude squared coherence of x and y
    :param f_hz: sample frequencies of PSD
    :param f_min_hz: minimum frequency to plot in Hz (x min limit)
    :param f_max_hz: maximum frequency to plot in Hz (x max limit)
    :param f_scale: scale of x axis. One of {"linear", "log", "symlog", "logit"}. Default is "log"
    :param sig_label: label for signal. Default is "PSD Sig"
    :param ref_label: label for reference. Default is "PSD Ref"
    :param psd_label: label for PSD. Default is "PSD (bits)"
    :param coh_label: label for coherence. Default is "Coherence"
    :param f_label: x axis label. Default is "Frequency (Hz)"
    :param fig_title: title of figure. Default is "Power spectral density and coherence"
    :param show_figure: show figure is True. Default is True

    :return: matplotlib figure instance
    """
    # Plot PSDs
    fig1 = plt.figure()
    fig1.set_size_inches(8, 6)
    plt.clf()
    ax1 = plt.subplot(211)
    ax1.plot(f_hz, psd_ref, 'r-', linewidth=2, label=ref_label)
    ax1.plot(f_hz, psd_sig, 'k-', label=sig_label)
    ax1.set_xscale(f_scale)
    ax1.legend()
    ax1.set_xlim([f_min_hz, f_max_hz])
    ax1.set_ylim([-16, 1])
    ax1.set_ylabel(psd_label)
    ax1.grid('on', which='both')
    ax1.set_title(fig_title)

    ax2 = plt.subplot(212)
    ax2.plot(f_hz, coherence_sig_ref, 'k-')
    ax2.set_xscale(f_scale)
    ax2.set_xlim([f_min_hz, f_max_hz])
    ax1.set_ylim([-16, 1])
    ax2.set_xlabel(f_label)
    ax2.set_ylabel(coh_label)
    ax2.grid('on', which='both')

    if show_figure is True:
        plt.show()

    return fig1


def plot_response_scatter(h_magnitude,
                          h_phase_deg,
                          color_guide,
                          f_hz,
                          f_min_hz,
                          f_max_hz,
                          f_scale: str = 'log',
                          fig_title: str = 'Response only valid at high coherence',
                          show_figure: bool = True) -> Figure:
    """
    Plot coherence response

    :param h_magnitude: magnitude, for example, absolute magnitude of response (which is power spectral density /
        cross-power spectral density)
    :param h_phase_deg: coherence phase degrees
    :param color_guide: parameters color guide, for example, magnitude squared coherence of x and y
    :param f_hz: frequency of coherence in Hz
    :param f_min_hz: minimum frequency to plot in Hz (x min limit)
    :param f_max_hz: maximum frequency to plot in Hz (x max limit)
    :param f_scale: scale of x axis. One of {"linear", "log", "symlog", "logit"}. Default is "log"
    :param fig_title: title of figure
    :param show_figure: show figure is True. Default is True

    :return: matplotlib figure instance
    """
    # plot magnitude and coherence
    fig = plt.figure()
    fig.set_size_inches(8, 6)
    ax1 = plt.subplot(211)
    im1 = ax1.scatter(x=f_hz, y=h_magnitude, c=color_guide, marker='o')
    ax1.set_xscale(f_scale)
    ax1.set_xlim([f_min_hz, f_max_hz])
    ax1.grid('on', which='both')
    hc = fig.colorbar(im1)
    hc.set_label('Coherence')
    ax1.set_ylabel('Magnitude ')
    ax1.set_title(fig_title)

    ax2 = plt.subplot(212)
    im2 = ax2.scatter(x=f_hz, y=h_phase_deg, c=color_guide, marker='o')
    ax2.set_xscale(f_scale)
    ax2.set_xlim([f_min_hz, f_max_hz])
    ax2.grid('on', which='both')

    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('Phase [deg]')
    hc = plt.colorbar(im2)
    hc.set_label('Coherence')

    if show_figure is True:
        plt.show()

    return fig
