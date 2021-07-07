"""
This module contains main utils for plotting RedPandas DataFrames.

Last updated: 6 July 2021
"""
import datetime as dt
from typing import List, Union, Optional

import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
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
                     t0_sig_epoch_s: float,
                     sig_id_label: Union[str, List[str]],
                     fig_title_show: bool = True,
                     fig_title: str = "STFT",
                     frequency_scaling: str = "log",
                     frequency_hz_ymin: float = rpd_scales.Slice.FU,
                     frequency_hz_ymax: float = rpd_scales.Slice.F0,
                     common_colorbar: bool = True,
                     mesh_color_scaling: Union[List[str], str] = 'auto',
                     mesh_color_range: Union[List[float], float] = 15) -> None:

    """
     Plots spectrogram for all signals in df

     :param df: input pandas data frame
     :param mesh_time_label: string for the mesh time column name in df. List of strings for multiple
     :param mesh_frequency_label: string for the mesh frequency column name in df. List of strings for multiple
     :param mesh_tfr_label: string for the mesh tfr column name in df. List of strings for multiple
     :param t0_sig_epoch_s: epoch time in seconds of first timestamp
     :param sig_id_label: string for column name with station ids in df. You can also provide a list of custom labels
     :param fig_title_show: include a title in the figure. Default is True
     :param fig_title: figure title label
     :param frequency_scaling: "log" or "lin". Default is "log"
     :param frequency_hz_ymin: y axis min lim
     :param frequency_hz_ymax: y axis max lim
     :param common_colorbar: display a colorbar for all mesh panels. Default is True
     :param mesh_color_scaling: colorbar scaling, "auto" or "range". Default is 'auto'
     :param mesh_color_range: Default is 15
     :return: plot
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


def plot_wiggles_pandas(df: pd.DataFrame,
                        sig_wf_label: Union[List[str], str],
                        sig_sample_rate_label: Union[List[str], str],
                        sig_id_label: Union[List[str], str],
                        station_id_str: Optional[str] = None,
                        x_label: str = "Time (s)",
                        y_label: str = "Signals",
                        fig_title_show: bool = True,
                        fig_title: str = 'Signals',
                        wf_color: str = 'midnightblue',
                        sig_timestamps_label: Optional[Union[List[str], str]] = None,
                        custom_yticks: Optional[Union[List[str], str]] = None) -> None:
    """
    More nuanced plots with minimal distraction. Optimized for pandas input.
    Add signal timestamps to sig_timestamps_label for more accurate representation.

    :param df: input pandas data frame
    :param sig_wf_label: single string or list of strings for the waveform column name in df
    :param sig_sample_rate_label: single string or list of strings for the sample rate in Hz column name in df
    :param sig_id_label: string for the station id column name in df
    :param station_id_str: string with name of one station to plot only that station. Default is None
    :param x_label: x label. Default is "Time (s)"
    :param y_label: y label. Default is "Signals"
    :param fig_title_show: include a title in the figure. Default is True
    :param fig_title: 'Normalized' + title label
    :param wf_color: waveform color. Default is midnightblue
    :param sig_timestamps_label: string or list of strings for column label in df with epoch time, default = None
    :param custom_yticks: provide custom names for yticks, list of strings (one label per channel component) or "index"
    :return: plot
    """

    # Create List of signal channels to loop through later
    # If given only one, aka a sting, make it a list of length 1
    if type(sig_wf_label) == str:
        sig_wf_label = [sig_wf_label]
    if type(sig_sample_rate_label) == str:
        sig_sample_rate_label = [sig_sample_rate_label]
    if type(sig_timestamps_label) == str:
        sig_timestamps_label = [sig_timestamps_label]

    # First, determine number of wiggles and ylabels that will be used
    wiggle_num_list = []  # number of wiggles
    wiggle_yticklabel = []  # name/y label of wiggles

    for index_sensor_in_list, sensor_in_list in enumerate(sig_wf_label):

        for index_n in df.index:
            if station_id_str is None or df[sig_id_label][index_n].find(station_id_str) != -1:

                if df[sensor_in_list][index_n].ndim == 1:  # aka audio
                    wiggle_num_list.append(1)  # append 1 wiggle cause it will only be one panel

                    # Establish ylabel for wiggle
                    if custom_yticks == "index":  # if ylabel for wiggle is index station
                        wiggle_yticklabel.append(df.index[index_n])

                    elif custom_yticks is None:
                        wiggle_yticklabel.append(df[sig_id_label][index_n])  # if ylabel for wiggles is custom list

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
                            wiggle_yticklabel.append(df[sig_id_label][index_n])  # if ylabel for wiggles is custom list

    # # if custom_yticks provided, make that the yticks
    if custom_yticks is not None and custom_yticks != "index":
        wiggle_yticklabel = custom_yticks

    wiggle_num = sum(wiggle_num_list)  # total number of signal that will be displayed

    # Make sure wiggle_num and # of ylabels match to avoid problems later on
    if len(wiggle_yticklabel) != wiggle_num:
        print('ERROR: The number of labels provided in the custom_yticks parameter does not match the number of signal '
              'channels provided in sig_wf_label or the number of stations in dataframe')
        print('Do not forget that accelerometer, gyroscope, and magnetometer have X, Y and Z components so a label is '
              'required for each component.')
        print('In case you provided a str in station_id_str, make sure the str actually exists in the dataframe')
        exit()

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

    xlim_min = np.empty(wiggle_num)
    xlim_max = np.empty(wiggle_num)

    # Establish min xlim aka min time
    if sig_timestamps_label is not None:
        epoch_j = []
        for index_station in df.index:

            if station_id_str is None or df[sig_id_label][index_station].find(station_id_str) != -1:

                for index_time_label, sensor_time_label in enumerate(sig_timestamps_label):

                    epoch_j.append(df[sensor_time_label][index_station].min())

        epoch_j = np.array(epoch_j)
        time_epoch_origin = np.min(epoch_j[np.nonzero(epoch_j)])

    else:
        time_epoch_origin = 0.

    index_sensor_label_ticklabels_list = 0  # keep track of total sensor wf including x/y/z per station
    for index_station in df.index:  # loop per station
        for index_sensor_in_list, label in enumerate(sig_wf_label):  # loop per sensor

            if station_id_str is None or df[sig_id_label][index_station].find(station_id_str) != -1:
                sensor_wf_df = df[label][index_station]

                if sig_timestamps_label is not None:
                    sensor_timestamps_label = sig_timestamps_label[index_sensor_in_list]
                    time_s = df[sensor_timestamps_label][index_station] - time_epoch_origin

                if sensor_wf_df.ndim == 1:  # sensor that is NOT acceleration/gyroscope/magnetometer

                    if sig_timestamps_label is None:
                        channel_sample_rate = sig_sample_rate_label[index_sensor_in_list]
                        if df[label][index_station][0] == df[label][index_station][1]:
                            time_s = np.arange(len(df[label][index_station])) / df[channel_sample_rate][index_station]
                        else:
                            time_s = np.arange(len(df[label][index_station])) / df[channel_sample_rate][index_station]

                    sig_j = df[label][index_station] / np.max(df[label][index_station])
                    ax1.plot(time_s, sig_j + wiggle_offset[index_sensor_label_ticklabels_list], color=wf_color)
                    xlim_min[index_sensor_label_ticklabels_list] = np.min(time_s)
                    xlim_max[index_sensor_label_ticklabels_list] = np.max(time_s)

                    index_sensor_label_ticklabels_list += 1

                else:

                    if sig_timestamps_label is None:
                        channel_sample_rate = sig_sample_rate_label[index_sensor_in_list]
                        if df[label][index_station][0][0] == df[label][index_station][0][1]:

                            time_s = np.arange(len(df[label][index_station][0])) / df[channel_sample_rate][index_station]
                        else:
                            time_s = np.arange(len(df[label][index_station][0])) / df[channel_sample_rate][index_station]

                    for index_dimension, sensor_array in enumerate(sensor_wf_df):

                        sig_j = sensor_array / np.max(sensor_array)
                        ax1.plot(time_s, sig_j + wiggle_offset[index_sensor_label_ticklabels_list], color=wf_color)
                        xlim_min[index_sensor_label_ticklabels_list] = np.min(time_s)
                        xlim_max[index_sensor_label_ticklabels_list] = np.max(time_s)

                        index_sensor_label_ticklabels_list += 1

    ax1.set_xlim(np.min(xlim_min), np.max(xlim_max))
    ax1.grid(True)
    if fig_title_show:
        ax1.set_title('Normalized ' + fig_title, size=text_size)
    ax1.set_ylabel(y_label, size=text_size)
    if time_epoch_origin > 0:
        x_label += " relative to " + dt.datetime.utcfromtimestamp(time_epoch_origin).strftime('%Y-%m-%d %H:%M:%S')
    ax1.set_xlabel(x_label, size=text_size)
    fig.tight_layout()


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
                 fig_title: str = "Power spectral density and coherence") -> None:
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
    :return: plot
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


def plot_response_scatter(h_magnitude,
                          h_phase_deg,
                          color_guide,
                          f_hz,
                          f_min_hz,
                          f_max_hz,
                          f_scale: str = 'log',
                          fig_title: str = 'Response only valid at high coherence') -> None:
    """
    Plot coherence response

    :param h_magnitude: magnitude, for example, absolute magnitude of response (which is power spectral density / cross-power spectral density)
    :param h_phase_deg: coherence phase degrees
    :param color_guide: parameters color guide, for example, magnitude squared coherence of x and y
    :param f_hz: frequency of coherence in Hz
    :param f_min_hz: minimum frequency to plot in Hz (x min limit)
    :param f_max_hz: maximum frequency to plot in Hz (x max limit)
    :param f_scale: scale of x axis. One of {"linear", "log", "symlog", "logit"}. Default is "log"
    :param fig_title: title of figure
    :return: plot
    """
    # plot magnitude and coherence
    fig = plt.figure()
    fig.set_size_inches(8, 6)
    ax1 = plt.subplot(211)
    # plt.hlines(1,1e-02,2e+01,linestyle='dashed',color='darkgrey')
    # plt.vlines([0.02,10],1e-2,2,'r','dashed')
    # h31=ax31.scatter(f, mag, 100, Cxy, '.', edgecolor='', cmap=CM, zorder=5)
    im1 = ax1.scatter(x=f_hz, y=h_magnitude, c=color_guide, marker='o')
    ax1.set_xscale(f_scale)
    ax1.set_xlim([f_min_hz, f_max_hz])
    ax1.grid('on', which='both')
    # h31.set_clim(CLIM)
    hc = fig.colorbar(im1)
    hc.set_label('Coherence')
    ax1.set_ylabel('Magnitude ')
    ax1.set_title(fig_title)

    ax2 = plt.subplot(212)
    # plt.hlines(0,1e-02,2e+01,linestyle='dashed',color='darkgrey')
    # plt.vlines([0.02,10],-180,180,'r','dashed')
    # h32=ax32.scatter(f, ph, 100, Cxy, '.', edgecolor='', cmap=CM, zorder=5)
    im2 = ax2.scatter(x=f_hz, y=h_phase_deg, c=color_guide, marker='o')
    ax2.set_xscale(f_scale)
    ax2.set_xlim([f_min_hz, f_max_hz])
    ax2.grid('on', which='both')

    # ax32.axis([.01, 20, -15, 15])
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('Phase [deg]')
    # ax32.grid('on',which='both')
    # h32.set_clim(CLIM)
    hc = plt.colorbar(im2)
    hc.set_label('Coherence')
