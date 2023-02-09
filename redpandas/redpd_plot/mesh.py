"""
Plot TFR
"""
import datetime as dt
from typing import List, Union

import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from libquantum.plot_templates import plot_time_frequency_reps as pnl

import redpandas.redpd_scales as rpd_scales
from redpandas.redpd_plot.parameters import FigureParameters as FigParam


def find_wiggle_num_tfr(df: pd.DataFrame,
                        mesh_tfr_label: Union[str, List[str]]) -> int:
    """
    Find number of signals to plot

    :param df: input pandas data frame
    :param mesh_tfr_label: string for the mesh tfr column name in df. List of strings for multiple columns
    :return: int, number of signals
    """
    # Determine overall number of mesh panels in fig
    wiggle_num_list = []  # number of wiggles
    for mesh_n in range(len(mesh_tfr_label)):
        mesh_tfr_label_individual = mesh_tfr_label[mesh_n]  # individual mesh label from list

        for n in df.index:
            # check column exists and not empty
            if mesh_tfr_label_individual in df.columns and type(df[mesh_tfr_label_individual][n]) != float:
                if df[mesh_tfr_label_individual][n].ndim == 2:  # aka audio
                    wiggle_num_list.append(1)  # append 1 wiggle cause it will only be one tfr panel
                else:
                    # Check if barometer, cause then only 1 wiggle
                    if mesh_tfr_label_individual.find("pressure") == 0 or mesh_tfr_label_individual.find("bar") == 0:
                        wiggle_num_list.append(1)
                    else:  # if not barometer, its is a 3c sensors aka gyroscope/accelerometer/magnetometer
                        wiggle_num_list.append(3)
            else:
                continue

    wiggle_num = sum(wiggle_num_list)  # total number of signal that will be displayed

    return wiggle_num


def find_ylabel_tfr(df: pd.DataFrame,
                    mesh_tfr_label: Union[str, List[str]],
                    sig_id_label: Union[str, List[str]]) -> List:
    """
    Find ylabels

    :param df: input pandas data frame
    :param mesh_tfr_label: string for the mesh tfr column name in df. List of strings for multiple columns
    :param sig_id_label: string for column name with station ids in df. Alternatively, you can also provide a
        list of strings with custom labels, for example: ["Audio", "Acc X", "Acc Y", "Acc Z"]. The number of custom labels
         provided needs to match the number of stations and signals in df.

    :return: list of strings with y labels
    """

    if sig_id_label == "index" or (type(sig_id_label) == str and sig_id_label in df.columns):
        wiggle_yticklabel = []  # name/y label of wiggles

        for mesh_n in range(len(mesh_tfr_label)):
            mesh_tfr_label_individual = mesh_tfr_label[mesh_n]  # individual mesh label from list
            for n in df.index:
                # check column exists and not empty
                if mesh_tfr_label_individual in df.columns and type(df[mesh_tfr_label_individual][n]) != float:
                    if df[mesh_tfr_label_individual][n].ndim == 2:  # aka audio
                        # Establish ylabel for wiggle
                        if sig_id_label == "index":  # if ylabel for wiggle is index station
                            wiggle_yticklabel.append(df.index[n])
                        else:
                            wiggle_yticklabel.append(df[sig_id_label][n])  # if ylabel for wiggles is custom list
                    else:
                        for index_dimension, _ in enumerate(df[mesh_tfr_label_individual][n]):
                            # Establish ylabel for wiggle
                            if sig_id_label == "index":  # if ylabel for wiggle is index station
                                wiggle_yticklabel.append(df.index[n])
                            else:
                                wiggle_yticklabel.append(df[sig_id_label][n])  # if ylabel for wiggles is custom list
                else:
                    continue
    else:
        wiggle_yticklabel = sig_id_label

    return wiggle_yticklabel


def find_x_max_min_lim(df: pd.DataFrame,
                       wiggle_num: int,
                       mesh_tfr_label: Union[str, List[str]],
                       mesh_time_label: Union[str, List[str]]):
    """
    Find max/min x-axis limit

    :param df: input pandas data frame
    :param wiggle_num: int, number of signals that will be plotted
    :param mesh_tfr_label: string for the mesh tfr column name in df. List of strings for multiple columns
    :param mesh_time_label: string for the mesh time column name in df. List of strings for multiple columns

    :return: values for x max and x min
    """
    # loop to find xlim and tfr global max/min
    x_lim_min = np.empty(wiggle_num)
    x_lim_max = np.empty(wiggle_num)

    index_wiggle_num_total = 0  # index to keep track of which wiggle
    for mesh_n in range(len(mesh_tfr_label)):
        for index_element in df.index:
            mesh_tfr_label_individual = mesh_tfr_label[mesh_n]  # individual mesh label from list
            mesh_time_label_individual = mesh_time_label[mesh_n]  # individual mesh label from list

            # check column exists and not empty
            if mesh_tfr_label_individual in df.columns and type(df[mesh_tfr_label_individual][index_element]) != float:

                if df[mesh_tfr_label_individual][index_element].ndim == 2:  # aka audio
                    # Extract max/min x limit for each wiggle that will be plotted
                    x_lim_min[index_wiggle_num_total] = np.min(df[mesh_time_label_individual][index_element])
                    x_lim_max[index_wiggle_num_total] = np.max(df[mesh_time_label_individual][index_element])

                    index_wiggle_num_total += 1

                else:

                    for index_dimension, _ in enumerate(df[mesh_tfr_label_individual][index_element]):
                        # Extract max/min x limit for each wiggle that will be plotted
                        x_lim_min[index_wiggle_num_total] = np.min(df[mesh_time_label_individual][index_element][index_dimension])
                        x_lim_max[index_wiggle_num_total] = np.max(df[mesh_time_label_individual][index_element][index_dimension])

                        index_wiggle_num_total += 1
            else:
                continue

    # Determine global min/max x limits
    x_lim_min_total = np.min(x_lim_min)
    x_lim_max_total = np.max(x_lim_max)

    return x_lim_max_total, x_lim_min_total


def find_tfr_max_min_lim(df: pd.DataFrame,
                         wiggle_num: int,
                         mesh_tfr_label: Union[str, List[str]]):
    """
    Find max/min limits TFR bits

    :param df: input pandas data frame
    :param wiggle_num: int, number of signals that will be plotted
    :param mesh_tfr_label: string for the mesh tfr column name in df. List of strings for multiple columns
    :return: values for tfr max and tfr min
    """

    # loop to find xlim and tfr global max/min
    tfr_min = np.empty(wiggle_num)
    tfr_max = np.empty(wiggle_num)

    index_wiggle_num_total = 0  # index to keep track of which wiggle

    # for mesh_n in range(len(mesh_tfr_label)):
    for index_element in df.index:
        for mesh_n in range(len(mesh_tfr_label)):
            mesh_tfr_label_individual = mesh_tfr_label[mesh_n]  # individual mesh label from list

            # check column exists and not empty
            if mesh_tfr_label_individual in df.columns and type(df[mesh_tfr_label_individual][index_element]) != float:

                if df[mesh_tfr_label_individual][index_element].ndim == 2:  # aka audio
                    # Extract max/min mesh tfr value for each wiggle that will be plotted
                    tfr_min[index_wiggle_num_total] = np.min(df[mesh_tfr_label_individual][index_element])
                    tfr_max[index_wiggle_num_total] = np.max(df[mesh_tfr_label_individual][index_element])

                    index_wiggle_num_total += 1

                else:

                    for index_dimension, _ in enumerate(df[mesh_tfr_label_individual][index_element]):
                        # Extract max/min mesh tfr value for each wiggle that will be plotted
                        tfr_min[index_wiggle_num_total] = np.min(df[mesh_tfr_label_individual][index_element][index_dimension])
                        tfr_max[index_wiggle_num_total] = np.max(df[mesh_tfr_label_individual][index_element][index_dimension])

                        index_wiggle_num_total += 1
            else:
                continue

    # global min/max limits tfr
    # tfr_min_total = np.min(tfr_min)
    tfr_max_total = np.max(tfr_max) - 3
    tfr_min_total = tfr_max_total - 18

    return tfr_max_total, tfr_min_total


def find_mesh_color_and_scaling(wiggle_num: int,
                                mesh_color_scaling: Union[List[str], str] = 'auto',
                                mesh_color_range: Union[List[float], float] = 15.0):
    """
    Find number of mesh color and scaling values if input is only int and string. Coded to avoid retyping same values
    over and over in cases where stations have only one type of sensor.

    :param wiggle_num: number of wiggles in df
    :param mesh_color_scaling: optional, colorbar scaling, "auto" or "range". Default is 'auto'. The parameter common_colorbar
        needs to be set to False to apply mesh_color_scaling
    :param mesh_color_range: optional, range of colorbar. Default is 15. The parameter common_colorbar needs to be set
        to False and mesh_color_scaling to "range "to apply mesh_color_range
    :return:
    """

    if (isinstance(mesh_color_range, float) is True or isinstance(mesh_color_range, int) is True) and isinstance(mesh_color_scaling, str):
        list_mesh_color_range = [mesh_color_range] * wiggle_num
        list_mesh_color_scaling = [mesh_color_scaling] * wiggle_num
        return list_mesh_color_scaling, list_mesh_color_range

    else:
        return mesh_color_scaling, mesh_color_range


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
                     common_colorbar: bool = False,
                     ytick_values_show: bool = True,
                     mesh_color_scaling: Union[List[str], str] = 'auto',
                     mesh_color_range: Union[List[float], float] = 15.0,
                     show_figure: bool = True) -> Figure:

    """
     Plots spectrogram for all signals in df

     :param df: input pandas data frame. REQUIRED
     :param mesh_time_label: string for the mesh time column name in df. List of strings for multiple columns. REQUIRED
     :param mesh_frequency_label: string for the mesh frequency column name in df. List of strings for multiple columns. REQUIRED
     :param mesh_tfr_label: string for the mesh tfr column name in df. List of strings for multiple columns. REQUIRED
     :param sig_id_label: string for column name with station ids in df. REQUIRED. Alternatively, you can also provide a
        list of strings with custom labels, for example: ["Audio", "Acc X", "Acc Y", "Acc Z"]. The number of custom labels
         provided needs to match the number of stations and signals in df.
     :param t0_sig_epoch_s: optional float, epoch time in seconds of first timestamp. Default is None
     :param fig_title_show: optional bool, include a title in the figure if True. Default is True
     :param fig_title: optional string, figure title label. Default is "STFT"
     :param frequency_scaling: optional string, determine frequency scaling "log" or "lin". Default is "log"
     :param frequency_hz_ymin: optional float, y axis min lim
     :param frequency_hz_ymax: optional float, y axis max lim
     :param common_colorbar: optional bool, display a colorbar for all mesh panels if True. Default is False
     :param ytick_values_show: optional bool, display ytick values. Default is False
     :param mesh_color_scaling: optional, colorbar scaling, "auto" or "range". Default is 'auto'. The parameter common_colorbar
        needs to be set to False to apply mesh_color_scaling
     :param mesh_color_range: optional, range of colorbar. Default is 15.0. The parameter common_colorbar needs to be set
        to False and mesh_color_scaling to "range" to apply mesh_color_range
     :param show_figure: optional bool, show figure if True. Default is True

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
    if len(mesh_tfr_label) != len(mesh_time_label) \
            or len(mesh_tfr_label) != len(mesh_frequency_label) \
            or len(mesh_time_label) != len(mesh_frequency_label):
        raise ValueError("mesh_time_label, mesh_tfr_label, "
                         "or mesh_frequency_label do not have the same length. Please check.")

    # Get wiggle number, y labels
    wiggle_num = find_wiggle_num_tfr(df=df,
                                     mesh_tfr_label=mesh_tfr_label)
    wiggle_yticklabel = find_ylabel_tfr(df=df,
                                        mesh_tfr_label=mesh_tfr_label,
                                        sig_id_label=sig_id_label)

    # Check wiggle_num and # of ylabels match
    if len(wiggle_yticklabel) != wiggle_num:
        raise ValueError(f"The number of labels provided in the custom_yticks parameter ({len(wiggle_yticklabel)}) "
                         f"does not match the number of signal channels provided in sig_wf_label "
                         f"or the number of stations in dataframe ({wiggle_num})."
                         f"\nDo not forget that accelerometer, gyroscope, and magnetometer have X, Y and Z components "
                         f"so a label is required for each component.")
    # Get x limits
    x_lim_max_total, x_lim_min_total = find_x_max_min_lim(df=df,
                                                          wiggle_num=wiggle_num,
                                                          mesh_tfr_label=mesh_tfr_label,
                                                          mesh_time_label=mesh_time_label)

    # Get global min/max mesh tfr limits if colorbar option on
    if common_colorbar is True:
        tfr_max_total, tfr_min_total = find_tfr_max_min_lim(df=df,
                                                            wiggle_num=wiggle_num,
                                                            mesh_tfr_label=mesh_tfr_label)
    else:
        mesh_color_scaling, mesh_color_range = find_mesh_color_and_scaling(wiggle_num=wiggle_num,
                                                                           mesh_color_scaling=mesh_color_scaling,
                                                                           mesh_color_range=mesh_color_range)

        # Check wiggle_num and values provided for mesh scaling match (if common_colorbar is False)
        if len(mesh_color_scaling) != wiggle_num:
            raise ValueError(f"The number of strings provided in the mesh_color_scaling "
                             f"parameter({len(mesh_color_scaling)}) does not match the number of signal channels "
                             f"provided in sig_wf_label or the number of stations in dataframe ({wiggle_num})."
                             f"\nDo not forget that accelerometer, gyroscope, and magnetometer have X, Y and Z "
                             f"components so a string is required for each component.")

        if len(mesh_color_range) != wiggle_num:
            raise ValueError(f"The number of values provided in the mesh_color_range parameter({len(mesh_color_range)})"
                             f"does not match the number of signal channels provided in sig_wf_label or the number of "
                             f"stations in dataframe ({wiggle_num})."
                             f"\nDo not forget that accelerometer, gyroscope, and magnetometer have X, Y and Z "
                             f"components so a value is required for each component.")

    # Set up figure
    if show_figure:
        fig = plt.figure(figsize=(FigParam().figure_size_x, FigParam().figure_size_y))
    else:
        fig: Figure = Figure(figsize=(FigParam().figure_size_x, FigParam().figure_size_y))

    if common_colorbar is True and ytick_values_show is False:  # for colorbar, two columns in fig
        gs = fig.add_gridspec(nrows=wiggle_num, ncols=2, figure=fig, width_ratios=[10., 0.1], wspace=0.03)
    elif common_colorbar is True and ytick_values_show is True:
        gs = fig.add_gridspec(nrows=wiggle_num, ncols=2, figure=fig, width_ratios=[10., 0.1], wspace=0.15)
    else:
        gs = fig.add_gridspec(nrows=wiggle_num, ncols=1, figure=fig)

    index_wiggle_yticklabels = 0  # index to keep track of which wiggle y label to apply
    index_panel_order = wiggle_num - 1  # index to keep track of which wiggle is being plotted, -1 cause index 0
    index_mesh_color_scale_panel = 0  # index to keep track of which mesh tfr color scale to apply if provided

    # loop to plot column label provided per station, reversed to match plot_wiggles
    for mesh_n in range(len(mesh_tfr_label)):  # for each column label provided
        for index_signal in df.index:
            # Start plotting each sensor/station
            mesh_tfr_label_individual = mesh_tfr_label[mesh_n]  # individual mesh label from list
            mesh_time_label_individual = mesh_time_label[mesh_n]  # individual mesh label from list
            mesh_frequency_label_individual = mesh_frequency_label[mesh_n]  # individual mesh label from list

            # check column exists and not empty
            if mesh_tfr_label_individual in df.columns and type(df[mesh_tfr_label_individual][index_signal]) != float:
                if df[mesh_tfr_label_individual][index_signal].ndim == 2:  # aka audio wiggle

                    if common_colorbar is True:
                        ax = fig.add_subplot(gs[index_panel_order, 0])
                        plotted = ax.pcolormesh(df[mesh_time_label_individual][index_signal],
                                                df[mesh_frequency_label_individual][index_signal],
                                                df[mesh_tfr_label_individual][index_signal],
                                                vmin=tfr_min_total,
                                                vmax=tfr_max_total,
                                                cmap=FigParam().color_map,
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
                                                cmap=FigParam().color_map,
                                                edgecolor='face',
                                                shading="auto",
                                                snap=True)

                    # This is a very useful bit of code
                    _, _, frequency_fix_ymin, frequency_fix_ymax = \
                        pnl.mesh_time_frequency_edges(frequency=df[mesh_frequency_label_individual][index_signal],
                                                      time=df[mesh_time_label_individual][index_signal],
                                                      frequency_ymin=frequency_hz_ymin,
                                                      frequency_ymax=frequency_hz_ymax,
                                                      frequency_scaling=frequency_scaling)

                    ax.set_ylim(frequency_fix_ymin, frequency_fix_ymax)

                    # Set up ytick labels and y scale
                    if frequency_scaling == "log":
                        ax.set_yscale("log", subs=None)
                        middle_point_diff = np.sqrt(frequency_fix_ymax*frequency_fix_ymin)
                        # ax.minorticks_off()

                    else:
                        middle_point_diff = (frequency_fix_ymax-frequency_fix_ymin)/2

                    # Plot yticks
                    if ytick_values_show is True:
                        # Plot primary ticks with y values
                        ax.yaxis.tick_right()
                        ax.yaxis.set_label_position("right")
                        ax.set_ylabel('Hz', size=FigParam().text_size)
                        if wiggle_num > 3:
                            ax.minorticks_off()

                        # Plot secondary ticks with name station
                        secax = ax.secondary_yaxis("left")
                        secax.set_yticks([middle_point_diff])  # set station label in the middle of the yaxis
                        secax.set_yticklabels([wiggle_yticklabel[index_wiggle_yticklabels]], size=FigParam().text_size)
                        secax.minorticks_off()
                        ax.tick_params(axis='y', which='major', labelsize=FigParam().text_size)

                    else:
                        # Plot yticks only name station
                        ax.set_yticks([middle_point_diff])  # set station label in the middle of the yaxis
                        ax.set_yticklabels([wiggle_yticklabel[index_wiggle_yticklabels]], size=FigParam().text_size)
                        ax.tick_params(axis='y', which='major', labelsize=FigParam().text_size)

                    # Set up ax limits
                    ax.set_xlim(x_lim_min_total, x_lim_max_total)
                    ax.tick_params(axis='x', which='major', labelsize=FigParam().text_size)
                    if index_panel_order < (wiggle_num - 1):  # plot x ticks for only last subplot
                        ax.set_xticks([])

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
                                                    cmap=FigParam().color_map,
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
                                                    cmap=FigParam().color_map,
                                                    edgecolor='face',
                                                    shading="auto",
                                                    snap=True)

                        # This is a very useful bit of code
                        _, _, frequency_fix_ymin, frequency_fix_ymax = \
                            pnl.mesh_time_frequency_edges(frequency=df[mesh_frequency_label_individual][index_signal][index_dimension],
                                                          time=df[mesh_time_label_individual][index_signal][index_dimension],
                                                          frequency_ymin=frequency_hz_ymin,
                                                          frequency_ymax=frequency_hz_ymax,
                                                          frequency_scaling=frequency_scaling)

                        ax.set_ylim(frequency_fix_ymin, frequency_fix_ymax)

                        # Set up ytick labels and y scale
                        if frequency_scaling == "log":
                            ax.set_yscale("log", subs=None)
                            middle_point_diff = np.sqrt(frequency_fix_ymax*frequency_fix_ymin)
                            ax.minorticks_off()

                        else:
                            middle_point_diff = (frequency_fix_ymax-frequency_fix_ymin)/2
                            ax.minorticks_off()

                        # Plot yticks
                        if ytick_values_show is True:
                            # Plot primary ticks with y values
                            ax.yaxis.tick_right()
                            ax.yaxis.set_label_position("right")
                            ax.set_ylabel('Hz', size=FigParam().text_size)
                            if wiggle_num > 3:
                                ax.minorticks_off()

                            # Plot secondary ticks with name station
                            secax = ax.secondary_yaxis("left")
                            secax.set_yticks([middle_point_diff])  # set station label in the middle of the yaxis
                            secax.set_yticklabels([wiggle_yticklabel[index_wiggle_yticklabels]], size=FigParam().text_size)
                            secax.minorticks_off()
                            ax.tick_params(axis='y', which='both', labelsize=FigParam().text_size)

                        else:
                            # Plot yticks only name station
                            ax.set_yticks([middle_point_diff])  # set station label in the middle of the yaxis
                            ax.set_yticklabels([wiggle_yticklabel[index_wiggle_yticklabels]], size=FigParam().text_size)
                            ax.tick_params(axis='y', which='major', labelsize=FigParam().text_size)

                        # Set up ax limits
                        ax.set_xlim(x_lim_min_total, x_lim_max_total)
                        ax.tick_params(axis='x', which='major', labelsize=FigParam().text_size)

                        if index_panel_order < (wiggle_num - 1):  # plot x ticks for only last subplot
                            ax.set_xticks([])

                        index_panel_order -= 1
                        index_mesh_color_scale_panel += 1
                        index_wiggle_yticklabels += 1
            else:
                continue

    # Find limits of axes subplot to create macro axes
    x0 = min([ax.get_position().x0 for ax in fig.axes])
    y0 = min([ax.get_position().y0 for ax in fig.axes])
    x1 = max([ax.get_position().x1 for ax in fig.axes])
    y1 = max([ax.get_position().y1 for ax in fig.axes])

    # Hide axes for common x and y labels
    axes = fig.add_axes([x0, y0, x1 - x0, y1 - y0], frameon=False)
    axes.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    # if ytick_values_show is True:
    #     plt.text(1.02, 1.085, "Hz", fontsize=FigParam().text_size_minor_yaxis, transform=ax.transAxes)

    # Common x and y labels
    if t0_sig_epoch_s is None:
        axes.set_xlabel("Time (s)", size=FigParam().text_size, labelpad=10)
    else:
        axes.set_xlabel("Time (s) relative to " + dt.datetime.utcfromtimestamp(t0_sig_epoch_s).strftime('%Y-%m-%d %H:%M:%S'),
                        size=FigParam().text_size, labelpad=10)

    # Format spacing in figure
    if isinstance(sig_id_label, str) is True:
        # Assumes Station IDs are ylabels
        left_spacing = 0.17
    else:
        left_spacing = 0.12

    if fig_title_show is True:
        axes.set_title(fig_title, size=FigParam().text_size + 2, y=1.05)
        # Adjust overall plot to maximize figure space for press if title on
        if common_colorbar is True and ytick_values_show is True:
            fig.subplots_adjust(left=left_spacing, top=0.92, hspace=0.25)
        elif common_colorbar is True and ytick_values_show is False:
            fig.subplots_adjust(left=left_spacing, top=0.92)
        elif common_colorbar is False and ytick_values_show is True:
            fig.subplots_adjust(left=left_spacing, right=0.91, top=0.92, hspace=0.2)
        else:
            fig.subplots_adjust(left=left_spacing, right=0.97, top=0.92)

    else:
        # Adjust overall plot to maximize figure space for press if title off
        if common_colorbar is False and ytick_values_show is False:
            fig.subplots_adjust(left=left_spacing, top=0.95, right=0.97)
        elif common_colorbar is False and ytick_values_show is True:
            fig.subplots_adjust(left=left_spacing, top=0.95, right=0.91, hspace=0.25)
        elif common_colorbar is True and ytick_values_show is True:
            fig.subplots_adjust(left=left_spacing, top=0.95, hspace=0.25)
        else:  # if common bar true
            fig.subplots_adjust(left=left_spacing, top=0.95)

    # Format colorbar
    if common_colorbar is True:
        cax = fig.add_subplot(gs[:, 1])
        mesh_panel_cbar: Colorbar = fig.colorbar(mappable=plotted, cax=cax)
        mesh_panel_cbar.ax.tick_params(labelsize=FigParam().text_size-2)
        mesh_panel_cbar.set_label('bits relative to max', rotation=270, size=FigParam().text_size, labelpad=25)

    if show_figure is True:
        plt.show()

    return fig
