"""
Plot TFR
"""
import datetime as dt
from typing import List, Union

import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from libquantum.plot_templates import plot_time_frequency_reps as pnl

import redpandas.redpd_scales as rpd_scales
from redpandas.redpd_plot.parameters import FigureParameters as FigParam


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
    if len(mesh_tfr_label) != len(mesh_time_label) \
            or len(mesh_tfr_label) != len(mesh_frequency_label) \
            or len(mesh_time_label) != len(mesh_frequency_label):
        raise ValueError("mesh_time_label, mesh_tfr_label, "
                         "or mesh_frequency_label do not have the same length. Please check.")

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
    fig = plt.figure(figsize=(FigParam().figure_size_x, FigParam().figure_size_y))
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
                ax.set_yticklabels([wiggle_yticklabel[index_signal]], size=FigParam().text_size)

                if index_panel_order < (wiggle_num - 1):  # plot x ticks for only last subplot
                    ax.set_xticks([])

                ax.tick_params(axis='both', which='major', labelsize=FigParam().text_size)

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
                    ax.set_yticklabels([wiggle_yticklabel[index_wiggle_yticklabels]], size=FigParam().text_size)

                    if index_panel_order < (wiggle_num - 1):  # plot x ticks for only last subplot
                        ax.set_xticks([])

                    ax.tick_params(axis='both', which='major', labelsize=FigParam().text_size)

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
               size=FigParam().text_size, labelpad=10)
    if fig_title_show:
        plt.title(fig_title, size=FigParam().text_size + 2, y=1.05)
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
        mesh_panel_cbar.ax.tick_params(labelsize=FigParam().text_size-2)
        mesh_panel_cbar.set_label('bits relative to max', rotation=270, size=FigParam().text_size, labelpad=25)

    if show_figure is True:
        plt.show()

    return fig
