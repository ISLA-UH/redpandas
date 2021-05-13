import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
import datetime as dt
import numpy as np
import pandas as pd
from libquantum.plot_templates import plot_time_frequency_reps as pnl
import redpandas.redpd_scales as rpd_scales
from typing import List

"""
Utils for plotting pandas dataframes
Created: 23 March 2021
Last updated: 12 May 2021
"""

# Wiggle plot scaling
scale = 1.25*1080/8
figure_size_x = int(1920/scale)
figure_size_y = int(1080/scale)
text_size = int(2.9*1080/scale)

# Colormap/
color_map = "inferno"  # 'hot_r'  # 'afmhot_r' #colormap for plotting


def plot_mesh_pandas(df: pd.DataFrame,
                     mesh_time_label: str,
                     mesh_frequency_label: str,
                     mesh_tfr_label: str,
                     t0_sig_epoch_s: float,
                     sig_id_label: str,
                     fig_title: str,
                     frequency_scaling: str = "log",
                     frequency_hz_ymin: float = rpd_scales.Slice.FU,
                     frequency_hz_ymax: float = rpd_scales.Slice.F0):

    """
     Plots spectrogram for all signals in df

     :param df: input pandas data frame
     :param mesh_time_label: string for the mesh time column name in df
     :param mesh_frequency_label: string for the mesh frequency column name in df
     :param mesh_tfr_label: string for the mesh tfr column name in df
     :param sig_id_label: string for column name with station ids in df
     :param t0_sig_epoch_s: epoch time in seconds of first timestamp
     :param fig_title: figure title label
     :param frequency_hz_ymin: y axis min lim
     :param frequency_hz_ymax: y axis max lim
     :param frequency_scaling: "log" or "lin". Default is "log"

     :return: plot
     """

    wiggle_num = len(df.index)  # total number of signal that will be displayed
    offset_scaling = 2**(np.log2(wiggle_num)+1.0)/wiggle_num
    wiggle_offset = np.arange(0, wiggle_num)*offset_scaling
    wiggle_yticks = wiggle_offset

    if sig_id_label == "index":
        wiggle_yticklabel = df.index
    else:
        wiggle_yticklabel = df[sig_id_label]

    # loop to find xlim and tfr global max/min
    x_lim_min = np.empty(wiggle_num)
    x_lim_max = np.empty(wiggle_num)
    tfr_min = np.empty(wiggle_num)
    tfr_max = np.empty(wiggle_num)

    for index_element in range(wiggle_num):
        x_lim_min[index_element] = np.min(df[mesh_time_label][index_element])
        x_lim_max[index_element] = np.max(df[mesh_time_label][index_element])
        tfr_min[index_element] = np.min(df[mesh_tfr_label][index_element])
        tfr_max[index_element] = np.max(df[mesh_tfr_label][index_element])

    # global min/max limits xlim
    x_lim_min_total = np.min(x_lim_min)
    x_lim_max_total = np.max(x_lim_max)

    # global min/max limits tfr
    # tfr_min_total = np.min(tfr_min)
    tfr_max_total = np.max(tfr_max) - 3
    tfr_min_total = tfr_max_total - 18

    # start of figure
    fig = plt.figure(figsize=(figure_size_x, figure_size_y))
    # This can be optimized/automated
    gs = fig.add_gridspec(nrows=wiggle_num, ncols=2, figure=fig, width_ratios=[10., 0.1], wspace=0.03)

    for panel_order, index_signal in enumerate(reversed(df.index)):
        ax = fig.add_subplot(gs[panel_order, 0])
        plotted = ax.pcolormesh(df[mesh_time_label][index_signal],
                                df[mesh_frequency_label][index_signal],
                                df[mesh_tfr_label][index_signal],
                                vmin=tfr_min_total,
                                vmax=tfr_max_total,
                                cmap=color_map,
                                edgecolor='face',
                                shading="auto",
                                snap=True)

        # set ax limits
        plt.xlim(x_lim_min_total, x_lim_max_total)

        # This is a very useful bit of code
        _, _, frequency_fix_ymin, frequency_fix_ymax = \
            pnl.mesh_time_frequency_edges(frequency=df[mesh_frequency_label][index_signal],
                                          time=df[mesh_time_label][index_signal],
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

        if panel_order < (wiggle_num-1):  # plot x ticks for only last subplot
            ax.set_xticks([])

        ax.tick_params(axis='both', which='major', labelsize=text_size)

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
    plt.title(fig_title, size=text_size + 2, y=1.05)

    # Format colorbar
    cax = fig.add_subplot(gs[:, 1])
    mesh_panel_cbar: Colorbar = fig.colorbar(mappable=plotted, cax=cax)
    mesh_panel_cbar.ax.tick_params(labelsize=text_size-2)
    mesh_panel_cbar.set_label('bits relative to max', rotation=270, size=text_size, labelpad=25)

    # Adjust overall plot to maximize figure space for press
    # plt.subplots_adjust(left=0.2, top=0.9)
    # plt.subplots_adjust(left=0.1, top=0.9)


def plot_wiggles_pandas(df: pd.DataFrame,
                        sig_wf_label: str,
                        sig_sample_rate_label: str,
                        sig_id_label: str,
                        x_label: str,
                        y_label: str,
                        fig_title: str = 'Signals',
                        wf_color: str = 'midnightblue',
                        sig_timestamps_label: str = None):
    """
    More nuanced plots with minimal distraction. Optimized for pandas input.
    From milton/hellborne/hell_gt_entropy_inv_clean
    Updated to plot data with different sample rates and number of points
    :param df: input pandas data frame
    :param sig_wf_label: string for the waveform name in df
    :param sig_sample_rate_label: string for the sample rate name in df
    :param x_label: x label
    :param y_label: y label
    :param fig_title: 'Normalized ' + title label
    :param wf_color: waveform color
    :param sig_timestamps_label: name for the epoch time in df, default = None
    :param sig_id_label: usually the index converted to a str, default = None
    :return: plot
    """

    fig, ax1 = plt.subplots(figsize=(figure_size_x, figure_size_y))

    # Number of non-consecutive pandas indexes don't always match
    wiggle_num = len(df.index)
    # print("Number of signals:", wiggle_num)
    offset_scaling = 2**(np.log2(wiggle_num)+1.0)/wiggle_num
    wiggle_offset = np.arange(0, wiggle_num)*offset_scaling
    wiggle_yticks = wiggle_offset
    if sig_id_label == "index":
        wiggle_yticklabel = df.index
    else:
        wiggle_yticklabel = df[sig_id_label]

    # For next iteration, include an epoch and/or elapsed time
    ax1.set_yticks(wiggle_yticks)
    ax1.set_yticklabels(wiggle_yticklabel)
    ax1.set_ylim(wiggle_offset[0]-offset_scaling, wiggle_offset[-1]+offset_scaling)
    ax1.tick_params(axis='both', which='both', labelsize=text_size)

    xlim_min = np.empty(wiggle_num)
    xlim_max = np.empty(wiggle_num)

    if sig_timestamps_label is not None:
        epoch_j = np.zeros(wiggle_num)
        for i, j in enumerate(df.index):
            epoch_j[i] = df[sig_timestamps_label][j].min()
        time_epoch_origin = np.min(epoch_j)
    else:
        time_epoch_origin = 0.

    for i, j in enumerate(df.index):
        if sig_timestamps_label is None:
            time_s = np.arange(len(df[sig_wf_label][j])) / df[sig_sample_rate_label][j]
        else:
            time_s = df[sig_timestamps_label][j] - time_epoch_origin

        sig_j = df[sig_wf_label][j] / np.max(df[sig_wf_label][j])
        ax1.plot(time_s, sig_j + wiggle_offset[i], color=wf_color)
        xlim_min[j] = np.min(time_s)
        xlim_max[j] = np.max(time_s)

    ax1.set_xlim(np.min(xlim_min), np.max(xlim_max))
    ax1.grid(True)
    ax1.set_title('Normalized ' + fig_title, size=text_size)
    ax1.set_ylabel(y_label, size=text_size)
    if time_epoch_origin > 0:
        x_label += " relative to " + dt.datetime.utcfromtimestamp(time_epoch_origin).strftime('%Y-%m-%d %H:%M:%S')
    ax1.set_xlabel(x_label, size=text_size)
    fig.tight_layout()


# TODO MC: what if i only want component x of a sensor
def plot_station_wiggles_pandas(df: pd.DataFrame,
                                station_id_str: str,
                                sensor_wf_label_list: List,
                                sensor_timestamps_label_list: List,
                                sig_id_label: str,
                                x_label: str,
                                y_label: str,
                                fig_title: str = 'Signals',
                                wf_color: str = 'midnightblue',
                                sensor_yticks_label_list: List[str] = None):

    fig, ax1 = plt.subplots(figsize=(figure_size_x, figure_size_y))

    wiggle_num = len(sensor_wf_label_list)  # depends on number of sensors and if sensors are 3d wf

    # Check if sensors are 3d and if so add to wiggle_num
    list_3d_sensor = ['accelerometer', 'gyroscope', 'magnetometer']
    sensor_wf_label_list_modified_for_ticklabels = sensor_wf_label_list.copy()   # Copy sensor list
    index_sensor_label_ticklabels_list = 0  # index for ticklabels once it includes x/y/z

    for index_sensor_label_in_list, sensor_label_in_list in enumerate(sensor_wf_label_list):

        if sensor_label_in_list.startswith(tuple(list_3d_sensor)):

            wiggle_num += 2  # for the waveforms that have not been counted yet

            # Modify the copy sensor labels list for yticks labels to include x,y,z labels
            sensor_wf_label_list_modified_for_ticklabels.remove(sensor_wf_label_list_modified_for_ticklabels[index_sensor_label_ticklabels_list])
            sensor_wf_label_list_modified_for_ticklabels.extend((sensor_wf_label_list[index_sensor_label_in_list] + '_x',
                                                                sensor_wf_label_list[index_sensor_label_in_list] + '_y',
                                                                sensor_wf_label_list[index_sensor_label_in_list] + '_z'))

            index_sensor_label_ticklabels_list += 2

        index_sensor_label_ticklabels_list += 1

    offset_scaling = 2**(np.log2(wiggle_num)+1.0)/wiggle_num
    wiggle_offset = np.arange(0, wiggle_num)*offset_scaling
    wiggle_yticks = wiggle_offset

    # Set ytick labels
    if sensor_yticks_label_list is None:
        wiggle_yticklabel = sensor_wf_label_list_modified_for_ticklabels
    else:
        wiggle_yticklabel = sensor_yticks_label_list

    ax1.set_yticks(wiggle_yticks)
    ax1.set_yticklabels(wiggle_yticklabel)
    ax1.set_ylim(wiggle_offset[0]-offset_scaling, wiggle_offset[-1]+offset_scaling)
    ax1.tick_params(axis='both', which='both', labelsize=text_size)

    xlim_min = np.empty(wiggle_num)
    xlim_max = np.empty(wiggle_num)

    station_row_index = df[sig_id_label].str.find(station_id_str)  # find index of desired station
    index_station = station_row_index[0]  # actually get index

    sensor_timestamps_label = sensor_timestamps_label_list[0]  # assume sensors same start time
    time_epoch_origin = df[sensor_timestamps_label][index_station][0]

    index_sensor_label_ticklabels_list = 0  # keep track of total sensor wf including x/y/z
    for index_sensor_in_list, label in enumerate(sensor_wf_label_list):

        # assume same order sensor wf and timestamp in lists
        sensor_wf_df = df[label][index_station]
        sensor_timestamps_label = sensor_timestamps_label_list[index_sensor_in_list]
        time_s = df[sensor_timestamps_label][index_station] - time_epoch_origin

        if sensor_wf_df.ndim == 1:  # sensor that is NOT acceleration/gyroscope/magnetometer

            sig_j = df[label][index_station] / np.max(df[label][index_station])
            ax1.plot(time_s, sig_j + wiggle_offset[index_sensor_label_ticklabels_list], color=wf_color)
            xlim_min[index_sensor_label_ticklabels_list] = np.min(time_s)
            xlim_max[index_sensor_label_ticklabels_list] = np.max(time_s)

            index_sensor_label_ticklabels_list += 1

        else:
            for index_dimension, sensor_array in enumerate(sensor_wf_df):

                sig_j = sensor_array / np.max(sensor_array)
                ax1.plot(time_s, sig_j + wiggle_offset[index_sensor_label_ticklabels_list], color=wf_color)
                xlim_min[index_sensor_label_ticklabels_list] = np.min(time_s)
                xlim_max[index_sensor_label_ticklabels_list] = np.max(time_s)

                index_sensor_label_ticklabels_list += 1

    ax1.set_xlim(np.min(xlim_min), np.max(xlim_max))
    ax1.grid(True)
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
                 f_scale: str = 'log',
                 sig_label: str = "PSD Sig",
                 ref_label: str = "PSD Ref",
                 psd_label: str = 'PSD (bits)',
                 coh_label: str = 'Coherence',
                 f_label: str = 'Frequency (Hz)',
                 fig_title: str = 'Power spectral density and coherence'):
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
                          fig_title: str = 'Response only valid at high coherence'):
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
