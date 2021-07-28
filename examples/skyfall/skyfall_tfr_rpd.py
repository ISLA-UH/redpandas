# Python libraries
import os.path
import matplotlib.pyplot as plt
import pandas as pd

# RedVox RedPandas and related RedVox modules
from redvox.common.data_window import DataWindow
import redpandas.redpd_preprocess as rpd_prep
import redpandas.redpd_df as rpd_df
import redpandas.redpd_plot.mesh as rpd_plot
import redpandas.redpd_tfr as rpd_tfr
import redpandas.redpd_datawin as rpd_dw
from libquantum.plot_templates import plot_time_frequency_reps as pnl

# Configuration file
from redpandas.redpd_config import DataLoadMethod
from examples.skyfall.skyfall_config_file import skyfall_config, tfr_config

axes = ["X", "Y", "Z"]
# TODO: build spectrogram version of libquantum's plot_wf_wf_wf_vert to match TDR plots?


def main():
    """
    RedVox RedPandas time-frequency representation of API900 data. Example: Skyfall.
    Last updated: 12 July 2021
    """

    print('Let the sky fall')

    # Label columns in dataframe
    station_label: str = "station_id"
    redvox_sdk_version_label: str = 'redvox_sdk_version'

    # Audio columns
    audio_data_label: str = "audio_wf"
    audio_epoch_s_label: str = "audio_epoch_s"
    audio_fs_label: str = "audio_sample_rate_nominal_hz"
    audio_tfr_bits_label: str = "audio_tfr_bits"
    audio_tfr_frequency_hz_label: str = "audio_tfr_frequency_hz"
    audio_tfr_time_s_label: str = "audio_tfr_time_s"

    # Barometer columns
    barometer_data_raw_label: str = "barometer_wf_raw"
    barometer_data_highpass_label: str = "barometer_wf_highpass"
    barometer_epoch_s_label: str = "barometer_epoch_s"
    barometer_fs_label: str = "barometer_sample_rate_hz"
    barometer_tfr_bits_label: str = "barometer_tfr_bits"
    barometer_tfr_frequency_hz_label: str = "barometer_tfr_frequency_hz"
    barometer_tfr_time_s_label: str = "barometer_tfr_time_s"

    # Accelerometer columns
    accelerometer_data_raw_label: str = "accelerometer_wf_raw"
    accelerometer_data_highpass_label: str = "accelerometer_wf_highpass"
    accelerometer_epoch_s_label: str = "accelerometer_epoch_s"
    accelerometer_fs_label: str = "accelerometer_sample_rate_hz"
    accelerometer_tfr_bits_label: str = "accelerometer_tfr_bits"
    accelerometer_tfr_frequency_hz_label: str = "accelerometer_tfr_frequency_hz"
    accelerometer_tfr_time_s_label: str = "accelerometer_tfr_time_s"

    # Gyroscope columns
    gyroscope_data_raw_label: str = "gyroscope_wf_raw"
    gyroscope_data_highpass_label: str = "gyroscope_wf_highpass"
    gyroscope_epoch_s_label: str = "gyroscope_epoch_s"
    gyroscope_fs_label: str = "gyroscope_sample_rate_hz"
    gyroscope_tfr_bits_label: str = "gyroscope_tfr_bits"
    gyroscope_tfr_frequency_hz_label: str = "gyroscope_tfr_frequency_hz"
    gyroscope_tfr_time_s_label: str = "gyroscope_tfr_time_s"

    # Magnetometer columns
    magnetometer_data_raw_label: str = "magnetometer_wf_raw"
    magnetometer_data_highpass_label: str = "magnetometer_wf_highpass"
    magnetometer_epoch_s_label: str = "magnetometer_epoch_s"
    magnetometer_fs_label: str = "magnetometer_sample_rate_hz"
    magnetometer_tfr_bits_label: str = "magnetometer_tfr_bits"
    magnetometer_tfr_frequency_hz_label: str = "magnetometer_tfr_frequency_hz"
    magnetometer_tfr_time_s_label: str = "magnetometer_tfr_time_s"

    # Load data options
    # RECOMMENDED: tfr_load_method="datawindow" in config file
    if tfr_config.tfr_load_method == DataLoadMethod.DATAWINDOW or \
            tfr_config.tfr_load_method == DataLoadMethod.PICKLE:
        print("Initiating Conversion from RedVox DataWindow to RedVox RedPandas:")
        if tfr_config.tfr_load_method == DataLoadMethod.DATAWINDOW:  # Option A: Create DataWindow object
            print("Constructing RedVox DataWindow...", end=" ")

            rdvx_data = rpd_dw.dw_from_redpd_config(config=skyfall_config)

        else:  # Option B: Load pickle with DataWindow object. Assume compressed
            print("Unpickling existing compressed RedVox DataWindow with JSON...", end=" ")
            rdvx_data: DataWindow = DataWindow.from_json_file(base_dir=skyfall_config.output_dir,
                                                              file_name=skyfall_config.output_filename_pkl_pqt)
        print(f"Done. RedVox SDK version: {rdvx_data.sdk_version}")

        # For option A or B, begin RedPandas
        df_skyfall_data = rpd_df.redpd_dataframe(rdvx_data, skyfall_config.sensor_labels)

    elif tfr_config.tfr_load_method == DataLoadMethod.PARQUET:  # Option C: Open dataframe from parquet file
        print("Loading existing RedPandas Parquet...", end=" ")
        df_skyfall_data = pd.read_parquet(os.path.join(skyfall_config.output_dir, skyfall_config.pd_pqt_file))
        print(f"Done. RedVox SDK version: {df_skyfall_data[redvox_sdk_version_label][0]}")

    else:
        print('\nNo data loading method selected.')
        exit()

    # PLOTTING
    print("\nInitiating time-frequency representation of Skyfall:")
    print(f"tfr_type: {tfr_config.tfr_type}, order: {tfr_config.tfr_order_number_N}")
    for station in df_skyfall_data.index:
        station_id_str = df_skyfall_data[station_label][station]  # Get the station id

        if audio_data_label and audio_fs_label in df_skyfall_data.columns:
            print('mic_sample_rate_hz: ', df_skyfall_data[audio_fs_label][station])
            print('mic_epoch_s_0: ', df_skyfall_data[audio_epoch_s_label][station][0])

            # Frame to mic start and end and plot
            event_reference_time_epoch_s = df_skyfall_data[audio_epoch_s_label][station][0]
            print(event_reference_time_epoch_s)

            df_skyfall_data = rpd_tfr.tfr_bits_panda(df=df_skyfall_data,
                                                     sig_wf_label=audio_data_label,
                                                     sig_sample_rate_label=audio_fs_label,
                                                     order_number_input=tfr_config.tfr_order_number_N,
                                                     tfr_type=tfr_config.tfr_type,
                                                     new_column_tfr_bits=audio_tfr_bits_label,
                                                     new_column_tfr_frequency_hz=audio_tfr_frequency_hz_label,
                                                     new_column_tfr_time_s=audio_tfr_time_s_label)

            pnl.plot_wf_mesh_vert(redvox_id=station_id_str,
                                  wf_panel_2_sig=df_skyfall_data[audio_data_label][station],
                                  wf_panel_2_time=df_skyfall_data[audio_epoch_s_label][station],
                                  mesh_time=df_skyfall_data[audio_tfr_time_s_label][station],
                                  mesh_frequency=df_skyfall_data[audio_tfr_frequency_hz_label][station],
                                  mesh_panel_0_tfr=df_skyfall_data[audio_tfr_bits_label][station],
                                  figure_title=skyfall_config.event_name +
                                               f": Audio, {tfr_config.tfr_type.upper()} and waveform",
                                  start_time_epoch=event_reference_time_epoch_s,
                                  mesh_panel_0_color_range=tfr_config.mc_range['Audio'],
                                  mesh_panel_0_colormap_scaling=tfr_config.mc_scale['Audio'],
                                  figure_title_show=tfr_config.show_fig_titles,
                                  wf_panel_2_units="Audio, Norm")

        if barometer_data_raw_label and barometer_data_highpass_label and barometer_fs_label in df_skyfall_data.columns:
            if skyfall_config.tdr_load_method == DataLoadMethod.PARQUET:
                # Reshape wf columns
                rpd_prep.df_column_unflatten(df=df_skyfall_data,
                                             col_wf_label=barometer_data_raw_label,
                                             col_ndim_label=barometer_data_raw_label + "_ndim")

                rpd_prep.df_column_unflatten(df=df_skyfall_data,
                                             col_wf_label=barometer_data_highpass_label,
                                             col_ndim_label=barometer_data_highpass_label + "_ndim")

            print('barometer_sample_rate_hz:', df_skyfall_data[barometer_fs_label][station])
            print('barometer_epoch_s_0:', df_skyfall_data[barometer_epoch_s_label][station][0])

            # baro height not currently used
            # barometer_height_m = \
            #    rpd_geo.bounder_model_height_from_pressure(df_skyfall_data[barometer_data_raw_label][station][0])
            # baro_height_from_bounder_km = barometer_height_m * METERS_TO_KM

            barometer_tfr_start_epoch: float = df_skyfall_data[barometer_epoch_s_label][0][0]
            if tfr_config.sensor_hp['Bar']:
                bar_sig_label, bar_hp_raw = barometer_data_highpass_label, 'hp'
            else:
                bar_sig_label, bar_hp_raw = barometer_data_raw_label, 'raw'
            print('Starting tfr_bits_panda for barometer:')
            df_skyfall_data = rpd_tfr.tfr_bits_panda(df=df_skyfall_data,
                                                     sig_wf_label=bar_sig_label,
                                                     sig_sample_rate_label=barometer_fs_label,
                                                     order_number_input=tfr_config.tfr_order_number_N,
                                                     tfr_type=tfr_config.tfr_type,
                                                     new_column_tfr_bits=barometer_tfr_bits_label,
                                                     new_column_tfr_frequency_hz=barometer_tfr_frequency_hz_label,
                                                     new_column_tfr_time_s=barometer_tfr_time_s_label)

            pnl.plot_wf_mesh_vert(redvox_id=station_id_str,
                                  wf_panel_2_sig=df_skyfall_data[bar_sig_label][station][0],
                                  wf_panel_2_time=df_skyfall_data[barometer_epoch_s_label][station],
                                  mesh_time=df_skyfall_data[barometer_tfr_time_s_label][station][0],
                                  mesh_frequency=df_skyfall_data[barometer_tfr_frequency_hz_label][station][0],
                                  mesh_panel_0_tfr=df_skyfall_data[barometer_tfr_bits_label][station][0],
                                  mesh_panel_0_colormap_scaling=tfr_config.mc_scale["Bar"],
                                  mesh_panel_0_color_range=tfr_config.mc_range["Bar"],
                                  figure_title=skyfall_config.event_name +
                                               f": Barometer, {tfr_config.tfr_type.upper()} and waveform",
                                  start_time_epoch=barometer_tfr_start_epoch,
                                  figure_title_show=tfr_config.show_fig_titles,
                                  wf_panel_2_units=f"Bar {bar_hp_raw}, kPa")

        # Repeat here
        if accelerometer_data_raw_label and accelerometer_fs_label and accelerometer_data_highpass_label \
                in df_skyfall_data.columns:
            if skyfall_config.tdr_load_method == DataLoadMethod.PARQUET:
                # Reshape wf columns
                rpd_prep.df_column_unflatten(df=df_skyfall_data,
                                             col_wf_label=accelerometer_data_raw_label,
                                             col_ndim_label=accelerometer_data_raw_label + "_ndim")
                rpd_prep.df_column_unflatten(df=df_skyfall_data,
                                             col_wf_label=accelerometer_data_highpass_label,
                                             col_ndim_label=accelerometer_data_highpass_label + "_ndim")

            print('accelerometer_sample_rate_hz:', df_skyfall_data[accelerometer_fs_label][station])
            print('accelerometer_epoch_s_0:', df_skyfall_data[accelerometer_epoch_s_label][station][0],
                  df_skyfall_data[accelerometer_epoch_s_label][station][-1])

            # time-domain plots in skyfall_tdr_rpd.py
            # # Plot 3c acceleration highpass waveforms
            # pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
            #                        wf_panel_2_sig=df_skyfall_data[accelerometer_data_highpass_label][station][2],
            #                        wf_panel_2_time=df_skyfall_data[accelerometer_epoch_s_label][station],
            #                        wf_panel_1_sig=df_skyfall_data[accelerometer_data_highpass_label][station][1],
            #                        wf_panel_1_time=df_skyfall_data[accelerometer_epoch_s_label][station],
            #                        wf_panel_0_sig=df_skyfall_data[accelerometer_data_highpass_label][station][0],
            #                        wf_panel_0_time=df_skyfall_data[accelerometer_epoch_s_label][station],
            #                        start_time_epoch=event_reference_time_epoch_s,
            #                        wf_panel_2_units="Acc Z, m/$s^2$",
            #                        wf_panel_1_units="Acc Y, m/$s^2$",
            #                        wf_panel_0_units="Acc X, m/$s^2$",
            #                        figure_title=EVENT_NAME + ": Accelerometer highpass",
            #                        figure_title_show=True,
            #                        label_panel_show=False,
            #                        labels_fontweight='bold')

            acceleromter_tfr_start_epoch: float = df_skyfall_data[accelerometer_epoch_s_label][0][0]
            if tfr_config.sensor_hp['Acc']:
                acc_sig_label, acc_hp_raw = accelerometer_data_highpass_label, 'hp'
            else:
                acc_sig_label, acc_hp_raw = accelerometer_data_raw_label, 'raw'
            print('Starting tfr_bits_panda for 3 channel acceleration:')
            df_skyfall_data = rpd_tfr.tfr_bits_panda(df=df_skyfall_data,
                                                     sig_wf_label=acc_sig_label,
                                                     sig_sample_rate_label=accelerometer_fs_label,
                                                     order_number_input=tfr_config.tfr_order_number_N,
                                                     tfr_type=tfr_config.tfr_type,
                                                     new_column_tfr_bits=accelerometer_tfr_bits_label,
                                                     new_column_tfr_frequency_hz=accelerometer_tfr_frequency_hz_label,
                                                     new_column_tfr_time_s=accelerometer_tfr_time_s_label)

            for ax_n in range(3):
                pnl.plot_wf_mesh_vert(redvox_id=station_id_str,
                                      wf_panel_2_sig=df_skyfall_data[acc_sig_label][station][ax_n],
                                      wf_panel_2_time=df_skyfall_data[accelerometer_epoch_s_label][station],
                                      mesh_time=df_skyfall_data[accelerometer_tfr_time_s_label][station][ax_n],
                                      mesh_frequency=df_skyfall_data[accelerometer_tfr_frequency_hz_label][station][
                                          ax_n],
                                      mesh_panel_0_tfr=df_skyfall_data[accelerometer_tfr_bits_label][station][ax_n],
                                      mesh_panel_0_colormap_scaling=tfr_config.mc_scale["Acc"],
                                      mesh_panel_0_color_range=tfr_config.mc_range["Acc"],
                                      figure_title=skyfall_config.event_name +
                                                   f": Accelerometer, {tfr_config.tfr_type.upper()} and waveform",
                                      start_time_epoch=acceleromter_tfr_start_epoch,
                                      figure_title_show=tfr_config.show_fig_titles,
                                      wf_panel_2_units=f"Acc {axes[ax_n]} {acc_hp_raw}, m/$s^2$")

        if gyroscope_data_raw_label and gyroscope_fs_label and gyroscope_data_highpass_label \
                in df_skyfall_data.columns:

            if skyfall_config.tdr_load_method == DataLoadMethod.PARQUET:
                # Reshape wf columns
                rpd_prep.df_column_unflatten(df=df_skyfall_data,
                                             col_wf_label=gyroscope_data_raw_label,
                                             col_ndim_label=gyroscope_data_raw_label + "_ndim")

                rpd_prep.df_column_unflatten(df=df_skyfall_data,
                                             col_wf_label=gyroscope_data_highpass_label,
                                             col_ndim_label=gyroscope_data_highpass_label + "_ndim")

            print('gyroscope_sample_rate_hz:', df_skyfall_data[gyroscope_fs_label][station])
            print('gyroscope_epoch_s_0:', df_skyfall_data[gyroscope_epoch_s_label][station][0],
                  df_skyfall_data[gyroscope_epoch_s_label][station][-1])

            # time-domain plots in skyfall_tdr_rpd.py
            # # Plot 3c gyroscope highpass waveforms
            # pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
            #                        wf_panel_2_sig=df_skyfall_data[gyroscope_data_highpass_label][station][2],
            #                        wf_panel_2_time=df_skyfall_data[gyroscope_epoch_s_label][station],
            #                        wf_panel_1_sig=df_skyfall_data[gyroscope_data_highpass_label][station][1],
            #                        wf_panel_1_time=df_skyfall_data[gyroscope_epoch_s_label][station],
            #                        wf_panel_0_sig=df_skyfall_data[gyroscope_data_highpass_label][station][0],
            #                        wf_panel_0_time=df_skyfall_data[gyroscope_epoch_s_label][station],
            #                        start_time_epoch=event_reference_time_epoch_s,
            #                        wf_panel_2_units="Gyr Z, rad/s$",
            #                        wf_panel_1_units="Gyr Y, rad/s$",
            #                        wf_panel_0_units="Gyr X, rad/s$",
            #                        figure_title=EVENT_NAME + ": Gyroscope highpass",
            #                        figure_title_show=True,
            #                        label_panel_show=False,
            #                        labels_fontweight='bold')

            gyroscope_tfr_start_epoch: float = df_skyfall_data[gyroscope_epoch_s_label][0][0]
            if tfr_config.sensor_hp['Gyr']:
                gyr_sig_label, gyr_hp_raw = gyroscope_data_highpass_label, 'hp'
            else:
                gyr_sig_label, gyr_hp_raw = gyroscope_data_raw_label, 'raw'
            print('Starting tfr_bits_panda for 3 channel gyroscope:')
            df_skyfall_data = rpd_tfr.tfr_bits_panda(df=df_skyfall_data,
                                                     sig_wf_label=gyr_sig_label,
                                                     sig_sample_rate_label=gyroscope_fs_label,
                                                     order_number_input=tfr_config.tfr_order_number_N,
                                                     tfr_type=tfr_config.tfr_type,
                                                     new_column_tfr_bits=gyroscope_tfr_bits_label,
                                                     new_column_tfr_frequency_hz=gyroscope_tfr_frequency_hz_label,
                                                     new_column_tfr_time_s=gyroscope_tfr_time_s_label)
            for ax_n in range(3):
                pnl.plot_wf_mesh_vert(redvox_id=station_id_str,
                                      wf_panel_2_sig=df_skyfall_data[gyr_sig_label][station][ax_n],
                                      wf_panel_2_time=df_skyfall_data[gyroscope_epoch_s_label][station],
                                      mesh_time=df_skyfall_data[gyroscope_tfr_time_s_label][station][ax_n],
                                      mesh_frequency=df_skyfall_data[gyroscope_tfr_frequency_hz_label][station][
                                          ax_n],
                                      mesh_panel_0_tfr=df_skyfall_data[gyroscope_tfr_bits_label][station][ax_n],
                                      mesh_panel_0_colormap_scaling=tfr_config.mc_scale["Gyr"],
                                      mesh_panel_0_color_range=tfr_config.mc_range["Gyr"],
                                      figure_title=skyfall_config.event_name +
                                                   f": Gyroscope, {tfr_config.tfr_type.upper()} and waveform",
                                      start_time_epoch=gyroscope_tfr_start_epoch,
                                      figure_title_show=tfr_config.show_fig_titles,
                                      wf_panel_2_units=f"Gyr {axes[ax_n]} {gyr_hp_raw}, rad/s")

        if magnetometer_data_raw_label and magnetometer_fs_label and magnetometer_data_highpass_label \
                in df_skyfall_data.columns:
            if skyfall_config.tdr_load_method == DataLoadMethod.PARQUET:
                # Reshape wf columns
                rpd_prep.df_column_unflatten(df=df_skyfall_data,
                                             col_wf_label=magnetometer_data_raw_label,
                                             col_ndim_label=magnetometer_data_raw_label + "_ndim")
                rpd_prep.df_column_unflatten(df=df_skyfall_data,
                                             col_wf_label=magnetometer_data_highpass_label,
                                             col_ndim_label=magnetometer_data_highpass_label + "_ndim")

            print('magnetometer_sample_rate_hz:', df_skyfall_data[magnetometer_fs_label][station])
            print('magnetometer_epoch_s_0:', df_skyfall_data[magnetometer_epoch_s_label][station][0],
                  df_skyfall_data[magnetometer_epoch_s_label][station][-1])

            # time-domain plots in skyfall_tdr_rpd.py
            # # Plot 3c magnetometer highpass waveforms
            # pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
            #                        wf_panel_2_sig=df_skyfall_data[magnetometer_data_highpass_label][station][2],
            #                        wf_panel_2_time=df_skyfall_data[magnetometer_epoch_s_label][station],
            #                        wf_panel_1_sig=df_skyfall_data[magnetometer_data_highpass_label][station][1],
            #                        wf_panel_1_time=df_skyfall_data[magnetometer_epoch_s_label][station],
            #                        wf_panel_0_sig=df_skyfall_data[magnetometer_data_highpass_label][station][0],
            #                        wf_panel_0_time=df_skyfall_data[magnetometer_epoch_s_label][station],
            #                        start_time_epoch=event_reference_time_epoch_s,
            #                        wf_panel_2_units="Mag Z, $\mu$T",
            #                        wf_panel_1_units="Mag Y, $\mu$T",
            #                        wf_panel_0_units="Mag X, $\mu$T",
            #                        figure_title=skyfall_config.event_name + ": Magnetometer highpass",
            #                        figure_title_show=True,
            #                        label_panel_show=False,
            #                        labels_fontweight='bold')

            magnetometer_tfr_start_epoch: float = df_skyfall_data[magnetometer_epoch_s_label][0][0]
            if tfr_config.sensor_hp['Mag']:
                mag_sig_label, mag_hp_raw = magnetometer_data_highpass_label, 'hp'
            else:
                mag_sig_label, mag_hp_raw = magnetometer_data_raw_label, 'raw'
            print('Starting tfr_bits_panda for 3 channel magnetometer:')
            df_skyfall_data = rpd_tfr.tfr_bits_panda(df=df_skyfall_data,
                                                     sig_wf_label=mag_sig_label,
                                                     sig_sample_rate_label=magnetometer_fs_label,
                                                     order_number_input=tfr_config.tfr_order_number_N,
                                                     tfr_type=tfr_config.tfr_type,
                                                     new_column_tfr_bits=magnetometer_tfr_bits_label,
                                                     new_column_tfr_frequency_hz=magnetometer_tfr_frequency_hz_label,
                                                     new_column_tfr_time_s=magnetometer_tfr_time_s_label)
            for ax_n in range(3):
                pnl.plot_wf_mesh_vert(redvox_id=station_id_str,
                                      wf_panel_2_sig=df_skyfall_data[mag_sig_label][station][ax_n],
                                      wf_panel_2_time=df_skyfall_data[magnetometer_epoch_s_label][station],
                                      mesh_time=df_skyfall_data[magnetometer_tfr_time_s_label][station][ax_n],
                                      mesh_frequency=df_skyfall_data[magnetometer_tfr_frequency_hz_label][station][
                                          ax_n],
                                      mesh_panel_0_tfr=df_skyfall_data[magnetometer_tfr_bits_label][station][ax_n],
                                      mesh_panel_0_colormap_scaling=tfr_config.mc_scale["Mag"],
                                      mesh_panel_0_color_range=tfr_config.mc_range["Mag"],
                                      figure_title=skyfall_config.event_name +
                                                   f": Magnetometer, {tfr_config.tfr_type.upper()} and waveform",
                                      start_time_epoch=magnetometer_tfr_start_epoch,
                                      figure_title_show=tfr_config.show_fig_titles,
                                      wf_panel_2_units=f"Mag {axes[ax_n]} {mag_hp_raw}, $\mu$T")

        # ALL-SENSOR PLOTS:

        # time-domain plots in skyfall_tdr_rpd.py
        # Plot TDR sensor wiggles
        # sensor_column_label_list = [audio_data_label, barometer_data_highpass_label,
        #                             accelerometer_data_highpass_label, gyroscope_data_highpass_label,
        #                             magnetometer_data_highpass_label]
        #
        # sensor_epoch_column_label_list = [audio_epoch_s_label, barometer_epoch_s_label,
        #                                   accelerometer_epoch_s_label, gyroscope_epoch_s_label,
        #                                   magnetometer_epoch_s_label]
        #
        # sensor_ticklabels_list = ['Audio', 'Bar hp', 'Acc X hp', 'Acc Y hp',
        #                           'Acc Z hp', 'Gyr X hp', 'Gyr Y hp', 'Gyr Z hp',
        #                           'Mag X hp', 'Mag Y hp', 'Mag Z hp']
        #
        # rpd_plot.plot_sensor_wiggles_pandas(df=df_skyfall_data,
        #                                     station_id_str='1637610021',
        #                                     sensor_wf_label_list=sensor_column_label_list,
        #                                     sensor_timestamps_label_list=sensor_epoch_column_label_list,
        #                                     sig_id_label='station_id',
        #                                     x_label='Time (s)',
        #                                     y_label='Sensor',
        #                                     fig_title_show=False,
        #                                     fig_title='sensor waveforms',
        #                                     wf_color='midnightblue',
        #                                     sensor_yticks_label_list=sensor_ticklabels_list)

        # Plot TFR sensor wiggles
        rpd_plot.plot_mesh_pandas(df=df_skyfall_data,
                                  mesh_time_label=[audio_tfr_time_s_label,
                                                   barometer_tfr_time_s_label,
                                                   accelerometer_tfr_time_s_label,
                                                   gyroscope_tfr_time_s_label,
                                                   magnetometer_tfr_time_s_label],
                                  mesh_frequency_label=[audio_tfr_frequency_hz_label,
                                                        barometer_tfr_frequency_hz_label,
                                                        accelerometer_tfr_frequency_hz_label,
                                                        gyroscope_tfr_frequency_hz_label,
                                                        magnetometer_tfr_frequency_hz_label],
                                  mesh_tfr_label=[audio_tfr_bits_label,
                                                  barometer_tfr_bits_label,
                                                  accelerometer_tfr_bits_label,
                                                  gyroscope_tfr_bits_label,
                                                  magnetometer_tfr_bits_label],
                                  t0_sig_epoch_s=df_skyfall_data[audio_epoch_s_label][0][0],
                                  sig_id_label=["Audio", "Bar",
                                                "Acc X", "Acc Y", "Acc Z",
                                                'Gyr X', 'Gyr Y', 'Gyr Z',
                                                'Mag X', 'Mag Y', 'Mag Z'],
                                  fig_title_show=tfr_config.show_fig_titles,
                                  fig_title="",
                                  frequency_scaling='log',
                                  common_colorbar=False,
                                  mesh_color_scaling=[tfr_config.mc_scale["Audio"], tfr_config.mc_scale["Bar"],
                                                      tfr_config.mc_scale["Acc"], tfr_config.mc_scale["Acc"],
                                                      tfr_config.mc_scale["Acc"],
                                                      tfr_config.mc_scale["Gyr"], tfr_config.mc_scale["Gyr"],
                                                      tfr_config.mc_scale["Gyr"],
                                                      tfr_config.mc_scale["Mag"], tfr_config.mc_scale["Mag"],
                                                      tfr_config.mc_scale["Mag"]],
                                  mesh_color_range=[tfr_config.mc_range["Audio"], tfr_config.mc_range["Bar"],
                                                    tfr_config.mc_range["Acc"], tfr_config.mc_range["Acc"],
                                                    tfr_config.mc_range["Acc"],
                                                    tfr_config.mc_range["Gyr"], tfr_config.mc_range["Gyr"],
                                                    tfr_config.mc_range["Gyr"],
                                                    tfr_config.mc_range["Mag"], tfr_config.mc_range["Mag"],
                                                    tfr_config.mc_range["Mag"]])

        plt.show()


if __name__ == "__main__":
    main()
