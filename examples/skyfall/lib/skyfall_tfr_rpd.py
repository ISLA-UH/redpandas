"""
Skyfall TFR RPD
"""
# Python libraries
import matplotlib.pyplot as plt
import numpy as np

# RedVox RedPandas and related RedVox modules
import redpandas.redpd_preprocess as rpd_prep
import redpandas.redpd_plot.mesh as rpd_plot
import redpandas.redpd_tfr as rpd_tfr
from quantum_inferno.plot_templates import plot_base as pbase, plot_templates as pnl
import examples.skyfall.lib.skyfall_dw as sf_dw

# Configuration file
from redpandas.redpd_config import DataLoadMethod
from examples.skyfall.skyfall_config_file import skyfall_config, tfr_config

AXES = ["X", "Y", "Z"]


def main():
    """
    RedVox RedPandas time-frequency representation of API900 data. Example: Skyfall.
    Last updated: 20210909
    """
    # Label columns in dataframe
    station_label: str = "station_id"

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
    df_skyfall_data = sf_dw.dw_main(tfr_config.tfr_load_method)

    # PLOTTING
    print("\nInitiating time-frequency representation of Skyfall:")
    print(f"tfr_type: {tfr_config.tfr_type}, order: {tfr_config.tfr_order_number_n}")
    for station in df_skyfall_data.index:
        station_id_str = df_skyfall_data[station_label][station]  # Get the station id

        if audio_data_label and audio_fs_label in df_skyfall_data.columns:
            print(f"mic_sample_rate_hz: {df_skyfall_data[audio_fs_label][station]}")
            print(f"mic_epoch_s_0: {df_skyfall_data[audio_epoch_s_label][station][0]}")

            # Frame to mic start and end and plot
            event_reference_time_epoch_s = df_skyfall_data[audio_epoch_s_label][station][0]
            print(event_reference_time_epoch_s)

            df_skyfall_data = rpd_tfr.tfr_bits_panda(df=df_skyfall_data,
                                                     sig_wf_label=audio_data_label,
                                                     sig_sample_rate_label=audio_fs_label,
                                                     order_number_input=tfr_config.tfr_order_number_n,
                                                     tfr_type=tfr_config.tfr_type,
                                                     new_column_tfr_bits=audio_tfr_bits_label,
                                                     new_column_tfr_frequency_hz=audio_tfr_frequency_hz_label,
                                                     new_column_tfr_time_s=audio_tfr_time_s_label)

            mesh_base = pbase.MeshBase(time=df_skyfall_data[audio_tfr_time_s_label][station],
                                       frequency=df_skyfall_data[audio_tfr_frequency_hz_label][station])
            mesh_pnl = pbase.MeshPanel(tfr=df_skyfall_data[audio_tfr_bits_label][station],
                                       colormap_scaling=tfr_config.mc_scale["Audio"],
                                       color_range=tfr_config.mc_range["Audio"])
            wf_base = pbase.WaveformPlotBase(station_id=station_id_str,
                                             figure_title=f"{skyfall_config.event_name}: Audio, "
                                                          f"{tfr_config.tfr_type.upper()} and waveform",
                                             figure_title_show=tfr_config.show_fig_titles,
                                             start_time_epoch=event_reference_time_epoch_s)
            wf_pnl = pbase.WaveformPanel(sig=df_skyfall_data[audio_data_label][station],
                                         time=df_skyfall_data[audio_epoch_s_label][station],
                                         units="Audio, Norm")
            fig_awf_mesh = pnl.plot_mesh_wf_vert(mesh_base, mesh_pnl, wf_base, wf_pnl)

        if barometer_data_raw_label and barometer_data_highpass_label and barometer_fs_label in df_skyfall_data.columns:
            if skyfall_config.tdr_load_method == DataLoadMethod.PARQUET:
                # Reshape wf columns
                rpd_prep.df_column_unflatten(df=df_skyfall_data,
                                             col_wf_label=barometer_data_raw_label,
                                             col_ndim_label=barometer_data_raw_label + "_ndim")

                rpd_prep.df_column_unflatten(df=df_skyfall_data,
                                             col_wf_label=barometer_data_highpass_label,
                                             col_ndim_label=barometer_data_highpass_label + "_ndim")

            print(f"barometer_sample_rate_hz: {df_skyfall_data[barometer_fs_label][station]}")
            print(f"barometer_epoch_s_0: {df_skyfall_data[barometer_epoch_s_label][station][0]}")

            barometer_tfr_start_epoch: float = df_skyfall_data[barometer_epoch_s_label][0][0]
            if tfr_config.sensor_hp["Bar"]:
                bar_sig_label, bar_hp_raw = barometer_data_highpass_label, "hp"
            else:
                bar_sig_label, bar_hp_raw = barometer_data_raw_label, "raw"
            print("Starting tfr_bits_panda for barometer:")
            df_skyfall_data = rpd_tfr.tfr_bits_panda(df=df_skyfall_data,
                                                     sig_wf_label=bar_sig_label,
                                                     sig_sample_rate_label=barometer_fs_label,
                                                     order_number_input=tfr_config.tfr_order_number_n,
                                                     tfr_type=tfr_config.tfr_type,
                                                     new_column_tfr_bits=barometer_tfr_bits_label,
                                                     new_column_tfr_frequency_hz=barometer_tfr_frequency_hz_label,
                                                     new_column_tfr_time_s=barometer_tfr_time_s_label)

            mesh_base = pbase.MeshBase(time=df_skyfall_data[barometer_tfr_time_s_label][station][0],
                                       frequency=df_skyfall_data[barometer_tfr_frequency_hz_label][station][0])
            mesh_pnl = pbase.MeshPanel(tfr=df_skyfall_data[barometer_tfr_bits_label][station][0],
                                       colormap_scaling=tfr_config.mc_scale["Bar"],
                                       color_range=tfr_config.mc_range["Bar"])
            wf_base = pbase.WaveformPlotBase(station_id=station_id_str,
                                             figure_title=f"{skyfall_config.event_name}: Barometer, "
                                                          f"{tfr_config.tfr_type.upper()} and waveform",
                                             figure_title_show=tfr_config.show_fig_titles,
                                             start_time_epoch=barometer_tfr_start_epoch)
            wf_pnl = pbase.WaveformPanel(sig=df_skyfall_data[bar_sig_label][station][0],
                                         time=df_skyfall_data[barometer_epoch_s_label][station],
                                         units=f"Bar {bar_hp_raw}, kPa")
            fig_bwf_mesh = pnl.plot_mesh_wf_vert(mesh_base, mesh_pnl, wf_base, wf_pnl)

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

            accelerometer_tfr_start_epoch: float = df_skyfall_data[accelerometer_epoch_s_label][0][0]
            if tfr_config.sensor_hp['Acc']:
                acc_sig_label, acc_hp_raw = accelerometer_data_highpass_label, 'hp'
            else:
                acc_sig_label, acc_hp_raw = accelerometer_data_raw_label, 'raw'
            print('Starting tfr_bits_panda for 3 channel acceleration:')
            df_skyfall_data = rpd_tfr.tfr_bits_panda(df=df_skyfall_data,
                                                     sig_wf_label=acc_sig_label,
                                                     sig_sample_rate_label=accelerometer_fs_label,
                                                     order_number_input=tfr_config.tfr_order_number_n,
                                                     tfr_type=tfr_config.tfr_type,
                                                     new_column_tfr_bits=accelerometer_tfr_bits_label,
                                                     new_column_tfr_frequency_hz=accelerometer_tfr_frequency_hz_label,
                                                     new_column_tfr_time_s=accelerometer_tfr_time_s_label)
            fig_axn_mesh = []

            for ax_n in range(3):
                mesh_base = \
                    pbase.MeshBase(time=df_skyfall_data[accelerometer_tfr_time_s_label][station][ax_n],
                                   frequency=df_skyfall_data[accelerometer_tfr_frequency_hz_label][station][ax_n])
                mesh_pnl = pbase.MeshPanel(tfr=df_skyfall_data[accelerometer_tfr_bits_label][station][ax_n],
                                           colormap_scaling=tfr_config.mc_scale["Acc"],
                                           color_range=tfr_config.mc_range["Acc"])
                wf_base = pbase.WaveformPlotBase(station_id=station_id_str,
                                                 figure_title=f"{skyfall_config.event_name}: Accelerometer, "
                                                              f"{tfr_config.tfr_type.upper()} and waveform",
                                                 figure_title_show=tfr_config.show_fig_titles,
                                                 start_time_epoch=accelerometer_tfr_start_epoch)
                wf_pnl = pbase.WaveformPanel(sig=df_skyfall_data[acc_sig_label][station][ax_n],
                                             time=df_skyfall_data[accelerometer_epoch_s_label][station],
                                             units=f"Acc {AXES[ax_n]} {acc_hp_raw}, m/$s^2$")
                fig_axn_mesh.append(pnl.plot_mesh_wf_vert(mesh_base, mesh_pnl, wf_base, wf_pnl))

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

            gyroscope_tfr_start_epoch: float = df_skyfall_data[gyroscope_epoch_s_label][0][0]
            if tfr_config.sensor_hp['Gyr']:
                gyr_sig_label, gyr_hp_raw = gyroscope_data_highpass_label, 'hp'
            else:
                gyr_sig_label, gyr_hp_raw = gyroscope_data_raw_label, 'raw'
            print('Starting tfr_bits_panda for 3 channel gyroscope:')
            df_skyfall_data = rpd_tfr.tfr_bits_panda(df=df_skyfall_data,
                                                     sig_wf_label=gyr_sig_label,
                                                     sig_sample_rate_label=gyroscope_fs_label,
                                                     order_number_input=tfr_config.tfr_order_number_n,
                                                     tfr_type=tfr_config.tfr_type,
                                                     new_column_tfr_bits=gyroscope_tfr_bits_label,
                                                     new_column_tfr_frequency_hz=gyroscope_tfr_frequency_hz_label,
                                                     new_column_tfr_time_s=gyroscope_tfr_time_s_label)
            fig_gxn_mesh = []
            for ax_n in range(3):
                mesh_base = pbase.MeshBase(time=df_skyfall_data[gyroscope_tfr_time_s_label][station][ax_n],
                                           frequency=df_skyfall_data[gyroscope_tfr_frequency_hz_label][station][ax_n])
                mesh_pnl = pbase.MeshPanel(tfr=df_skyfall_data[gyroscope_tfr_bits_label][station][ax_n],
                                           colormap_scaling=tfr_config.mc_scale["Gyr"],
                                           color_range=tfr_config.mc_range["Gyr"])
                wf_base = pbase.WaveformPlotBase(station_id=station_id_str,
                                                 figure_title=f"{skyfall_config.event_name}: Gyroscope, "
                                                              f"{tfr_config.tfr_type.upper()} and waveform",
                                                 figure_title_show=tfr_config.show_fig_titles,
                                                 start_time_epoch=gyroscope_tfr_start_epoch)
                wf_pnl = pbase.WaveformPanel(sig=df_skyfall_data[gyr_sig_label][station][ax_n],
                                             time=df_skyfall_data[gyroscope_epoch_s_label][station],
                                             units=f"Gyr {AXES[ax_n]} {gyr_hp_raw}, rad/s")
                fig_gxn_mesh.append(pnl.plot_mesh_wf_vert(mesh_base, mesh_pnl, wf_base, wf_pnl))

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

            magnetometer_tfr_start_epoch: float = df_skyfall_data[magnetometer_epoch_s_label][0][0]
            if tfr_config.sensor_hp['Mag']:
                mag_sig_label, mag_hp_raw = magnetometer_data_highpass_label, 'hp'
            else:
                mag_sig_label, mag_hp_raw = magnetometer_data_raw_label, 'raw'
            print('Starting tfr_bits_panda for 3 channel magnetometer:')
            df_skyfall_data = rpd_tfr.tfr_bits_panda(df=df_skyfall_data,
                                                     sig_wf_label=mag_sig_label,
                                                     sig_sample_rate_label=magnetometer_fs_label,
                                                     order_number_input=tfr_config.tfr_order_number_n,
                                                     tfr_type=tfr_config.tfr_type,
                                                     new_column_tfr_bits=magnetometer_tfr_bits_label,
                                                     new_column_tfr_frequency_hz=magnetometer_tfr_frequency_hz_label,
                                                     new_column_tfr_time_s=magnetometer_tfr_time_s_label)
            fig_mxn_mesh = []
            for ax_n in range(3):
                mesh_base = \
                    pbase.MeshBase(time=df_skyfall_data[magnetometer_tfr_time_s_label][station][ax_n],
                                   frequency=df_skyfall_data[magnetometer_tfr_frequency_hz_label][station][ax_n])
                mesh_pnl = pbase.MeshPanel(tfr=df_skyfall_data[magnetometer_tfr_bits_label][station][ax_n],
                                           colormap_scaling=tfr_config.mc_scale["Mag"],
                                           color_range=tfr_config.mc_range["Mag"])
                wf_base = pbase.WaveformPlotBase(station_id=station_id_str,
                                                 figure_title=f"{skyfall_config.event_name}: Magnetometer, "
                                                              f"{tfr_config.tfr_type.upper()} and waveform",
                                                 figure_title_show=tfr_config.show_fig_titles,
                                                 start_time_epoch=magnetometer_tfr_start_epoch)
                wf_pnl = pbase.WaveformPanel(sig=df_skyfall_data[mag_sig_label][station][ax_n],
                                             time=df_skyfall_data[magnetometer_epoch_s_label][station],
                                             units=f"Mag {AXES[ax_n]} {mag_hp_raw}, $\mu$T")
                fig_mxn_mesh.append(pnl.plot_mesh_wf_vert(mesh_base, mesh_pnl, wf_base, wf_pnl))

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
                                  common_colorbar=True,
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
                                                    tfr_config.mc_range["Mag"]],
                                  ytick_values_show=False)

        plt.show()


if __name__ == "__main__":
    main()
