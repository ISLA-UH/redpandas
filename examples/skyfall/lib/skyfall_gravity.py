"""
SkyFall Gravity
"""
# todo: address possible invalid values in building plots section
# Python libraries
import matplotlib.pyplot as plt
import numpy as np
from quantum_inferno.plot_templates import plot_base as pbase, plot_templates as pnl

# RedVox RedPandas and related RedVox modules
import examples.skyfall.lib.skyfall_dw as sf_dw
import redpandas.redpd_gravity as rpd_grav
import redpandas.redpd_preprocess as rpd_prep

# Configuration files
from examples.skyfall.skyfall_config_file import skyfall_config
from redpandas.redpd_config import DataLoadMethod


def main():
    """
    RedVox RedPandas time-domain representation of API900 data.
    """
    # Label columns in dataframe
    station_label: str = "station_id"

    # Audio columns
    audio_data_label: str = "audio_wf"
    audio_epoch_s_label: str = "audio_epoch_s"
    audio_fs_label: str = "audio_sample_rate_nominal_hz"

    # Accelerometer columns
    accelerometer_data_raw_label: str = "accelerometer_wf_raw"
    accelerometer_data_highpass_label: str = "accelerometer_wf_highpass"
    accelerometer_epoch_s_label: str = "accelerometer_epoch_s"
    accelerometer_fs_label: str = "accelerometer_sample_rate_hz"

    # Load data options
    df_skyfall_data = sf_dw.dw_main(skyfall_config.tdr_load_method)

    # Start of building plots
    print("\nInitiating time-domain representation of Skyfall:")
    for station in df_skyfall_data.index:
        station_id_str = df_skyfall_data[station_label][station]  # Get the station id

        if audio_data_label and audio_fs_label in df_skyfall_data.columns:
            print('mic_sample_rate_hz: ', df_skyfall_data[audio_fs_label][station])
            print('mic_epoch_s_0: ', df_skyfall_data[audio_epoch_s_label][station][0])

            # Frame to mic start and end and plot
            event_reference_time_epoch_s = df_skyfall_data[audio_epoch_s_label][station][0]
        else:
            raise ValueError("Missing Audio label in the data.")

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

            # Plot 3c acceleration raw waveforms
            pnl_wfb = pbase.WaveformPlotBase(station_id=station_id_str,
                                             figure_title=skyfall_config.event_name + ": Accelerometer raw",
                                             figure_title_show=False, # for press
                                             start_time_epoch=event_reference_time_epoch_s,
                                             label_panel_show=True,  # for press
                                             labels_fontweight='bold')
            pnl_a = pbase.WaveformPanel(sig=df_skyfall_data[accelerometer_data_raw_label][station][0],
                                        time=df_skyfall_data[accelerometer_epoch_s_label][station],
                                        label="Acc X, m/$s^2$")
            pnl_b = pbase.WaveformPanel(sig=df_skyfall_data[accelerometer_data_raw_label][station][1],
                                        time=df_skyfall_data[accelerometer_epoch_s_label][station],
                                        label="Acc Y, m/$s^2$")
            pnl_c = pbase.WaveformPanel(sig=df_skyfall_data[accelerometer_data_raw_label][station][2],
                                        time=df_skyfall_data[accelerometer_epoch_s_label][station],
                                        label="Acc Z, m/$s^2$")
            fig_3c_acc_raw = pnl.plot_wf_3_vert(pnl_wfb, pnl_a, pnl_b, pnl_c)

            # get time and sample rate
            accelerometer_time_s = df_skyfall_data[accelerometer_epoch_s_label][station]
            accelerometer_sample_rate = 1 / np.mean(np.diff(accelerometer_time_s))

            # get gravity and linear acceleration
            gravity_x, linear_x = rpd_grav.get_gravity_and_linear_acceleration(
                accelerometer=df_skyfall_data[accelerometer_data_raw_label][station][0],
                sensor_sample_rate_hz=accelerometer_sample_rate,
                low_pass_sample_rate_hz=2)

            gravity_y, linear_y = rpd_grav.get_gravity_and_linear_acceleration(
                accelerometer=df_skyfall_data[accelerometer_data_raw_label][station][1],
                sensor_sample_rate_hz=accelerometer_sample_rate,
                low_pass_sample_rate_hz=2)

            gravity_z, linear_z = rpd_grav.get_gravity_and_linear_acceleration(
                accelerometer=df_skyfall_data[accelerometer_data_raw_label][station][2],
                sensor_sample_rate_hz=accelerometer_sample_rate,
                low_pass_sample_rate_hz=2)

            # Plot 3c acceleration gravity waveforms
            pnl_wfb.figure_title = skyfall_config.event_name + ": Gravity"
            pnl_a.sig = gravity_x
            pnl_a.label = "LP Acc X, m/$s^2$"
            pnl_b.sig = gravity_y
            pnl_b.label = "LP Acc Y, m/$s^2$"
            pnl_c.sig = gravity_z
            pnl_c.label = "LP Acc Z, m/$s^2$"
            fig_3c_grv_raw = pnl.plot_wf_3_vert(pnl_wfb, pnl_a, pnl_b, pnl_c)

            # Plot 3c acceleration linear waveforms
            pnl_wfb.figure_title=skyfall_config.event_name + ": Linear Acceleration"
            pnl_a.sig = linear_x
            pnl_a.label = "Linear Acc Z, m/$s^2$"
            pnl_b.sig = linear_y
            pnl_b.label = "Linear Acc Y, m/$s^2$"
            pnl_c.sig = linear_z
            pnl_c.label = "Linear Acc Z, m/$s^2$"
            fig_3c_lin_raw = pnl.plot_wf_3_vert(pnl_wfb, pnl_a, pnl_b, pnl_c)

        plt.show()


if __name__ == "__main__":
    main()
