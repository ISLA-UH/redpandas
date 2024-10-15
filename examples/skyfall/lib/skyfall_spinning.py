"""
SkyFall spinning
"""
# Python libraries
import matplotlib.pyplot as plt
import numpy as np

# RedVox RedPandas and related RedVox modules
import examples.skyfall.lib.skyfall_dw as sf_dw
import redpandas.redpd_preprocess as rpd_prep
from quantum_inferno.plot_templates import plot_base as pbase, plot_templates as pnl

# Configuration files
from redpandas.redpd_config import DataLoadMethod
from examples.skyfall.skyfall_config_file import skyfall_config


def main():
    """
    RedVox RedPandas time-domain representation of API900 data. Example: Skyfall.
    """
    # Label columns in dataframe
    station_label: str = "station_id"

    # Audio columns
    audio_data_label: str = "audio_wf"
    audio_epoch_s_label: str = "audio_epoch_s"
    audio_fs_label: str = "audio_sample_rate_nominal_hz"

    # Gyroscope columns
    gyroscope_data_raw_label: str = "gyroscope_wf_raw"
    gyroscope_data_highpass_label: str = "gyroscope_wf_highpass"
    gyroscope_epoch_s_label: str = "gyroscope_epoch_s"
    gyroscope_fs_label: str = "gyroscope_sample_rate_hz"

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
            print('gyroscope max rotation rate, rad/s:',
                  np.max(df_skyfall_data[gyroscope_data_raw_label][station][2]))
            print('gyroscope max rotation rate, Hz:',
                  np.max(df_skyfall_data[gyroscope_data_raw_label][station][2] / (2*np.pi)))
            # Plot 3c raw gyroscope waveforms
            pnl_wfb = pbase.WaveformPlotBase(station_id=station_id_str,
                                             figure_title=skyfall_config.event_name + ": Gyroscope raw",
                                             figure_title_show=False,
                                             start_time_epoch=event_reference_time_epoch_s,
                                             label_panel_show=True,  # for press
                                             labels_fontweight='bold')
            pnl_a = pbase.WaveformPanel(sig=df_skyfall_data[gyroscope_data_raw_label][station][0] / (2*np.pi),
                                        time=df_skyfall_data[gyroscope_epoch_s_label][station],
                                        label="Gyr X, rotation/s")
            pnl_b = pbase.WaveformPanel(sig=df_skyfall_data[gyroscope_data_raw_label][station][1] / (2*np.pi),
                                        time=df_skyfall_data[gyroscope_epoch_s_label][station],
                                        label="Gyr Y, rotation/s")
            pnl_c = pbase.WaveformPanel(sig=df_skyfall_data[gyroscope_data_raw_label][station][2] / (2*np.pi),
                                        time=df_skyfall_data[gyroscope_epoch_s_label][station],
                                        label="Gyr Z, rotation/s")
            fig_3c_gyr_raw = pnl.plot_wf_3_vert(pnl_wfb, pnl_a, pnl_b, pnl_c)

        plt.show()


if __name__ == "__main__":
    main()
