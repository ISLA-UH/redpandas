"""
TFR for Space X sonic boom on 20210918
"""
from redpandas.redpd_df import redpd_dataframe, export_df_to_parquet
from redpandas.redpd_plot.wiggles import plot_wiggles_pandas
import redpandas.redpd_tfr as rpd_tfr
from redpandas.redpd_plot.mesh import plot_mesh_pandas
import redpandas.redpd_dq as rpd_dq
from redpandas.redpd_preprocess import df_column_unflatten, df_unflatten

# from redvox.common.data_window import DataWindow
from redvox.common.data_window import DataWindow, EventOrigin
from redvox.common.data_window import DataWindowConfig
import redvox.common.date_time_utils as dt

from libquantum.plot_templates import plot_time_frequency_reps as pnl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


INPUT_DIR = "/Users/meritxell/Documents/20210916_spacex/boom"
# Note: Need RedPandas v1.2.12 to run

if __name__ == '__main__':

    DWAConfig = DataWindowConfig(input_dir=INPUT_DIR,
                                 station_ids=[
                                     "1637610001",
                                      "2551278155",
                                      "1637610012",
                                      "1637610015",
                                      "1637610014",
                                      "872266036"
                                              ],
                                 start_datetime=dt.datetime_from_epoch_seconds_utc(1632006000),
                                 end_datetime=dt.datetime_from_epoch_seconds_utc(1632006330))

    rdvx_data: DataWindow = DataWindow(event_name="dw",
                                       config=DWAConfig,
                                       out_dir="/Users/meritxell/Desktop/test",
                                       out_type="lz4",
                                       debug=True)

    df0 = redpd_dataframe(input_dw=rdvx_data,
                          sensor_labels=["audio",
                                         "barometer",
                                         "accelerometer",
                                         "gyroscope",
                                         "magnetometer",
                                         "clock",
                                         "synchronization",
                                         "location",
                                         "health"
                                         ])

    # Check audio wiggles ok
    fig_wiggles = plot_wiggles_pandas(df=df0,
                                      sig_wf_label=["audio_wf"],
                                      sig_timestamps_label=["audio_epoch_s"],
                                      sig_id_label="station_id",
                                      show_figure=False)

    # TFR
    sensors_to_tfr = ["audio", 'accelerometer']  # "barometer", "accelerometer", "magnetometer", "gyroscope"]
    for sensor in sensors_to_tfr:
        if sensor == "audio":
            sensor_wf = "audio_wf"
            sensor_fs = "audio_sample_rate_nominal_hz"
        else:
            sensor_wf = f"{sensor}_wf_highpass"
            sensor_fs = f"{sensor}_sample_rate_hz"

        df0 = rpd_tfr.tfr_bits_panda(df=df0,
                                     sig_wf_label=sensor_wf,
                                     sig_sample_rate_label=sensor_fs,
                                     order_number_input=24,
                                     tfr_type='stft',
                                     new_column_tfr_bits=f"{sensor}_tfr_bits",
                                     new_column_tfr_frequency_hz=f"{sensor}_tfr_frequency_hz",
                                     new_column_tfr_time_s=f"{sensor}_tfr_time_s")

    export_df_to_parquet(df=df0,
                         output_dir_pqt="/Users/meritxell/Desktop")

    df_sensors = pd.read_parquet("/Users/meritxell/Desktop/Redvox_df.parquet")

    # df_column_unflatten(df=df_sensors,
    #                     col_wf_label="accelerometer_tfr_bits",
    #                     col_ndim_label="accelerometer_tfr_bits_ndim")

    df_unflatten(df_sensors)

    # rpd_dq.mic_sync(rdvx_data)
    # rpd_dq.station_channel_timing(rdvx_data)
    # rpd_dq.station_metadata(rdvx_data)

    fig_mesh = plot_mesh_pandas(df=df_sensors,
                                mesh_time_label="audio_tfr_time_s",
                                mesh_frequency_label="audio_tfr_frequency_hz",
                                mesh_tfr_label="audio_tfr_bits",
                                sig_id_label="station_id",
                                t0_sig_epoch_s=rdvx_data.get_station("1637610014")[0].audio_sensor().first_data_timestamp()/1e6,
                                fig_title="STFT for audio_wf",
                                frequency_scaling="log",
                                # frequency_hz_ymax=320,
                                common_colorbar=False,
                                mesh_color_scaling=["range", "range", "range", "range", "range"],

                                mesh_color_range=[16.0, 16.0, 16.0, 16.0, 16.0],
                                show_figure=True,
                                ytick_values_show=True)
    for station in df_sensors.index:
        if df_sensors["station_id"][station] == "0872266036":
            pnl.plot_wf_mesh_vert(redvox_id=df_sensors["station_id"][station],
                                  wf_panel_2_sig=df_sensors["audio_wf"][station],
                                  wf_panel_2_time=df_sensors["audio_epoch_s"][station],
                                  mesh_time=df_sensors["audio_tfr_time_s"][station],
                                  mesh_frequency=df_sensors["audio_tfr_frequency_hz"][station],
                                  mesh_panel_0_tfr=df_sensors["audio_tfr_bits"][station],
                                  start_time_epoch=df_sensors["audio_epoch_s"][0][0],
                                  frequency_scaling="log",
                                  frequency_hz_ymax=320,
                                  frequency_hz_ymin=20,
                                  mesh_panel_0_colormap_scaling="range",
                                  mesh_panel_0_color_range=12.,
                                  wf_panel_2_units="Audio",
                                  figure_title=f"STFT")

        else:
            pnl.plot_wf_mesh_vert(redvox_id=df_sensors["station_id"][station],
                                  wf_panel_2_sig=df_sensors["audio_wf"][station],
                                  wf_panel_2_time=df_sensors["audio_epoch_s"][station],
                                  mesh_time=df_sensors["audio_tfr_time_s"][station],
                                  mesh_frequency=df_sensors["audio_tfr_frequency_hz"][station],
                                  mesh_panel_0_tfr=df_sensors["audio_tfr_bits"][station],
                                  start_time_epoch=df_sensors["audio_epoch_s"][0][0],
                                  frequency_scaling="log",
                                  mesh_panel_0_colormap_scaling="range",
                                  mesh_panel_0_color_range=16.,
                                  wf_panel_2_units="Audio",
                                  figure_title=f"STFT")

    # Check barometer wiggles
    fig_wiggles_bar = plot_wiggles_pandas(df=df_sensors,
                                          sig_wf_label=["barometer_wf_highpass"],
                                          sig_timestamps_label=["barometer_epoch_s"],
                                          sig_id_label="station_id",
                                          show_figure=True)
    # Check accelerometer wiggles
    fig_wiggles_acc = plot_wiggles_pandas(df=df_sensors,
                                          sig_wf_label=["accelerometer_wf_highpass"],
                                          sig_timestamps_label=["accelerometer_epoch_s"],
                                          sig_id_label="station_id",
                                          show_figure=True)
    plt.show()
