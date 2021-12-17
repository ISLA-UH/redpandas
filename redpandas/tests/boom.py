"""
TFR for Space X sonic boom on 20210918
"""
from redpandas.redpd_df import redpd_dataframe, export_df_to_parquet
from redpandas.redpd_plot.wiggles import plot_wiggles_pandas
import redpandas.redpd_tfr as rpd_tfr
from redpandas.redpd_plot.mesh import plot_mesh_pandas
import redpandas.redpd_dq as rpd_dq
from redpandas.redpd_preprocess import df_column_unflatten

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
                                     "1637610015",
                                      "2551278155",
                                      "1637610012",
                                      "1637610015",
                                      "1637610014",
                                      "872266036"
                                              ],
                                 start_datetime=dt.datetime_from_epoch_seconds_utc(1632006000),
                                 end_datetime=dt.datetime_from_epoch_seconds_utc(1632006330))

    rdvx_data: DataWindow = DataWindow(event_name="dw",
                                       # event_origin=DWOrigin,
                                       config=DWAConfig,
                                       out_dir="/Users/meritxell/Desktop/test",
                                       out_type="lz4",
                                       debug=True)
    # path = rdvx_dataa.save()
    # print(path)
    # station = rdvx_data.get_station("0872266036")[0]
    # print(station.health_sensor().get_data_channel("network_type"))
    #
    # exit()

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

    # # Create new columns with shape tuple for future unflattening/reshaping
    # df0[[f'station_id_ndim']] = df0[[f'station_id']].applymap(np.shape)
    # # Change tuples to 1D np.array to save it to parquet
    # df0[[f'station_id_ndim']] = df0[[f'station_id_ndim']].applymap(np.asarray)
    # # Flatten each row in wf columns
    # df0[[f'station_id']] = df0[[f'station_id']].applymap(np.ravel)
    #
    # print(df0[f'station_id'])
    # print(df0['station_id_ndim'])
    #
    # exit()


    # # Check audio wiggles ok
    # fig_wiggles = plot_wiggles_pandas(df=df0,
    #                                   sig_wf_label=["audio_wf"],
    #                                   sig_timestamps_label=["audio_epoch_s"],
    #                                   sig_id_label="station_id",
    #                                   station_id_str="1637610078",
    #                                   show_figure=False)

    # TFR
    sensors_to_tfr = ["audio", 'accelerometer']  #"barometer", "accelerometer", "magnetometer", "gyroscope"]
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
                         output_dir_pqt="/Users/meritxell/Desktop",
                         tfr_column_label=None,
                         # tfr_column_label=["audio_tfr_bits", "accelerometer_tfr_bits"],
                         # tfr_frequency_label=["audio_tfr_frequency_hz", "accelerometer_tfr_frequency_hz"],
                         tfr_frequency_label=None,
                         # tfr_time_label=["audio_tfr_time_s", "accelerometer_tfr_time_s"]
                         tfr_time_label=None
                         )

    # df_sensors = pd.read_parquet("/Users/meritxell/Desktop/Redvox_df.parquet")
    #
    # df_column_unflatten(df=df_sensors,
    #                     col_wf_label="accelerometer_tfr_bits",
    #                     col_ndim_label="accelerometer_tfr_bits_ndim")

    exit()
    print(rdvx_data.event_name)
    print(rdvx_data)

    rpd_dq.mic_sync(rdvx_data)
    rpd_dq.station_channel_timing(rdvx_data)
    rpd_dq.station_metadata(rdvx_data)

    fig_mesh = plot_mesh_pandas(df=df0,
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
    for station in df0.index:
        if df0["station_id"][station] == "0872266036":
            pnl.plot_wf_mesh_vert(redvox_id=df0["station_id"][station],
                                  wf_panel_2_sig=df0["audio_wf"][station],
                                  wf_panel_2_time=df0["audio_epoch_s"][station],
                                  mesh_time=df0["audio_tfr_time_s"][station],
                                  mesh_frequency=df0["audio_tfr_frequency_hz"][station],
                                  mesh_panel_0_tfr=df0["audio_tfr_bits"][station],
                                  start_time_epoch=df0["audio_epoch_s"][0][0],
                                  frequency_scaling="log",
                                  frequency_hz_ymax=320,
                                  frequency_hz_ymin=20,
                                  mesh_panel_0_colormap_scaling="range",
                                  mesh_panel_0_color_range=12.,
                                  wf_panel_2_units="Audio",
                                  figure_title=f"STFT")

        else:
            pnl.plot_wf_mesh_vert(redvox_id=df0["station_id"][station],
                                  wf_panel_2_sig=df0["audio_wf"][station],
                                  wf_panel_2_time=df0["audio_epoch_s"][station],
                                  mesh_time=df0["audio_tfr_time_s"][station],
                                  mesh_frequency=df0["audio_tfr_frequency_hz"][station],
                                  mesh_panel_0_tfr=df0["audio_tfr_bits"][station],
                                  start_time_epoch=df0["audio_epoch_s"][0][0],
                                  frequency_scaling="log",
                                  mesh_panel_0_colormap_scaling="range",
                                  mesh_panel_0_color_range=16.,
                                  wf_panel_2_units="Audio",
                                  figure_title=f"STFT")

    # Check barometer wiggles
    fig_wiggles_bar = plot_wiggles_pandas(df=df0,
                                          sig_wf_label=["barometer_wf_highpass"],
                                          sig_timestamps_label=["barometer_epoch_s"],
                                          sig_id_label="station_id",
                                          show_figure=False)
    # Check accelerometer wiggles
    fig_wiggles_acc = plot_wiggles_pandas(df=df0,
                                          sig_wf_label=["accelerometer_wf_highpass"],
                                          sig_timestamps_label=["accelerometer_epoch_s"],
                                          sig_id_label="station_id",
                                          show_figure=False)

    for station in df0.index:
        if type(df0["accelerometer_wf_highpass"][station]) != float:
            pnl.plot_wf_wf_wf_vert(redvox_id=df0["station_id"][station],
                                   wf_panel_2_sig=df0["audio_wf"][station],
                                   wf_panel_2_time=df0["audio_epoch_s"][station],
                                   wf_panel_1_sig=df0["barometer_wf_highpass"][station][0],
                                   wf_panel_1_time=df0["barometer_epoch_s"][station],
                                   wf_panel_0_sig=df0["accelerometer_wf_highpass"][station][2],
                                   wf_panel_0_time=df0["accelerometer_epoch_s"][station],
                                   start_time_epoch=df0["audio_epoch_s"][station][0],
                                   wf_panel_2_units="Mic, Norm",
                                   wf_panel_1_units="Bar hp, kPa",
                                   wf_panel_0_units="Acc Z hp, m/$s^2$")

    plt.show()