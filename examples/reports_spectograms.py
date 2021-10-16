"""
Wiggles and mesh flow for 13+ stations reports. Only Audio sensor.
"""
from redvox.common.data_window import DataWindow
from redpandas.redpd_df import redpd_dataframe
from redpandas.redpd_filter import decimate_signal_pandas, decimate_individual_station
from redpandas.redpd_tfr import tfr_bits_panda
from redpandas.redpd_plot.mesh import plot_mesh_pandas
import redvox.common.date_time_utils as dt_utils
from redpandas.redpd_plot.wiggles import plot_wiggles_pandas

import pandas as pd


def decimate_signal_pandas_audio_rdvx(df: pd.DataFrame,
                                      sig_wf_label: str = "audio_wf",
                                      sig_timestamps_label: str = "audio_epoch_s",
                                      sample_rate_hz_label: str = "audio_sample_rate_nominal_hz"):
    """
    Decimates signal data to 8kHz. Makes columns "decimated_audio_wf", "decimated_audio_epoch_s" and
    "decimated_audio_sample_rate_hz"

    :param df: input pandas dataframe
    :param sig_wf_label: Default is "audio_wf"
    :param sig_timestamps_label: Default is "audio_epoch_s"
    :param sample_rate_hz_label: Default is "audio_sample_rate_nominal_hz"

    :return: original df with new columns with decimated data
    """

    list_all_decimated_data = []
    list_all_decimated_timestamps = []
    list_all_decimated_sample_rate_hz = []
    for row in df.index:
        if df[sample_rate_hz_label][row] == 48000:
            decimated_timestamp, decimated_data = decimate_individual_station(downsampling_factor=6,
                                                                              filter_order=8,
                                                                              sig_epoch_s=df[sig_timestamps_label][row],
                                                                              sig_wf=df[sig_wf_label][row],
                                                                              sample_rate_hz=df[sample_rate_hz_label][row])
            new_sample_rate_hz = df[sample_rate_hz_label][row]/6

        elif df[sample_rate_hz_label][row] == 16000:
            decimated_timestamp, decimated_data = decimate_individual_station(downsampling_factor=2,
                                                                              filter_order=8,
                                                                              sig_epoch_s=df[sig_timestamps_label][row],
                                                                              sig_wf=df[sig_wf_label][row],
                                                                              sample_rate_hz=df[sample_rate_hz_label][row])
            new_sample_rate_hz = df[sample_rate_hz_label][row]/2

        else:
            decimated_timestamp = df[sig_timestamps_label][row]
            decimated_data = df[sig_wf_label][row]
            new_sample_rate_hz = df[sample_rate_hz_label][row]

        list_all_decimated_data.append(decimated_data)
        list_all_decimated_timestamps.append(decimated_timestamp)
        list_all_decimated_sample_rate_hz.append(new_sample_rate_hz)

    df["decimated_audio_wf"] = list_all_decimated_data
    df["decimaated_audio_epoch_s"] = list_all_decimated_timestamps
    df["decimated_audio_sample_rate_hz"] = list_all_decimated_sample_rate_hz

    return df


if __name__ == "__main__":

    # INPUT_DIR = "/Users/meritxell/Desktop/skyfall_dummy_test"  # 13 stations
    # INPUT_DIR = "/Users/meritxell/Documents/api_m_pipeline_tests/20210507_800Hz_weekend"  # 40 stations
    INPUT_DIR = "/Users/meritxell/Documents/api_m_pipeline_tests/20210416_SDK_Image"  # 48kHz
    # dw = DataWindow(input_dir=INPUT_DIR,
    #                 start_datetime=dt_utils.datetime_from_epoch_seconds_utc(1603806314),
    #                 end_datetime=dt_utils.datetime_from_epoch_seconds_utc(1603806314+60))
    # dw = DataWindow(input_dir=INPUT_DIR)
    # dw = DataWindow.deserialize("/Users/meritxell/Documents/api_m_pipeline_tests/"
    #                             "20210416_SDK_Image/SDK_Image_20210416.pickle")
    # df0 = redpd_dataframe(input_dw=dw)
    df0 = pd.read_parquet(f"{INPUT_DIR}/SDK_Image_20210416_df.parquet")
    print(df0.columns)
    # df0 = decimate_signal_pandas(df=df0,
    #                              downsample_frequency_hz=8000,
    #                              sig_wf_label="audio_wf",
    #                              sig_id_label="station_id",
    #                              sig_timestamps_label="audio_epoch_s",
    #                              sample_rate_hz_label="audio_sample_rate_nominal_hz",
    #                              new_column_label_decimated_sig="decimated_audio_wf",
    #                              new_column_label_decimated_sample_rate_hz="decimated_audio_sample_rate_hz",
    #                              new_column_label_decimated_sig_timestamps="decimated_audio_epoch_s",
    #                              verbose=True)

    df0 = decimate_signal_pandas_audio_rdvx(df=df0,
                                            sig_wf_label="audio_wf",
                                            sig_timestamps_label="audio_epoch_s",
                                            sample_rate_hz_label="audio_sample_rate_nominal_hz")

    df0 = tfr_bits_panda(df=df0,
                         sig_wf_label="decimated_audio_wf",
                         # sig_wf_label="audio_wf",
                         sig_sample_rate_label="decimated_audio_sample_rate_hz",
                         # sig_sample_rate_label="audio_sample_rate_nominal_hz",
                         order_number_input=12,
                         tfr_type='stft',
                         new_column_tfr_bits="audio_tfr_bits",
                         new_column_tfr_frequency_hz="audio_tfr_frequency_hz",
                         new_column_tfr_time_s="audio_tfr_time_s")

    # Split dataframe into 13 stations long dataframes
    list_of_dfs = [df0.loc[station:station+13-1, :] for station in range(0, len(df0), 13)]

    # Plot wiggles and mesh for each sub-dataframe
    # I did list comprehension for storage assuming figs will be saved and displayed at another time
    # TODO: discuss mesh parameters w/ Milton
    fig_wiggles_list = [plot_wiggles_pandas(df=df_in_list, show_figure=False) for df_in_list in list_of_dfs]
    fig_mesh_list = [plot_mesh_pandas(df=df_in_list,
                                      mesh_time_label="audio_tfr_time_s",
                                      mesh_frequency_label="audio_tfr_frequency_hz",
                                      mesh_tfr_label="audio_tfr_bits",
                                      t0_sig_epoch_s=df0["audio_epoch_s"][0][0],
                                      sig_id_label="station_id",
                                      frequency_scaling="log",
                                      common_colorbar=False,
                                      mesh_color_scaling="range",
                                      mesh_color_range=15,
                                      ytick_values_show=True, show_figure=True) for df_in_list in list_of_dfs]

    print(f"Sanity check-wiggles figs:{len(fig_wiggles_list)}, mesh figs:{len(fig_mesh_list)}")
