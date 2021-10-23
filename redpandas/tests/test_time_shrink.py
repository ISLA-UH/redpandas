"""
Wiggles and mesh flow for 13+ stations reports. Only Audio sensor.
"""
from redvox.common.data_window import DataWindow
from redpandas.redpd_df import redpd_dataframe
from redpandas.redpd_filter import decimate_signal_pandas_audio_rdvx
from redpandas.redpd_tfr import tfr_bits_panda
from redpandas.redpd_plot.mesh import plot_mesh_pandas
import redvox.common.date_time_utils as dt_utils
from redpandas.redpd_plot.wiggles import plot_wiggles_pandas


if __name__ == "__main__":

    INPUT_DIR = "/Users/meritxell/Desktop/skyfall_dummy_test"  # 13 stations
    # INPUT_DIR = "/Users/meritxell/Documents/api_m_pipeline_tests/20210507_800Hz_weekend"  # 40 stations

    # dw = DataWindow(input_dir=INPUT_DIR,
    #                 start_datetime=dt_utils.datetime_from_epoch_seconds_utc(1603806314),
    #                 end_datetime=dt_utils.datetime_from_epoch_seconds_utc(1603806314+60))

    # Make DataWindow
    dw = DataWindow(input_dir=INPUT_DIR)
    # Make pandas
    df0 = redpd_dataframe(input_dw=dw)

    print(df0.columns)

    # Adds columns "decimated_audio_wf", "decimated_audio_sample_rate_hz" and "decimated_audio_epoch_s" to df
    df0 = decimate_signal_pandas_audio_rdvx(df=df0,
                                            sig_wf_label="audio_wf",
                                            sig_timestamps_label="audio_epoch_s",
                                            sample_rate_hz_label="audio_sample_rate_nominal_hz")

    new_start_time = 1
    new_end_time = 2
    for row in df0.index:
        if "epoch" in df0.column:
            # find closest timestamp to new time
            timestamps = df0
            # find data equivalent
            # cut to new time window and data window through each sensor
