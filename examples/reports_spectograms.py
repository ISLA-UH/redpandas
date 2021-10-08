"""

"""
from redvox.common.data_window import DataWindow
from redpandas.redpd_df import redpd_dataframe
from redpandas.redpd_filter import decimate_signal_pandas
from redpandas.redpd_tfr import tfr_bits_panda
from redpandas.redpd_plot.mesh import plot_mesh_pandas
import redvox.common.date_time_utils as dt_utils

if __name__ == "__main__":
    # TODO:
    #  1. fit ids inside graph - DONE
    #  2. deal with range/scaling maybe make it such that input can be only 1 value
    #  3. fix yticks scaling spacing
    #  4. consider putting limit to 13 signals at one time - think about how to limit it

    INPUT_DIR = "/Users/meritxell/Desktop/skyfall_dummy_test"

    dw = DataWindow(input_dir=INPUT_DIR,
                    start_datetime=dt_utils.datetime_from_epoch_seconds_utc(1603806314),
                    end_datetime=dt_utils.datetime_from_epoch_seconds_utc(1603806314+60*3))
    df0 = redpd_dataframe(input_dw=dw)

    df0 = decimate_signal_pandas(df=df0,
                                 downsample_frequency_hz=800,
                                 sig_wf_label="audio_wf",
                                 sig_id_label="station_id",
                                 sig_timestamps_label="audio_epoch_s",
                                 sample_rate_hz_label="audio_sample_rate_nominal_hz",
                                 new_column_label_decimated_sig="decimated_audio_wf",
                                 new_column_label_decimated_sample_rate_hz="decimated_audio_sample_rate_hz",
                                 new_column_label_decimated_sig_timestamps="decimated_audio_epoch_s")

    df0 = tfr_bits_panda(df=df0,
                         sig_wf_label="decimated_audio_wf",
                         sig_sample_rate_label="decimated_audio_sample_rate_hz",
                         order_number_input=12,
                         tfr_type='stft',
                         new_column_tfr_bits="audio_tfr_bits",
                         new_column_tfr_frequency_hz="audio_tfr_frequency_hz",
                         new_column_tfr_time_s="audio_tfr_time_s")

    fig = plot_mesh_pandas(df=df0,
                           mesh_time_label="audio_tfr_time_s",
                           mesh_frequency_label="audio_tfr_frequency_hz",
                           mesh_tfr_label="audio_tfr_bits",
                           sig_id_label="station_id",
                           frequency_hz_ymax=400,
                           common_colorbar=True,
                           mesh_color_scaling=["range", "range", "range", "range", "range",
                                               "range", "range", "range", "range", "range",
                                               "range", "range", "range"],
                           mesh_color_range=[15,15,15,15,15,
                                             15,15,15,15,15,
                                             15,15,15],
                           ytick_values_show=True)