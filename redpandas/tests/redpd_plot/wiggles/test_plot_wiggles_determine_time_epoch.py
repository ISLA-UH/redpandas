import unittest
import pandas as pd
import numpy as np
import redpandas.redpd_plot.wiggles as rpd_wiggles


class TestDetermineTimeEpochOrigin(unittest.TestCase):

    def setUp(self) -> None:
        # Create audio 1
        self.start_time_audio1 = 1.
        self.end_time1 = 10.
        self.sample_rate_audio1 = 100.
        self.signal_time_audio1 = np.arange(self.start_time_audio1, self.end_time1, 1/self.sample_rate_audio1)

        # Create barometer 1
        self.start_time_barometer1 = 2.
        self.sample_rate_barometer1 = 31.
        self.signal_time_barometer1 = np.arange(self.start_time_barometer1, self.end_time1,
                                                1/self.sample_rate_barometer1)

        # Create audio 2
        self.start_time_audio2 = 0.5
        self.end_time2 = 10.
        self.sample_rate_audio2 = 100.
        self.signal_time_audio2 = np.arange(self.start_time_audio2, self.end_time2, 1/self.sample_rate_audio2)

        # Create barometer 2
        self.start_time_barometer2 = 0.2
        self.sample_rate_barometer2 = 31.
        self.signal_time_barometer2 = np.arange(self.start_time_barometer2, self.end_time2,
                                                1/self.sample_rate_barometer2)

        # Create df
        self.dict_to_df = {0: {"station_id": "1234567890",
                               "audio_sensor_name": "synch_audio",
                               "audio_sample_rate_nominal_hz": self.sample_rate_audio1,
                               "audio_epoch_s": self.signal_time_audio1,
                               "barometer_sensor_name": "synch_barometer",
                               "barometer_sample_rate_nominal_hz": self.sample_rate_barometer1,
                               "barometer_epoch_s": self.signal_time_barometer1},
                           1: {"station_id": "2345678901",   # Add another station
                               "audio_sensor_name": "synch_audio",
                               "audio_sample_rate_nominal_hz": self.sample_rate_audio2,
                               "audio_epoch_s": self.signal_time_audio2,
                               "barometer_sensor_name": "synch_barometer",
                               "barometer_sample_rate_nominal_hz": self.sample_rate_barometer2,
                               "barometer_epoch_s": self.signal_time_barometer2}}

        self.df_data = pd.DataFrame(self.dict_to_df).T

    def test_result_with_correct_timestamp_input_all_stations(self):
        self.time_epoch_origin = rpd_wiggles.determine_time_epoch_origin(df=self.df_data,
                                                                         sig_timestamps_label=["audio_epoch_s"],
                                                                         station_id_str=None,
                                                                         sig_id_label="station_id")
        self.assertEqual(self.time_epoch_origin, 0.5)

    def test_result_with_correct_string_timestamp_input_all_stations(self):
        self.time_epoch_origin = rpd_wiggles.determine_time_epoch_origin(df=self.df_data,
                                                                         sig_timestamps_label="audio_epoch_s",
                                                                         station_id_str=None,
                                                                         sig_id_label="station_id")
        self.assertEqual(self.time_epoch_origin, 0.5)

    def test_result_with_incorrect_timestamp_input_all_stations(self):
        with self.assertRaises(ValueError): rpd_wiggles.determine_time_epoch_origin(df=self.df_data,
                                                                                    sig_timestamps_label=["audio_s"],
                                                                                    station_id_str=None,
                                                                                    sig_id_label="station_id")

    def test_result_with_multiple_timestamp_input_all_stations(self):
        self.time_epoch_origin = rpd_wiggles.determine_time_epoch_origin(df=self.df_data,
                                                                         sig_timestamps_label=["audio_epoch_s",
                                                                                               "barometer_epoch_s"],
                                                                         station_id_str=None,
                                                                         sig_id_label="station_id")
        self.assertEqual(self.time_epoch_origin, 0.2)

    def test_result_with_multiple_timestamp_input_correct_one_station(self):
        self.time_epoch_origin = rpd_wiggles.determine_time_epoch_origin(df=self.df_data,
                                                                         sig_timestamps_label=["audio_epoch_s",
                                                                                               "barometer_epoch_s"],
                                                                         station_id_str="1234567890",
                                                                         sig_id_label="station_id")
        self.assertEqual(self.time_epoch_origin, 1.0)

    def tearDown(self):
        self.start_time_audio = None
        self.end_time = None
        self.sample_rate_audio = None
        self.signal_time_audio = None
        self.start_time_barometer = None
        self.sample_rate_barometer = None
        self.signal_time_barometer = None
        self.dict_to_df = None
        self.df_data = None
        self.time_epoch_origin = None


if __name__ == '__main__':
    unittest.main()
