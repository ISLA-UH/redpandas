import unittest
import pandas as pd
import numpy as np
import redpandas.redpd_plot.wiggles as rpd_wiggles


class TestNumWiggles(unittest.TestCase):
    def setUp(self) -> None:
        # Create audio
        self.start_time = 0
        self.end_time = 10
        self.sample_rate_audio = 100
        self.signal_time_audio = np.arange(self.start_time, self.end_time, 1/self.sample_rate_audio)
        self.frequency = 3
        self.amplitude = 1
        self.sinewave_audio = self.amplitude * np.sin(2 * np.pi * self.frequency * self.signal_time_audio)

        # Create barometer
        self.sample_rate_barometer = 31
        self.signal_time_barometer = np.arange(self.start_time, self.end_time, 1/self.sample_rate_barometer)
        self.sinewave_barometer_base = self.amplitude * np.sin(2 * np.pi * self.frequency * self.signal_time_barometer)
        self.sinewave_barometer = self.sinewave_barometer_base.reshape((1, len(self.sinewave_barometer_base)))

        # Create accelerometer
        self.sample_rate_acc = 30
        self.signal_time_acc = np.arange(self.start_time, self.end_time, 1/(self.sample_rate_acc/3))
        self.length_for_signal = np.arange(self.start_time, self.end_time, 1/self.sample_rate_acc)
        self.sinewave_acc_base = self.amplitude * np.sin(2 * np.pi * self.frequency * self.length_for_signal)
        self.points_per_row = int(len(self.sinewave_acc_base)/3)
        self.sinewave_acc = self.sinewave_acc_base.reshape((3, self.points_per_row))

        # Create df
        self.dict_to_df = {0: {"station_id": "1234567890",
                               "audio_sensor_name": "synch_audio",
                               "audio_sample_rate_nominal_hz": self.sample_rate_audio,
                               "audio_epoch_s": self.signal_time_audio,
                               "audio_wf": self.sinewave_audio,
                               "barometer_sensor_name": "synch_barometer",
                               "barometer_sample_rate_nominal_hz": self.sample_rate_barometer,
                               "barometer_epoch_s": self.signal_time_barometer,
                               "barometer_wf_raw": self.sinewave_barometer,
                               "accelerometer_epoch_s": self.signal_time_acc,
                               "accelerometer_wf_raw": self.sinewave_acc},
                           1: {"station_id": "2345678901",   # Add another station
                               "audio_sensor_name": "synch_audio",
                               "audio_sample_rate_nominal_hz": self.sample_rate_audio,
                               "audio_epoch_s": self.signal_time_audio,
                               "audio_wf": self.sinewave_audio,
                               "barometer_sensor_name": "synch_barometer",
                               "barometer_sample_rate_nominal_hz": self.sample_rate_barometer,
                               "barometer_epoch_s": self.signal_time_barometer,
                               "barometer_wf_raw": self.sinewave_barometer,
                               "accelerometer_epoch_s": self.signal_time_acc,
                               "accelerometer_wf_raw": self.sinewave_acc}}

        self.df_data = pd.DataFrame(self.dict_to_df).T

    def test_num_wiggle_is_2_audio(self):

        self.num_wiggle = rpd_wiggles.find_wiggle_num(df=self.df_data,
                                                      sig_wf_label=["audio_wf"],
                                                      sig_timestamps_label=["audio_epoch_s"])

        self.assertEqual(self.num_wiggle, 2)

    def test_num_wiggle_is_2_barometer(self):

        self.num_wiggle = rpd_wiggles.find_wiggle_num(df=self.df_data,
                                                      sig_wf_label=["barometer_wf_raw"],
                                                      sig_timestamps_label=["barometer_epoch_s"])
        self.assertEqual(self.num_wiggle, 2)

    def test_num_wiggle_is_4_audio_barometer(self):

        self.num_wiggle = rpd_wiggles.find_wiggle_num(df=self.df_data,
                                                      sig_wf_label=["audio_wf", "barometer_wf_raw"],
                                                      sig_timestamps_label=["audio_epoch_s", "barometer_epoch_s"])

        self.assertEqual(self.num_wiggle, 4)

    def test_num_wiggle_is_10_audio_barometer_acc(self):

        self.num_wiggle = rpd_wiggles.find_wiggle_num(df=self.df_data,
                                                      sig_wf_label=["audio_wf", "barometer_wf_raw",
                                                                    "accelerometer_wf_raw"],
                                                      sig_timestamps_label=["audio_epoch_s", "barometer_epoch_s",
                                                                            "accelerometer_epoch_s"])
        self.assertEqual(self.num_wiggle, 10)

    def test_num_wiggle_if_one_station(self):

        self.num_wiggle = rpd_wiggles.find_wiggle_num(df=self.df_data,
                                                      sig_wf_label=["audio_wf", "barometer_wf_raw",
                                                                    "accelerometer_wf_raw"],
                                                      sig_timestamps_label=["audio_epoch_s", "barometer_epoch_s",
                                                                            "accelerometer_epoch_s"],
                                                      station_id_str="1234567890")
        self.assertEqual(self.num_wiggle, 5)

    def tearDown(self):
        self.start_time = None
        self.end_time = None
        self.sample_rate_audio = None
        self.signal_time_audio = None
        self.frequency = None
        self.amplitude = None
        self.sinewave_audio = None
        self.sample_rate_barometer = None
        self.signal_time_barometer = None
        self.sinewave_barometer = None
        self.dict_to_df = None
        self.df_data = None
        self.yticks = None
        self.num_wiggle = None
        self.sinewave_barometer_base = None
        self.sample_rate_acc = None
        self.signal_time_acc = None
        self.sinewave_acc_base = None
        self.points_per_row = None
        self.sinewave_acc = None


class TestIrregularFindNumWiggles(unittest.TestCase):

    def setUp(self) -> None:
        # Create audio
        self.start_time = 0
        self.end_time = 10
        self.sample_rate_audio = 100
        self.signal_time_audio = np.arange(self.start_time, self.end_time, 1/self.sample_rate_audio)
        self.frequency = 3
        self.amplitude = 1
        self.sinewave_audio = self.amplitude * np.sin(2 * np.pi * self.frequency * self.signal_time_audio)

        # Create barometer
        self.sample_rate_barometer = 31
        self.signal_time_barometer = np.arange(self.start_time, self.end_time, 1/self.sample_rate_barometer)
        self.sinewave_barometer_base = self.amplitude * np.sin(2 * np.pi * self.frequency * self.signal_time_barometer)
        self.sinewave_barometer = self.sinewave_barometer_base.reshape((1, len(self.sinewave_barometer_base)))

        # Create accelerometer
        self.sample_rate_acc = 30
        self.signal_time_acc = np.arange(self.start_time, self.end_time, 1/(self.sample_rate_acc/3))
        self.length_for_signal = np.arange(self.start_time, self.end_time, 1/self.sample_rate_acc)
        self.sinewave_acc_base = self.amplitude * np.sin(2 * np.pi * self.frequency * self.length_for_signal)
        self.points_per_row = int(len(self.sinewave_acc_base)/3)
        self.sinewave_acc = self.sinewave_acc_base.reshape((3, self.points_per_row))

        # Create irregular df with one station with audio and barometer, other only audio
        self.dict_to_df_irregular = {0: {"station_id": "1234567890",
                                         "audio_sensor_name": "synch_audio",
                                         "audio_sample_rate_nominal_hz": self.sample_rate_audio,
                                         "audio_epoch_s": self.signal_time_audio,
                                         "audio_wf": self.sinewave_audio,
                                         "barometer_sensor_name": "synch_barometer",
                                         "barometer_sample_rate_nominal_hz": self.sample_rate_barometer,
                                         "barometer_epoch_s": self.signal_time_barometer,
                                         "barometer_wf_raw": self.sinewave_barometer},
                                     1: {"station_id": "2345678901",   # Add another station
                                         "audio_sensor_name": "synch_audio",
                                         "audio_sample_rate_nominal_hz": self.sample_rate_audio,
                                         "audio_epoch_s": self.signal_time_audio,
                                         "audio_wf": self.sinewave_audio,
                                         "accelerometer_epoch_s": self.signal_time_acc,
                                         "accelerometer_wf_raw": self.sinewave_acc}}

        self.df_data_irregular = pd.DataFrame(self.dict_to_df_irregular).T

    def test_irregular_num_wiggle_is_1_barometer(self):

        self.num_wiggle = rpd_wiggles.find_wiggle_num(df=self.df_data_irregular,
                                                      sig_wf_label=["barometer_wf_raw"],
                                                      sig_timestamps_label=["barometer_epoch_s"])
        self.assertEqual(self.num_wiggle, 1)

    def test_irregular_num_wiggle_is_3_audio_barometer(self):

        self.num_wiggle = rpd_wiggles.find_wiggle_num(df=self.df_data_irregular,
                                                      sig_wf_label=["audio_wf", "barometer_wf_raw"],
                                                      sig_timestamps_label=["audio_epoch_s", "barometer_epoch_s"])

        self.assertEqual(self.num_wiggle, 3)

    def test_irregular_num_wiggle_is_6_audio_barometer(self):

        self.num_wiggle = rpd_wiggles.find_wiggle_num(df=self.df_data_irregular,
                                                      sig_wf_label=["audio_wf", "barometer_wf_raw",
                                                                    "accelerometer_wf_raw"],
                                                      sig_timestamps_label=["audio_epoch_s", "barometer_epoch_s",
                                                                            "accelerometer_epoch_s"])

        self.assertEqual(self.num_wiggle, 6)

    def test_irregular_num_wiggle_if_one_station(self):

        self.num_wiggle = rpd_wiggles.find_wiggle_num(df=self.df_data_irregular,
                                                      sig_wf_label=["audio_wf", "barometer_wf_raw",
                                                                    "accelerometer_wf_raw"],
                                                      sig_timestamps_label=["audio_epoch_s", "barometer_epoch_s",
                                                                            "accelerometer_epoch_s"],
                                                      station_id_str="1234567890")
        self.assertEqual(self.num_wiggle, 2)

    def tearDown(self):
        self.start_time = None
        self.end_time = None
        self.sample_rate_audio = None
        self.signal_time_audio = None
        self.frequency = None
        self.amplitude = None
        self.sinewave_audio = None
        self.sample_rate_barometer = None
        self.signal_time_barometer = None
        self.sinewave_barometer = None
        self.dict_to_df = None
        self.df_data = None
        self.yticks = None
        self.num_wiggle = None
        self.sinewave_barometer_base = None
        self.sample_rate_acc = None
        self.signal_time_acc = None
        self.sinewave_acc_base = None
        self.points_per_row = None
        self.sinewave_acc = None


if __name__ == '__main__':
    unittest.main()

