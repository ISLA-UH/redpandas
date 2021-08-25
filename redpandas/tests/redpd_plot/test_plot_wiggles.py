import unittest
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
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
                                                      sig_wf_label=["audio_wf"])

        self.assertEqual(self.num_wiggle, 2)

    def test_num_wiggle_is_2_barometer(self):

        self.num_wiggle = rpd_wiggles.find_wiggle_num(df=self.df_data,
                                                      sig_wf_label=["barometer_wf_raw"])
        self.assertEqual(self.num_wiggle, 2)

    def test_num_wiggle_is_4_audio_barometer(self):

        self.num_wiggle = rpd_wiggles.find_wiggle_num(df=self.df_data,
                                                      sig_wf_label=["audio_wf", "barometer_wf_raw"])

        self.assertEqual(self.num_wiggle, 4)

    def test_num_wiggle_is_10_audio_barometer_acc(self):

        self.num_wiggle = rpd_wiggles.find_wiggle_num(df=self.df_data,
                                                      sig_wf_label=["audio_wf", "barometer_wf_raw",
                                                                    "accelerometer_wf_raw"])
        self.assertEqual(self.num_wiggle, 10)

    def test_num_wiggle_if_one_station(self):

        self.num_wiggle = rpd_wiggles.find_wiggle_num(df=self.df_data,
                                                      sig_wf_label=["audio_wf", "barometer_wf_raw",
                                                                    "accelerometer_wf_raw"],
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


class TestYticks(unittest.TestCase):
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

    def test_yticks_is_station_id_one_sensor_no_custom_yticks(self):
        self.yticks = rpd_wiggles.find_ylabel(df=self.df_data,
                                              sig_wf_label=["audio_wf"],
                                              sig_id_label="station_id",
                                              station_id_str=None,
                                              custom_yticks=None)
        self.assertEqual(self.yticks, ["1234567890", "2345678901"])

    def test_yticks_multiple_stations_multiple_sensors(self):
        self.yticks = rpd_wiggles.find_ylabel(df=self.df_data,
                                              sig_wf_label=["audio_wf", "barometer_wf_raw"],
                                              sig_id_label="station_id",
                                              station_id_str=None,
                                              custom_yticks=None)
        self.assertEqual(self.yticks, ['1234567890 aud', '1234567890 bar raw', '2345678901 aud', '2345678901 bar raw'])

    def test_yticks_multiple_stations_multiple_sensors_with_3c_sensors(self):
        self.yticks = rpd_wiggles.find_ylabel(df=self.df_data,
                                              sig_wf_label=["audio_wf", "barometer_wf_raw",
                                                            "accelerometer_wf_raw"],
                                              sig_id_label="station_id",
                                              station_id_str=None,
                                              custom_yticks=None)
        self.assertEqual(self.yticks, ["1234567890 aud", "1234567890 bar raw", "1234567890 acc X raw", "1234567890 acc Y raw",
                                       "1234567890 acc Z raw", "2345678901 aud", "2345678901 bar raw",
                                       "2345678901 acc X raw", "2345678901 acc Y raw", "2345678901 acc Z raw"])

    def test_yticks_one_station_multiple_sensors(self):
        self.yticks = rpd_wiggles.find_ylabel(df=self.df_data,
                                              sig_wf_label=["audio_wf", "barometer_wf_raw"],
                                              sig_id_label="station_id",
                                              station_id_str="1234567890")
        self.assertEqual(self.yticks, ['aud', 'bar raw'])

    def test_yticks_one_station_multiple_sensors_with_3c_sensors(self):
        self.yticks = rpd_wiggles.find_ylabel(df=self.df_data,
                                              sig_wf_label=["audio_wf", "barometer_wf_raw",
                                                            "accelerometer_wf_raw"],
                                              sig_id_label="station_id",
                                              station_id_str="1234567890")
        self.assertEqual(self.yticks, ['aud', 'bar raw', 'acc X raw', 'acc Y raw', 'acc Z raw'])

    def test_yticks_if_custom_yticks_index(self):
        self.yticks = rpd_wiggles.find_ylabel(df=self.df_data,
                                              sig_wf_label=["audio_wf"],
                                              sig_id_label="station_id",
                                              station_id_str=None,
                                              custom_yticks="index")

        self.assertEqual(self.yticks, [0, 1])

    def test_yticks_if_correct_input_custom_yticks(self):
        self.yticks = rpd_wiggles.find_ylabel(df=self.df_data,
                                              sig_wf_label=["audio_wf"],
                                              sig_id_label="station_id",
                                              station_id_str=None,
                                              custom_yticks=['a', 'b'])
        self.assertEqual(self.yticks, ['a', 'b'])

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
                                                      sig_wf_label=["barometer_wf_raw"])
        self.assertEqual(self.num_wiggle, 1)

    def test_irregular_num_wiggle_is_3_audio_barometer(self):

        self.num_wiggle = rpd_wiggles.find_wiggle_num(df=self.df_data_irregular,
                                                      sig_wf_label=["audio_wf", "barometer_wf_raw"])

        self.assertEqual(self.num_wiggle, 3)

    def test_irregular_num_wiggle_is_6_audio_barometer(self):

        self.num_wiggle = rpd_wiggles.find_wiggle_num(df=self.df_data_irregular,
                                                      sig_wf_label=["audio_wf", "barometer_wf_raw",
                                                                    "accelerometer_wf_raw"])

        self.assertEqual(self.num_wiggle, 6)

    def test_irregular_num_wiggle_if_one_station(self):

        self.num_wiggle = rpd_wiggles.find_wiggle_num(df=self.df_data_irregular,
                                                      sig_wf_label=["audio_wf", "barometer_wf_raw",
                                                                    "accelerometer_wf_raw"],
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


class TestIrregularFindYticks(unittest.TestCase):

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

    def test_irregular_yticks_is_station_id_one_sensor_no_custom_yticks(self):
        self.yticks = rpd_wiggles.find_ylabel(df=self.df_data_irregular,
                                              sig_wf_label=["audio_wf"],
                                              sig_id_label="station_id",
                                              station_id_str=None,
                                              custom_yticks=None)
        self.assertEqual(self.yticks, ["1234567890", "2345678901"])

    def test_irregular_yticks_multiple_stations_multiple_sensors(self):
        self.yticks = rpd_wiggles.find_ylabel(df=self.df_data_irregular,
                                              sig_wf_label=["audio_wf", "barometer_wf_raw"],
                                              sig_id_label="station_id",
                                              station_id_str=None,
                                              custom_yticks=None)
        self.assertEqual(self.yticks, ['1234567890 aud', '1234567890 bar raw', '2345678901 aud'])

    def test_irregular_yticks_one_station_multiple_sensors_with_3c_sensors(self):
        self.yticks = rpd_wiggles.find_ylabel(df=self.df_data_irregular,
                                              sig_wf_label=["audio_wf",
                                                            "accelerometer_wf_raw"],
                                              sig_id_label="station_id",
                                              station_id_str="2345678901")
        self.assertEqual(self.yticks, ['aud', 'acc X raw', 'acc Y raw', 'acc Z raw'])

    def test_irregular_yticks_one_station_multiple_sensors(self):
        self.yticks = rpd_wiggles.find_ylabel(df=self.df_data_irregular,
                                              sig_wf_label=["audio_wf", "barometer_wf_raw"],
                                              sig_id_label="station_id",
                                              station_id_str="2345678901")
        self.assertEqual(self.yticks, ['aud'])

    def test_irregular_yticks_if_custom_yticks_index(self):
        self.yticks = rpd_wiggles.find_ylabel(df=self.df_data_irregular,
                                              sig_wf_label=["audio_wf"],
                                              sig_id_label="station_id",
                                              station_id_str=None,
                                              custom_yticks="index")

        self.assertEqual(self.yticks, [0, 1])

    def test_irregular_yticks_if_correct_input_custom_yticks(self):
        self.yticks = rpd_wiggles.find_ylabel(df=self.df_data_irregular,
                                              sig_wf_label=["audio_wf"],
                                              sig_id_label="station_id",
                                              station_id_str=None,
                                              custom_yticks=['a', 'b'])
        self.assertEqual(self.yticks, ['a', 'b'])

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


class TestPlotWigglesPandas(unittest.TestCase):

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

    def test_return_is_figure_instance(self):
        self.figure = rpd_wiggles.plot_wiggles_pandas(df=self.df_data_irregular,
                                                      show_figure=False)

        self.assertEqual(type(self.figure), Figure)

    def test_wrong_number_of_columns(self):
        with self.assertRaises(ValueError): rpd_wiggles.plot_wiggles_pandas(df=self.df_data_irregular,
                                                                            sig_wf_label="audio_wf",
                                                                            sig_timestamps_label=["audio_epoch_s",
                                                                                                  "barometer_epoch_s"])

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

