import unittest
import pandas as pd
import numpy as np
import redpandas.redpd_plot.wiggles as rpd_wiggles


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
                                              sig_timestamps_label=["audio_epoch_s"],
                                              sig_id_label="station_id",
                                              station_id_str=None,
                                              custom_yticks=None)
        self.assertEqual(self.yticks, ["1234567890", "2345678901"])

    def test_yticks_multiple_stations_multiple_sensors(self):
        self.yticks = rpd_wiggles.find_ylabel(df=self.df_data,
                                              sig_wf_label=["audio_wf", "barometer_wf_raw"],
                                              sig_timestamps_label=["audio_epoch_s", "barometer_epoch_s"],
                                              sig_id_label="station_id",
                                              station_id_str=None,
                                              custom_yticks=None)
        self.assertEqual(self.yticks, ['1234567890 aud', '2345678901 aud', '1234567890 bar raw', '2345678901 bar raw'])

    def test_yticks_multiple_stations_multiple_sensors_with_3c_sensors(self):
        self.yticks = rpd_wiggles.find_ylabel(df=self.df_data,
                                              sig_wf_label=["audio_wf", "barometer_wf_raw",
                                                            "accelerometer_wf_raw"],
                                              sig_timestamps_label=["audio_epoch_s", "barometer_epoch_s",
                                                                    "accelerometer_epoch_s"],
                                              sig_id_label="station_id",
                                              station_id_str=None,
                                              custom_yticks=None)
        self.assertEqual(self.yticks, ["1234567890 aud", "2345678901 aud",
                                       "1234567890 bar raw", "2345678901 bar raw",
                                       "1234567890 acc X raw", "1234567890 acc Y raw", "1234567890 acc Z raw",
                                       "2345678901 acc X raw", "2345678901 acc Y raw", "2345678901 acc Z raw"])

    def test_yticks_one_station_multiple_sensors(self):
        self.yticks = rpd_wiggles.find_ylabel(df=self.df_data,
                                              sig_wf_label=["audio_wf", "barometer_wf_raw"],
                                              sig_timestamps_label=["audio_epoch_s", "barometer_epoch_s"],
                                              sig_id_label="station_id",
                                              station_id_str="1234567890")
        self.assertEqual(self.yticks, ['aud', 'bar raw'])

    def test_yticks_one_station_multiple_sensors_with_3c_sensors(self):
        self.yticks = rpd_wiggles.find_ylabel(df=self.df_data,
                                              sig_wf_label=["audio_wf", "barometer_wf_raw",
                                                            "accelerometer_wf_raw"],
                                              sig_timestamps_label=["audio_epoch_s", "barometer_epoch_s",
                                                                    "accelerometer_epoch_s"],
                                              sig_id_label="station_id",
                                              station_id_str="1234567890")
        self.assertEqual(self.yticks, ['aud', 'bar raw', 'acc X raw', 'acc Y raw', 'acc Z raw'])

    def test_yticks_if_custom_yticks_index(self):
        self.yticks = rpd_wiggles.find_ylabel(df=self.df_data,
                                              sig_wf_label=["audio_wf"],
                                              sig_timestamps_label=["audio_epoch_s"],
                                              sig_id_label="station_id",
                                              station_id_str=None,
                                              custom_yticks="index")

        self.assertEqual(self.yticks, [0, 1])

    def test_yticks_if_correct_input_custom_yticks(self):
        self.yticks = rpd_wiggles.find_ylabel(df=self.df_data,
                                              sig_wf_label=["audio_wf"],
                                              sig_timestamps_label=["audio_epoch_s"],
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
                                              sig_timestamps_label=["audio_epoch_s"],
                                              sig_id_label="station_id",
                                              station_id_str=None,
                                              custom_yticks=None)
        self.assertEqual(self.yticks, ["1234567890", "2345678901"])

    def test_irregular_yticks_multiple_stations_multiple_sensors(self):
        self.yticks = rpd_wiggles.find_ylabel(df=self.df_data_irregular,
                                              sig_wf_label=["audio_wf", "barometer_wf_raw"],
                                              sig_timestamps_label=["audio_epoch_s", "barometer_epoch_s"],
                                              sig_id_label="station_id",
                                              station_id_str=None,
                                              custom_yticks=None)
        self.assertEqual(self.yticks, ['1234567890 aud', '2345678901 aud', '1234567890 bar raw'])

    def test_irregular_yticks_one_station_multiple_sensors_with_3c_sensors(self):
        self.yticks = rpd_wiggles.find_ylabel(df=self.df_data_irregular,
                                              sig_wf_label=["audio_wf",
                                                            "accelerometer_wf_raw"],
                                              sig_timestamps_label=["audio_epoch_s", "accelerometer_epoch_s"],
                                              sig_id_label="station_id",
                                              station_id_str="2345678901")
        self.assertEqual(self.yticks, ['aud', 'acc X raw', 'acc Y raw', 'acc Z raw'])

    def test_irregular_yticks_one_station_multiple_sensors(self):
        self.yticks = rpd_wiggles.find_ylabel(df=self.df_data_irregular,
                                              sig_wf_label=["audio_wf", "barometer_wf_raw"],
                                              sig_timestamps_label=["audio_epoch_s", "barometer_epoch_s"],
                                              sig_id_label="station_id",
                                              station_id_str="2345678901")
        self.assertEqual(self.yticks, ['aud'])

    def test_irregular_yticks_if_custom_yticks_index(self):
        self.yticks = rpd_wiggles.find_ylabel(df=self.df_data_irregular,
                                              sig_wf_label=["audio_wf"],
                                              sig_timestamps_label=["audio_epoch_s"],
                                              sig_id_label="station_id",
                                              station_id_str=None,
                                              custom_yticks="index")

        self.assertEqual(self.yticks, [0, 1])

    def test_irregular_yticks_if_correct_input_custom_yticks(self):
        self.yticks = rpd_wiggles.find_ylabel(df=self.df_data_irregular,
                                              sig_wf_label=["audio_wf"],
                                              sig_timestamps_label=["audio_epoch_s"],
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


if __name__ == '__main__':
    unittest.main()

