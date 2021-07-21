import unittest
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
import redpandas.redpd_plot as rpd_plot


class TestFindWiggleNumYticks(unittest.TestCase):
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
        sinewave_barometer = self.amplitude * np.sin(2 * np.pi * self.frequency * self.signal_time_barometer)
        self.sinewave_barometer = sinewave_barometer.reshape((1, len(sinewave_barometer)))

        # Create df
        self.dict_to_df = {0: {"station_id": "1234567890",
                               "audio_sensor_name": "synch_audio",
                               "audio_sample_rate_nominal_hz": self.sample_rate_audio,
                               "audio_epoch_s": self.signal_time_audio,
                               "audio_wf": self.sinewave_audio,
                               "barometer_sensor_name": "synch_barometer",
                               "barometer_sample_rate_nominal_hz": self.sample_rate_barometer,
                               "barometer_epoch_s": self.signal_time_barometer,
                               "barometer_wf": self.sinewave_barometer},
                           1: {"station_id": "2345678901",   # Add another station
                               "audio_sensor_name": "synch_audio",
                               "audio_sample_rate_nominal_hz": self.sample_rate_audio,
                               "audio_epoch_s": self.signal_time_audio,
                               "audio_wf": self.sinewave_audio,
                               "barometer_sensor_name": "synch_barometer",
                               "barometer_sample_rate_nominal_hz": self.sample_rate_barometer,
                               "barometer_epoch_s": self.signal_time_barometer,
                               "barometer_wf": self.sinewave_barometer}}

        self.df_data = pd.DataFrame(self.dict_to_df).T

    def test_num_wiggle_is_2_audio(self):

        self.num_wiggle, _ = rpd_plot.find_wiggle_num_yticks(df=self.df_data,
                                                             sig_wf_label=["audio_wf"],
                                                             sig_id_label="station_id",
                                                             station_id_str=None,
                                                             custom_yticks=None)

        self.assertEqual(self.num_wiggle, 2)

    def test_num_wiggle_is_2_barometer(self):

        self.num_wiggle, _ = rpd_plot.find_wiggle_num_yticks(df=self.df_data,
                                                             sig_wf_label=["barometer_wf"],
                                                             sig_id_label="station_id",
                                                             station_id_str=None,
                                                             custom_yticks=None)
        self.assertEqual(self.num_wiggle, 2)

    def test_num_wiggle_is_4_audio_barometer(self):

        self.num_wiggle, _ = rpd_plot.find_wiggle_num_yticks(df=self.df_data,
                                                             sig_wf_label=["audio_wf", "barometer_wf"],
                                                             sig_id_label="station_id",
                                                             station_id_str=None)

        self.assertEqual(self.num_wiggle, 4)

    def test_yticks_is_station_id(self):
        _, self.yticks = rpd_plot.find_wiggle_num_yticks(df=self.df_data,
                                                         sig_wf_label=["audio_wf"],
                                                         sig_id_label="station_id",
                                                         station_id_str=None,
                                                         custom_yticks=None)

        self.assertEqual(self.yticks, ["1234567890", "2345678901"])

    def test_yticks_if_custom_yticks_index(self):
        _, self.yticks = rpd_plot.find_wiggle_num_yticks(df=self.df_data,
                                                         sig_wf_label=["audio_wf"],
                                                         sig_id_label="station_id",
                                                         station_id_str=None,
                                                         custom_yticks="index")

        self.assertEqual(self.yticks, [0, 1])

    def test_yticks_if_correct_input_custom_yticks(self):
        _, self.yticks = rpd_plot.find_wiggle_num_yticks(df=self.df_data,
                                                         sig_wf_label=["audio_wf"],
                                                         sig_id_label="station_id",
                                                         station_id_str=None,
                                                         custom_yticks=['a', 'b'])
        self.assertEqual(self.yticks, ['a', 'b'])

    def test_yticks_if_incorrect_input_custom_yticks(self):
        with self.assertRaises(ValueError): rpd_plot.find_wiggle_num_yticks(df=self.df_data,
                                                                            sig_wf_label=["audio_wf"],
                                                                            sig_id_label="station_id",
                                                                            station_id_str=None,
                                                                            custom_yticks=['a', 'b', 'c'])

    def test_yticks_if_correct_input_custom_yticks_but_wrong_station(self):
        with self.assertRaises(ValueError): rpd_plot.find_wiggle_num_yticks(df=self.df_data,
                                                                            sig_wf_label=["audio_wf"],
                                                                            sig_id_label="station_id",
                                                                            station_id_str="283976987",
                                                                            custom_yticks=['a', 'b'])

    def test_yticks_if_correct_input_custom_yticks_and_correct_station(self):
        with self.assertRaises(ValueError): rpd_plot.find_wiggle_num_yticks(df=self.df_data,
                                                                            sig_wf_label=["audio_wf"],
                                                                            sig_id_label="station_id",
                                                                            station_id_str="1234567890",
                                                                            custom_yticks=['a', 'b'])

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


class TestIrregularFindWiggleNumYticks(unittest.TestCase):

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
        sinewave_barometer = self.amplitude * np.sin(2 * np.pi * self.frequency * self.signal_time_barometer)
        self.sinewave_barometer = sinewave_barometer.reshape((1, len(sinewave_barometer)))

        # Create irregular df with one station with audio and barometer, other only audio
        self.dict_to_df_irregular = {0: {"station_id": "1234567890",
                                         "audio_sensor_name": "synch_audio",
                                         "audio_sample_rate_nominal_hz": self.sample_rate_audio,
                                         "audio_epoch_s": self.signal_time_audio,
                                         "audio_wf": self.sinewave_audio,
                                         "barometer_sensor_name": "synch_barometer",
                                         "barometer_sample_rate_nominal_hz": self.sample_rate_barometer,
                                         "barometer_epoch_s": self.signal_time_barometer,
                                         "barometer_wf": sinewave_barometer},
                                     1: {"station_id": "2345678901",   # Add another station
                                         "audio_sensor_name": "synch_audio",
                                         "audio_sample_rate_nominal_hz": self.sample_rate_audio,
                                         "audio_epoch_s": self.signal_time_audio,
                                         "audio_wf": self.sinewave_audio}}

        self.df_data_irregular = pd.DataFrame(self.dict_to_df_irregular).T

    def test_irregular_num_wiggle_is_1_barometer(self):

        self.num_wiggle, _ = rpd_plot.find_wiggle_num_yticks(df=self.df_data_irregular,
                                                             sig_wf_label=["barometer_wf"],
                                                             sig_id_label="station_id",
                                                             station_id_str=None,
                                                             custom_yticks=None)
        self.assertEqual(self.num_wiggle, 1)

    def test_irregular_num_wiggle_is_3_audio_barometer(self):

        self.num_wiggle, _ = rpd_plot.find_wiggle_num_yticks(df=self.df_data_irregular,
                                                             sig_wf_label=["audio_wf", "barometer_wf"],
                                                             sig_id_label="station_id",
                                                             station_id_str=None)

        self.assertEqual(self.num_wiggle, 3)

    def test_irregular_yticks_is_station_id(self):
        _, self.yticks = rpd_plot.find_wiggle_num_yticks(df=self.df_data_irregular,
                                                         sig_wf_label=["audio_wf"],
                                                         sig_id_label="station_id",
                                                         station_id_str=None,
                                                         custom_yticks=None)

        self.assertEqual(self.yticks, ["1234567890", "2345678901"])

    def test_irregular_yticks_if_custom_yticks_index(self):
        _, self.yticks = rpd_plot.find_wiggle_num_yticks(df=self.df_data_irregular,
                                                         sig_wf_label=["audio_wf"],
                                                         sig_id_label="station_id",
                                                         station_id_str=None,
                                                         custom_yticks="index")

        self.assertEqual(self.yticks, [0, 1])

    def test_irregular_yticks_if_correct_input_custom_yticks(self):
        _, self.yticks = rpd_plot.find_wiggle_num_yticks(df=self.df_data_irregular,
                                                         sig_wf_label=["audio_wf"],
                                                         sig_id_label="station_id",
                                                         station_id_str=None,
                                                         custom_yticks=['a', 'b'])
        self.assertEqual(self.yticks, ['a', 'b'])

    def test_irregular_yticks_if_incorrect_input_custom_yticks(self):
        with self.assertRaises(ValueError): rpd_plot.find_wiggle_num_yticks(df=self.df_data_irregular,
                                                                            sig_wf_label=["audio_wf"],
                                                                            sig_id_label="station_id",
                                                                            station_id_str=None,
                                                                            custom_yticks=['a', 'b', 'c'])

    def test_irregular_yticks_if_correct_input_custom_yticks_but_wrong_station(self):
        with self.assertRaises(ValueError): rpd_plot.find_wiggle_num_yticks(df=self.df_data_irregular,
                                                                            sig_wf_label=["audio_wf"],
                                                                            sig_id_label="station_id",
                                                                            station_id_str="283976987",
                                                                            custom_yticks=['a', 'b'])

    def test_irregular_yticks_if_correct_input_custom_yticks_and_correct_station(self):
        with self.assertRaises(ValueError): rpd_plot.find_wiggle_num_yticks(df=self.df_data_irregular,
                                                                            sig_wf_label=["audio_wf"],
                                                                            sig_id_label="station_id",
                                                                            station_id_str="1234567890",
                                                                            custom_yticks=['a', 'b'])

    def test_irregular_yticks_if_no_sensor_data_in_station(self):
        with self.assertRaises(ValueError): rpd_plot.find_wiggle_num_yticks(df=self.df_data_irregular,
                                                                            sig_wf_label=["barometer_wf"],
                                                                            sig_id_label="station_id",
                                                                            station_id_str="1234567890",
                                                                            custom_yticks=['a', 'b'])

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


class TestCheckIfStationExistsInDf(unittest.TestCase):

    def setUp(self) -> None:
        # Create df
        self.dict_to_df = {0: {"station_id": "1234567890"},
                           1: {"station_id": "2345678901"}}

        self.df_data = pd.DataFrame(self.dict_to_df).T

    def test_correct_station(self):
        self.assertTrue(rpd_plot.check_if_station_exists_in_df(df=self.df_data,
                                                               station_id_str="1234567890",
                                                               sig_id_label="station_id"))

    def test_incorrect_station(self):
        self.assertFalse(rpd_plot.check_if_station_exists_in_df(df=self.df_data,
                                                                station_id_str="1234560",
                                                                sig_id_label="station_id"))

    def tearDown(self):
        self.dict_to_df = None
        self.df_data = None


class TestCheckIfColumnExistsInDf(unittest.TestCase):

    def setUp(self):
        self.dict_to_df = {0: {"audio_wf": "0"},
                           1: {"barometer_wf": "1"}}
        self.df_data = pd.DataFrame(self.dict_to_df).T

    def test_correct_column_name(self):
        self.assertTrue(rpd_plot.check_if_column_exists_in_df(df=self.df_data,
                                                              label="audio_wf"))

    def test_incorrect_column_name(self):
        self.assertFalse(rpd_plot.check_if_column_exists_in_df(df=self.df_data,
                                                               label="audio"))

    def tearDown(self):
        self.dict_to_df = None
        self.df_data = None


class TestDetermineTimeEpochOrigin(unittest.TestCase):

    def setUp(self) -> None:
        # Create audio
        self.start_time_audio = 1.
        self.end_time = 10.
        self.sample_rate_audio = 100.
        self.signal_time_audio = np.arange(self.start_time_audio, self.end_time, 1/self.sample_rate_audio)

        # Create barometer
        self.start_time_barometer = 2.
        self.sample_rate_barometer = 31.
        self.signal_time_barometer = np.arange(self.start_time_barometer, self.end_time, 1/self.sample_rate_barometer)

        # Create df
        self.dict_to_df = {0: {"station_id": "1234567890",
                               "audio_sensor_name": "synch_audio",
                               "audio_sample_rate_nominal_hz": self.sample_rate_audio,
                               "audio_epoch_s": self.signal_time_audio,
                               "barometer_sensor_name": "synch_barometer",
                               "barometer_sample_rate_nominal_hz": self.sample_rate_barometer,
                               "barometer_epoch_s": self.signal_time_barometer},
                           1: {"station_id": "2345678901",   # Add another station
                               "audio_sensor_name": "synch_audio",
                               "audio_sample_rate_nominal_hz": self.sample_rate_audio,
                               "audio_epoch_s": self.signal_time_audio,
                               "barometer_sensor_name": "synch_barometer",
                               "barometer_sample_rate_nominal_hz": self.sample_rate_barometer,
                               "barometer_epoch_s": self.signal_time_barometer}}

        self.df_data = pd.DataFrame(self.dict_to_df).T

    def test_result_if_no_timestamp_input_all_stations(self):
        self.time_epoch_origin = rpd_plot.determine_time_epoch_origin(df=self.df_data,
                                                                      sig_timestamps_label=None,
                                                                      station_id_str=None,
                                                                      sig_id_label="station_id")
        self.assertEqual(self.time_epoch_origin, 0.0)

    def test_result_if_no_timestamp_input_with_correct_one_station(self):
        self.time_epoch_origin = rpd_plot.determine_time_epoch_origin(df=self.df_data,
                                                                      sig_timestamps_label=None,
                                                                      station_id_str="2345678901",
                                                                      sig_id_label="station_id")
        self.assertEqual(self.time_epoch_origin, 0.0)

    def test_result_if_no_timestamp_input_with_incorrect_one_station(self):
        with self.assertRaises(ValueError): rpd_plot.determine_time_epoch_origin(df=self.df_data,
                                                                                 sig_timestamps_label=None,
                                                                                 station_id_str="345678901",
                                                                                 sig_id_label="station_id")

    def test_result_with_correct_timestamp_input_all_stations(self):
        self.time_epoch_origin = rpd_plot.determine_time_epoch_origin(df=self.df_data,
                                                                      sig_timestamps_label=["audio_epoch_s"],
                                                                      station_id_str=None,
                                                                      sig_id_label="station_id")
        self.assertEqual(self.time_epoch_origin, 1.0)

    def test_result_with_incorrect_timestamp_input_all_stations(self):
        with self.assertRaises(ValueError): rpd_plot.determine_time_epoch_origin(df=self.df_data,
                                                                                 sig_timestamps_label=["audio_s"],
                                                                                 station_id_str=None,
                                                                                 sig_id_label="station_id")

    def test_result_with_multiple_timestamp_input_all_stations(self):
        self.time_epoch_origin = rpd_plot.determine_time_epoch_origin(df=self.df_data,
                                                                      sig_timestamps_label=["audio_epoch_s",
                                                                                            "barometer_epoch_s"],
                                                                      station_id_str=None,
                                                                      sig_id_label="station_id")
        self.assertEqual(self.time_epoch_origin, 1.0)

    def test_result_with_multiple_timestamp_input_correct_one_station(self):
        self.time_epoch_origin = rpd_plot.determine_time_epoch_origin(df=self.df_data,
                                                                      sig_timestamps_label=["audio_epoch_s",
                                                                                            "barometer_epoch_s"],
                                                                      station_id_str="2345678901",
                                                                      sig_id_label="station_id")
        self.assertEqual(self.time_epoch_origin, 1.0)

    def test_result_with_multiple_timestamp_input_incorrect_one_station(self):
        with self.assertRaises(ValueError): rpd_plot.determine_time_epoch_origin(df=self.df_data,
                                                                                 sig_timestamps_label=["audio_epoch_s",
                                                                                                       "barometer_epoch_s"],
                                                                                 station_id_str="345678901",
                                                                                 sig_id_label="station_id")

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


# class TestPlotWigglesPandas(unittest.TestCase):
#
#     def setUp(self) -> None:
#
#
#     def test_return_is_figure_instance(self):
#         self.figure = rpd_plot.plot_wiggles_pandas()
#
#         self.assertEqual(type(self.figure), Figure)
#
#     def

if __name__ == '__main__':
    unittest.main()