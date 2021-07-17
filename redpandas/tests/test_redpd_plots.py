import unittest
import pandas as pd
import numpy as np
import redpandas.redpd_plot as rpd_plot


class TestFindWiggleNumYticks(unittest.TestCase):
    def setUp(self) -> None:
        # Audio
        self.start_time = 0
        self.end_time = 10
        self.sample_rate_audio = 100
        self.signal_time_audio = np.arange(self.start_time, self.end_time, 1/self.sample_rate_audio)
        self.frequency = 3
        self.amplitude = 1
        self.sinewave_audio = self.amplitude * np.sin(2 * np.pi * self.frequency * self.signal_time_audio)

        # Barometer
        self.sample_rate_barometer = 30
        self.signal_time_barometer = np.arange(self.start_time, self.end_time, 1/self.sample_rate_audio)
        self.sinewave_barometer = self.amplitude * np.sin(2 * np.pi * self.frequency * self.signal_time_barometer)

        self.dict_to_df = {"station_id": "1234567890",
                           "audio_sensor_name": "synch_audio",
                           "audio_sample_rate_nominal_hz": self.sample_rate_audio,
                           "audio_epoch_s": self.signal_time_audio,
                           "audio_wf": self.sinewave_audio,
                           "barometer_sensor_name": "synch_barometer",
                           "barometer_sample_rate_nominal_hz": self.sample_rate_audio,
                           "barometer_epoch_s": self.signal_time_barometer,
                           "barometer_wf": self.sinewave_barometer}

        self.df_data = pd.DataFrame.from_dict(self.dict_to_df)

    def test_results_num_wiggle_is_2(self):

        self.num_wiggle, _ = rpd_plot.find_wiggle_num_yticks(df=self.df_data,
                                                             sig_wf_label="audio_wf",
                                                             sig_id_label="station_id",
                                                             station_id_str=None,
                                                             custom_yticks=None)
        self.assertEqual(self.num_wiggle, 1)

    def test_results_num_wiggle_is_4(self):

        self.num_wiggle, _ = rpd_plot.find_wiggle_num_yticks(df=self.df_data,
                                                             sig_wf_label="audio_wf",
                                                             sig_id_label="station_id",
                                                             station_id_str=None,
                                                             custom_yticks=None)
        self.assertEqual(self.num_wiggle, 1)

    def test_results_yticks_is_station_id(self):
        _, self.yticks = rpd_plot.find_wiggle_num_yticks(df=self.df_data,
                                                         sig_wf_label="audio_wf",
                                                         sig_id_label="station_id",
                                                         station_id_str=None,
                                                         custom_yticks=None)

        self.assertEqual(self.yticks, ["1234567890", "1234567890"])


    def tearDown(self):
        self.example_station = None
        self.sensor_sample_rate_hz = None
        self.sensor_epoch_s = None
        self.sensor_raw = None
        self.sensor_nans = None