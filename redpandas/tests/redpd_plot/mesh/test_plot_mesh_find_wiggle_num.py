import unittest
import numpy as np
import pandas as pd
from libquantum.spectra import stft_from_sig
import redpandas.redpd_plot.mesh as rpd_mesh


class TestFindWiggleNumTfr(unittest.TestCase):
    def setUp(self) -> None:
        # Create audio mesh
        self.start_time = 0
        self.end_time = 10
        self.sample_rate_audio = 100
        self.signal_time_audio = np.arange(self.start_time, self.end_time, 1/self.sample_rate_audio)
        self.frequency = 3
        self.amplitude = 1
        self.sinewave_audio = self.amplitude * np.sin(2 * np.pi * self.frequency * self.signal_time_audio)

        # Create audio mesh
        self.audio_STFT, self.audio_STFT_bits, _, _ = \
            stft_from_sig(sig_wf=self.sinewave_audio,
                          frequency_sample_rate_hz=self.sample_rate_audio,
                          band_order_Nth=3)
        # Create barometer
        self.sample_rate_barometer = 31
        self.signal_time_barometer = np.arange(self.start_time, self.end_time, 1/self.sample_rate_barometer)
        self.sinewave_barometer_base = self.amplitude * np.sin(2 * np.pi * self.frequency * self.signal_time_barometer)
        self.sinewave_barometer = self.sinewave_barometer_base.reshape((1, len(self.sinewave_barometer_base)))

        # Create barometer mesh
        self.bar_stft_all = []
        self.bar_stft_bits_all = []

        for dimension in range(len(self.sinewave_barometer)):
            self.bar_STFT, self.bar_STFT_bits, _, _ = \
                stft_from_sig(sig_wf=self.sinewave_barometer[dimension],
                              frequency_sample_rate_hz=self.sample_rate_barometer,
                              band_order_Nth=3)

            self.bar_stft_all.append(self.bar_STFT)
            self.bar_stft_bits_all.append(self.bar_STFT_bits)

        # Create accelerometer
        self.sample_rate_acc = 30
        self.signal_time_acc = np.arange(self.start_time, self.end_time, 1/(self.sample_rate_acc/3))
        self.length_for_signal = np.arange(self.start_time, self.end_time, 1/self.sample_rate_acc)
        self.sinewave_acc_base = self.amplitude * np.sin(2 * np.pi * self.frequency * self.length_for_signal)
        self.points_per_row = int(len(self.sinewave_acc_base)/3)
        self.sinewave_acc = self.sinewave_acc_base.reshape((3, self.points_per_row))

        # Create accelerometer mesh
        self.acc_stft_all = []
        self.acc_stft_bits_all = []
        for dimension in range(self.sinewave_acc.ndim):
            self.acc_STFT, self.acc_STFT_bits, _, _ = \
                stft_from_sig(sig_wf=self.sinewave_acc[dimension],
                              frequency_sample_rate_hz=self.sample_rate_acc,
                              band_order_Nth=3)

            self.acc_stft_all.append(self.acc_STFT)
            self.acc_stft_bits_all.append(self.acc_STFT_bits)

        # Create df
        self.dict_to_df_multiple = {0: {"station_id": "1234567890",
                                        "audio_stft": self.audio_STFT,
                                        "audio_stft_bits": self.audio_STFT_bits},
                                    1: {"station_id": "2345678901",   # Add another station
                                        "audio_stft": self.audio_STFT,
                                        "audio_stft_bits": self.audio_STFT_bits}}
        self.df_data = pd.DataFrame(self.dict_to_df_multiple).T

        # Make bar/acc mesh arrays, add them to lists, add column to df.
        self.bar_stft_array = np.array(self.bar_stft_all)
        self.bar_stft_bits_array = np.array(self.bar_stft_bits_all)

        self.bar_stft_all_for_both_stations = []
        self.bar_stft_bits_all_for_both_stations = []

        self.acc_stft_array = np.array(self.acc_stft_all)
        self.acc_stft_bits_array = np.array(self.acc_stft_bits_all)

        self.acc_stft_all_for_both_stations = []
        self.acc_stft_bits_all_for_both_stations = []

        for number_stations_is_2 in range(2):
            self.acc_stft_all_for_both_stations.append(self.acc_stft_array)
            self.acc_stft_bits_all_for_both_stations.append(self.acc_stft_bits_array)

            self.bar_stft_all_for_both_stations.append(self.bar_stft_array)
            self.bar_stft_bits_all_for_both_stations.append(self.bar_stft_bits_array)

        # Add columns with mesh
        self.df_data["barometer_stft"] = self.bar_stft_all_for_both_stations
        self.df_data["barometer_stft_bits"] = self.bar_stft_bits_all_for_both_stations

        self.df_data["accelerometer_stft"] = self.acc_stft_all_for_both_stations
        self.df_data["accelerometer_stft_bits"] = self.acc_stft_bits_all_for_both_stations

    def test_wiggles_aud_is_2(self):
        self.wiggle_num = rpd_mesh.find_wiggle_num_tfr(df=self.df_data,
                                                       mesh_tfr_label=["audio_stft_bits"])
        self.assertEqual(self.wiggle_num, 2)

    def test_wiggles_acc_is_6(self):
        self.wiggle_num = rpd_mesh.find_wiggle_num_tfr(df=self.df_data,
                                                       mesh_tfr_label=["accelerometer_stft_bits"])
        self.assertEqual(self.wiggle_num, 6)

    def test_wiggles_aud_acc_is_8(self):
        self.wiggle_num = rpd_mesh.find_wiggle_num_tfr(df=self.df_data,
                                                       mesh_tfr_label=["accelerometer_stft_bits", "audio_stft_bits"])
        self.assertEqual(self.wiggle_num, 8)

    def test_wiggles_barometer_is_2(self):
        self.wiggle_num = rpd_mesh.find_wiggle_num_tfr(df=self.df_data,
                                                       mesh_tfr_label=["barometer_stft_bits"])
        self.assertEqual(self.wiggle_num, 2)

    def test_wiggles_aud_bar_acc_is_10(self):
        self.wiggle_num = rpd_mesh.find_wiggle_num_tfr(df=self.df_data,
                                                       mesh_tfr_label=["accelerometer_stft_bits", "audio_stft_bits",
                                                                       "barometer_stft_bits"])
        self.assertEqual(self.wiggle_num, 10)

    def tearDown(self):
        self.start_time = None
        self.end_time = None
        self.sample_rate_audio = None
        self.signal_time_audio = None
        self.frequency = None
        self.amplitude = None
        self.sinewave_audio = None
        self.audio_STFT = None
        self.audio_STFT_bits = None
        self.sample_rate_acc = None
        self.signal_time_acc = None
        self.length_for_signal = None
        self.sinewave_acc_base = None
        self.points_per_row = None
        self.sinewave_acc = None
        self.acc_stft_all = None
        self.acc_stft_bits_all = None
        self.acc_STFT = None
        self.acc_STFT_bits = None
        self.dict_to_df_multiple = None
        self.df_data = None
        self.acc_stft_array = None
        self.acc_stft_bits_array = None
        self.acc_stft_all_for_both_stations = None
        self.acc_stft_bits_all_for_both_stations = None
        self.wiggle_num = None
        self.sample_rate_barometer = None
        self.signal_time_barometer = None
        self.sinewave_barometer_base = None
        self.sinewave_barometer = None
        self.bar_stft_all = None
        self.bar_stft_bits_all = None
        self.bar_STFT = None
        self.bar_STFT_bits = None
        self.bar_stft_array = None
        self.bar_stft_bits_array = None
        self.bar_stft_all_for_both_stations = None
        self.bar_stft_bits_all_for_both_stations = None


class TestFindWiggleNumTfrIrregular(unittest.TestCase):
    def setUp(self) -> None:
        # Create audio mesh
        self.start_time = 0
        self.end_time = 10
        self.sample_rate_audio = 100
        self.signal_time_audio = np.arange(self.start_time, self.end_time, 1/self.sample_rate_audio)
        self.frequency = 3
        self.amplitude = 1
        self.sinewave_audio = self.amplitude * np.sin(2 * np.pi * self.frequency * self.signal_time_audio)

        # Create audio mesh
        self.audio_STFT, self.audio_STFT_bits, _, _ = \
            stft_from_sig(sig_wf=self.sinewave_audio,
                          frequency_sample_rate_hz=self.sample_rate_audio,
                          band_order_Nth=3)
        # Create barometer
        self.sample_rate_barometer = 31
        self.signal_time_barometer = np.arange(self.start_time, self.end_time, 1/self.sample_rate_barometer)
        self.sinewave_barometer_base = self.amplitude * np.sin(2 * np.pi * self.frequency * self.signal_time_barometer)
        self.sinewave_barometer = self.sinewave_barometer_base.reshape((1, len(self.sinewave_barometer_base)))

        # Create barometer mesh
        self.bar_stft_all = []
        self.bar_stft_bits_all = []

        for dimension in range(len(self.sinewave_barometer)):
            self.bar_STFT, self.bar_STFT_bits, _, _ = \
                stft_from_sig(sig_wf=self.sinewave_barometer[dimension],
                              frequency_sample_rate_hz=self.sample_rate_barometer,
                              band_order_Nth=3)

            self.bar_stft_all.append(self.bar_STFT)
            self.bar_stft_bits_all.append(self.bar_STFT_bits)

        # Create accelerometer
        self.sample_rate_acc = 30
        self.signal_time_acc = np.arange(self.start_time, self.end_time, 1/(self.sample_rate_acc/3))
        self.length_for_signal = np.arange(self.start_time, self.end_time, 1/self.sample_rate_acc)
        self.sinewave_acc_base = self.amplitude * np.sin(2 * np.pi * self.frequency * self.length_for_signal)
        self.points_per_row = int(len(self.sinewave_acc_base)/3)
        self.sinewave_acc = self.sinewave_acc_base.reshape((3, self.points_per_row))

        # Create accelerometer mesh
        self.acc_stft_all = []
        self.acc_stft_bits_all = []

        for dimension in range(3):
            self.acc_STFT, self.acc_STFT_bits, _, _ = \
                stft_from_sig(sig_wf=self.sinewave_acc[dimension],
                              frequency_sample_rate_hz=self.sample_rate_acc,
                              band_order_Nth=3)

            self.acc_stft_all.append(self.acc_STFT)
            self.acc_stft_bits_all.append(self.acc_STFT_bits)

        # Create df
        self.dict_to_df_multiple = {0: {"station_id": "1234567890",
                                        "audio_stft": self.audio_STFT,
                                        "audio_stft_bits": self.audio_STFT_bits},
                                    1: {"station_id": "2345678901",   # Add another station
                                        "audio_stft": self.audio_STFT,
                                        "audio_stft_bits": self.audio_STFT_bits}}
        self.df_data = pd.DataFrame(self.dict_to_df_multiple).T

        # Make bar/acc mesh arrays, add them to lists, add column to df.
        self.bar_stft_array = np.array(self.bar_stft_all)
        self.bar_stft_bits_array = np.array(self.bar_stft_bits_all)

        self.bar_stft_all_for_both_stations = []
        self.bar_stft_bits_all_for_both_stations = []
        self.bar_stft_time_all_for_both_stations = []
        self.bar_stft_frequency_all_for_both_stations = []

        self.acc_stft_array = np.array(self.acc_stft_all)
        self.acc_stft_bits_array = np.array(self.acc_stft_bits_all)

        self.acc_stft_all_for_both_stations = []
        self.acc_stft_bits_all_for_both_stations = []

        # Only one station has acc and bar data
        self.acc_stft_all_for_both_stations.append(float("Nan"))
        self.acc_stft_all_for_both_stations.append(self.acc_stft_array)

        self.acc_stft_bits_all_for_both_stations.append(float("Nan"))
        self.acc_stft_bits_all_for_both_stations.append(self.acc_stft_bits_array)

        self.bar_stft_all_for_both_stations.append(float("Nan"))
        self.bar_stft_all_for_both_stations.append(self.bar_stft_array)

        self.bar_stft_bits_all_for_both_stations.append(float("Nan"))
        self.bar_stft_bits_all_for_both_stations.append(self.bar_stft_bits_array)

        # Add columns with mesh
        self.df_data["barometer_stft"] = self.bar_stft_all_for_both_stations
        self.df_data["barometer_stft_bits"] = self.bar_stft_bits_all_for_both_stations

        self.df_data["accelerometer_stft"] = self.acc_stft_all_for_both_stations
        self.df_data["accelerometer_stft_bits"] = self.acc_stft_bits_all_for_both_stations

    def test_wiggles_acc_is_3(self):
        self.wiggle_num = rpd_mesh.find_wiggle_num_tfr(df=self.df_data,
                                                       mesh_tfr_label=["accelerometer_stft_bits"])
        self.assertEqual(self.wiggle_num, 3)

    def test_wiggles_aud_acc_is_5(self):
        self.wiggle_num = rpd_mesh.find_wiggle_num_tfr(df=self.df_data,
                                                       mesh_tfr_label=["accelerometer_stft_bits", "audio_stft_bits"])
        self.assertEqual(self.wiggle_num, 5)

    def test_wiggles_barometer_is_1(self):
        self.wiggle_num = rpd_mesh.find_wiggle_num_tfr(df=self.df_data,
                                                       mesh_tfr_label=["barometer_stft_bits"])
        self.assertEqual(self.wiggle_num, 1)

    def test_wiggles_aud_bar_acc_is_6(self):
        self.wiggle_num = rpd_mesh.find_wiggle_num_tfr(df=self.df_data,
                                                       mesh_tfr_label=["accelerometer_stft_bits", "audio_stft_bits",
                                                                       "barometer_stft_bits"])
        self.assertEqual(self.wiggle_num, 6)

    def tearDown(self):
        self.start_time = None
        self.end_time = None
        self.sample_rate_audio = None
        self.signal_time_audio = None
        self.frequency = None
        self.amplitude = None
        self.sinewave_audio = None
        self.audio_STFT = None
        self.audio_STFT_bits = None
        self.sample_rate_acc = None
        self.signal_time_acc = None
        self.length_for_signal = None
        self.sinewave_acc_base = None
        self.points_per_row = None
        self.sinewave_acc = None
        self.acc_stft_all = None
        self.acc_stft_bits_all = None
        self.acc_STFT = None
        self.acc_STFT_bits = None
        self.dict_to_df_multiple = None
        self.df_data = None
        self.acc_stft_array = None
        self.acc_stft_bits_array = None
        self.acc_stft_all_for_both_stations = None
        self.acc_stft_bits_all_for_both_stations = None
        self.wiggle_num = None
        self.sample_rate_barometer = None
        self.signal_time_barometer = None
        self.sinewave_barometer_base = None
        self.sinewave_barometer = None
        self.bar_stft_all = None
        self.bar_stft_bits_all = None
        self.bar_STFT = None
        self.bar_STFT_bits = None
        self.bar_stft_array = None
        self.bar_stft_bits_array = None
        self.bar_stft_all_for_both_stations = None
        self.bar_stft_bits_all_for_both_stations = None


if __name__ == '__main__':
    unittest.main()
