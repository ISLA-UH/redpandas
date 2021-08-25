import unittest
import numpy as np
from libquantum.spectra import stft_from_sig


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

        self.audio_STFT, self.audio_STFT_bits, self.audio_time_stft_s, self.audio_frequency_stft_hz = \
            stft_from_sig(sig_wf=self.sinewave_audio,
                          frequency_sample_rate_hz=self.sample_rate_audio,
                          band_order_Nth=3)

        # Create accelerometer
        self.sample_rate_acc = 30
        self.signal_time_acc = np.arange(self.start_time, self.end_time, 1/(self.sample_rate_acc/3))
        self.length_for_signal = np.arange(self.start_time, self.end_time, 1/self.sample_rate_acc)
        self.sinewave_acc_base = self.amplitude * np.sin(2 * np.pi * self.frequency * self.length_for_signal)
        self.points_per_row = int(len(self.sinewave_acc_base)/3)
        self.sinewave_acc = self.sinewave_acc_base.reshape((3, self.points_per_row))

        self.acc_stft_all = []
        self.acc_stft_time_all = []
        self.acc_stft_bits_all = []
        self.acc_stft_frequency_all = []
        for dimension in range(self.sinewave_acc.ndim):
            self.acc_STFT, self.acc_STFT_bits, self.acc_time_stft_s, self.acc_frequency_stft_hz = \
                stft_from_sig(sig_wf=self.sinewave_acc[dimension],
                              frequency_sample_rate_hz=self.sample_rate_acc,
                              band_order_Nth=3)
            self.acc_stft_all.append(self.acc_STFT)
            self.acc_stft_bits_all.append(self.acc_STFT_bits)
            self.acc_stft_time_all.append(self.acc_time_stft_s)
            self.acc_stft_frequency_all.append(self.acc_frequency_stft_hz)

        # Create df
        self.dict_to_df_single = {0: {"station_id": "1234567890",
                                      "audio_sensor_name": "synch_audio",
                                      "audio_sample_rate_nominal_hz": self.sample_rate_audio,
                                      "audio_epoch_s": self.signal_time_audio,
                                      "audio_wf": self.sinewave_audio,
                                      "audio_stft":self.audio_STFT,
                                      "audio_stft_bits":self.audio_STFT_bits,
                                      "audio_stft_time_s":self.audio_time_stft_s,
                                      "audio_stft_frequency_hz":self.audio_frequency_stft_hz,
                                      "accelerometer_epoch_s": self.signal_time_acc,
                                      "accelerometer_wf_raw": self.sinewave_acc,
                                      "accelerometer_stft": self.acc_stft_frequency_all,
                                      "accelerometer_stft_bits": self.acc_stft_bits_all,
                                      "accelerometer_stft_time_s":self.acc_stft_time_all,
                                      "accelerometer_stft_frequency_s": self.acc_stft_frequency_all}}

        self.dict_to_df_multiple = {0: {"station_id": "1234567890",
                                        "audio_sensor_name": "synch_audio",
                                        "audio_sample_rate_nominal_hz": self.sample_rate_audio,
                                        "audio_epoch_s": self.signal_time_audio,
                                        "audio_wf": self.sinewave_audio,
                                        "audio_stft":self.audio_STFT,
                                        "audio_stft_bits":self.audio_STFT_bits,
                                        "audio_stft_time_s":self.audio_time_stft_s,
                                        "audio_stft_frequency_hz":self.audio_frequency_stft_hz,
                                        "accelerometer_epoch_s": self.signal_time_acc,
                                        "accelerometer_wf_raw": self.sinewave_acc,
                                        "accelerometer_stft": self.acc_stft_frequency_all,
                                        "accelerometer_stft_bits": self.acc_stft_bits_all,
                                        "accelerometer_stft_time_s":self.acc_stft_time_all,
                                        "accelerometer_stft_frequency_s": self.acc_stft_frequency_all},
                                    1: {"station_id": "2345678901",   # Add another station
                                        "audio_sensor_name": "synch_audio",
                                        "audio_sample_rate_nominal_hz": self.sample_rate_audio,
                                        "audio_epoch_s": self.signal_time_audio,
                                        "audio_wf": self.sinewave_audio,
                                        "audio_stft":self.audio_STFT,
                                        "audio_stft_bits":self.audio_STFT_bits,
                                        "audio_stft_time_s":self.audio_time_stft_s,
                                        "audio_stft_frequency_hz":self.audio_frequency_stft_hz,
                                        "accelerometer_epoch_s": self.signal_time_acc,
                                        "accelerometer_wf_raw": self.sinewave_acc,
                                        "accelerometer_stft": self.acc_stft_frequency_all,
                                        "accelerometer_stft_bits": self.acc_stft_bits_all,
                                        "accelerometer_stft_time_s":self.acc_stft_time_all,
                                        "accelerometer_stft_frequency_s": self.acc_stft_frequency_all}}



# class TestFindYlabelTfr(unittest.TestCase):

# class TestFindXMaxMinLim(unittest.TestCase):

# class TestFindTfrMaxMinLim(unittest.TestCase):

# class TestPlotMeshPandas(unittest.TestCase):
if __name__ == '__main__':
    unittest.main()
