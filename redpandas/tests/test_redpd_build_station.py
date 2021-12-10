import unittest
import numpy as np
import redpandas.redpd_build_station as rpd_build_sta
from redvox.common.data_window import DataWindow

# # Load data once for speed
# rdvx_data: DataWindow = DataWindow.from_json_file(base_dir="test_data",
#                                                   file_name="aud_bar_acc_mag_gyr_loc_soh_clock_sync")
# example_station = rdvx_data.get_station("1637610021")[0]

# TODO:
#  - Datasets with no: barometer, accelerometer, gyroscope, magnetometer, location, best location, clock, synch, health
#  - Datasets with: image, luminosity
#  - Write tests accordingly


# class TestUnevenSensor(unittest.TestCase):
#     def setUp(self) -> None:
#         self.example_station = example_station
#         sensor_sample_rate_hz, sensor_epoch_s, sensor_raw, sensor_nans = \
#             rpd_build_sta.sensor_uneven(station=self.example_station,
#                                         sensor_label="barometer")
#         self.sensor_sample_rate_hz = sensor_sample_rate_hz
#         self.sensor_epoch_s = sensor_epoch_s
#         self.sensor_raw = sensor_raw
#         self.sensor_nans = sensor_nans
#
#     def test_results_is_not_None(self):
#         # Should not be None as there is data
#         self.assertIsNotNone(self.sensor_sample_rate_hz)
#         self.assertIsNotNone(self.sensor_epoch_s)
#         self.assertIsNotNone(self.sensor_raw)
#         self.assertIsNotNone(self.sensor_nans)
#
#     def test_results_type(self):
#         # Check return type is what it should be
#         self.assertTrue(type(self.sensor_sample_rate_hz) is float)
#         self.assertTrue(type(self.sensor_epoch_s) is np.ndarray)
#         self.assertTrue(type(self.sensor_raw) is np.ndarray)
#         self.assertTrue(type(self.sensor_nans) is np.ndarray)
#
#     def test_no_data(self):
#         # Should return None cause there is no data
#         sensor_sample_rate_hz, sensor_epoch_s, sensor_raw, sensor_nans = \
#             rpd_build_sta.sensor_uneven(station=example_station,
#                                         sensor_label="light")
#         self.assertIsNone(sensor_sample_rate_hz)
#         self.assertIsNone(sensor_epoch_s)
#         self.assertIsNone(sensor_raw)
#         self.assertIsNone(sensor_nans)
#
#     def tearDown(self):
#         self.example_station = None
#         self.sensor_sample_rate_hz = None
#         self.sensor_epoch_s = None
#         self.sensor_raw = None
#         self.sensor_nans = None
#
#
# class TestAudioWfTimeBuildStation(unittest.TestCase):
#     def setUp(self) -> None:
#         self.example_station = rdvx_data.get_station("1637610021")[0]
#
#     def test_result_type_simple(self):
#         # Should return dictionary
#         returned_dict = rpd_build_sta.audio_wf_time_build_station(station=self.example_station,
#                                                                   mean_type="simple",
#                                                                   raw=False)
#         self.assertTrue(type(returned_dict) is dict)
#
#     def test_result_type_lin(self):
#         returned_dict = rpd_build_sta.audio_wf_time_build_station(station=self.example_station,
#                                                                   mean_type="lin",
#                                                                   raw=False)
#         self.assertTrue(type(returned_dict) is dict)
#
#     def test_result_type_raw(self):
#         returned_dict = rpd_build_sta.audio_wf_time_build_station(station=self.example_station,
#                                                                   mean_type="simple",
#                                                                   raw=True)
#         self.assertTrue(type(returned_dict) is dict)
#
#     def tearDown(self):
#         self.example_station = None
#
#
# class TestLocationBuildStation(unittest.TestCase):
#     def setUp(self) -> None:
#         self.example_station = rdvx_data.get_station("1637610021")[0]
#
#     def test_result_type(self):
#         # Should return dictionary
#         returned_dict = rpd_build_sta.location_build_station(station=self.example_station)
#         self.assertTrue(type(returned_dict) is dict)
#
#     def tearDown(self):
#         self.example_station = None
#
#
# class TestBestLocationBuildStation(unittest.TestCase):
#     def setUp(self) -> None:
#         self.example_station = rdvx_data.get_station("1637610021")[0]
#
#     def test_result_type(self):
#         # Should return dictionary
#         returned_dict = rpd_build_sta.best_location_build_station(station=self.example_station)
#         self.assertTrue(type(returned_dict) is dict)
#
#     def tearDown(self):
#         self.example_station = None
#
#
# class TestStateOfHealthBuildStation(unittest.TestCase):
#     def setUp(self) -> None:
#         self.example_station = rdvx_data.get_station("1637610021")[0]
#
#     def test_result_type(self):
#         # Should return dictionary
#         returned_dict = rpd_build_sta.state_of_health_build_station(station=self.example_station)
#         self.assertTrue(type(returned_dict) is dict)
#
#     def tearDown(self):
#         self.example_station = None
#
#
# class TestImageBuildStation(unittest.TestCase):
#     def setUp(self) -> None:
#         self.example_station = rdvx_data.get_station("1637610021")[0]
#
#     def test_result_type(self):
#         # Should return dictionary
#         returned_dict = rpd_build_sta.image_build_station(station=self.example_station)
#         self.assertTrue(type(returned_dict) is dict)
#
#     def tearDown(self):
#         self.example_station = None
#
#
# class TestSynchronizationBuildStation(unittest.TestCase):
#     def setUp(self) -> None:
#         self.example_station = rdvx_data.get_station("1637610021")[0]
#
#     def test_result_type(self):
#         # Should return dictionary
#         returned_dict = rpd_build_sta.synchronization_build_station(station=self.example_station)
#         self.assertTrue(type(returned_dict) is dict)
#
#     def tearDown(self):
#         self.example_station = None
#
#
# class TestClockBuildStation(unittest.TestCase):
#     def setUp(self) -> None:
#         self.example_station = rdvx_data.get_station("1637610021")[0]
#
#     def test_result_type(self):
#         # Should return dictionary
#         returned_dict = rpd_build_sta.clock_build_station(station=self.example_station)
#         self.assertTrue(type(returned_dict) is dict)
#
#     def tearDown(self):
#         self.example_station = None
#
#
# class TestLightBuildStation(unittest.TestCase):
#     def setUp(self) -> None:
#         self.example_station = rdvx_data.get_station("1637610021")[0]
#
#     def test_result_type(self):
#         # Should return dictionary
#         returned_dict = rpd_build_sta.light_build_station(station=self.example_station)
#         self.assertTrue(type(returned_dict) is dict)
#
#     def tearDown(self):
#         self.example_station = None
#
#
# class TestBuildStation(unittest.TestCase):
#
#     def setUp(self) -> None:
#         self.example_station = rdvx_data.get_station("1637610021")[0]
#
#     def test_result_type_audio_obspy(self):
#         returned_dict = rpd_build_sta.build_station(station=self.example_station,
#                                                     sensor_label="audio",
#                                                     highpass_type="obspy",
#                                                     frequency_filter_low=1/100,
#                                                     filter_order=4)
#         self.assertTrue(type(returned_dict) is dict)
#
#     def test_result_type_audio_butter(self):
#         returned_dict = rpd_build_sta.build_station(station=self.example_station,
#                                                     sensor_label="audio",
#                                                     highpass_type="butter",
#                                                     frequency_filter_low=1/100,
#                                                     filter_order=4)
#         self.assertTrue(type(returned_dict) is dict)
#
#     def test_result_type_audio_rc(self):
#         returned_dict = rpd_build_sta.build_station(station=self.example_station,
#                                                     sensor_label="audio",
#                                                     highpass_type="rc",
#                                                     frequency_filter_low=1/100,
#                                                     filter_order=4)
#         self.assertTrue(type(returned_dict) is dict)
#
#     def test_result_type_barometer_obspy(self):
#         returned_dict = rpd_build_sta.build_station(station=self.example_station,
#                                                     sensor_label="barometer",
#                                                     highpass_type="obspy",
#                                                     frequency_filter_low=1/100,
#                                                     filter_order=4)
#         self.assertTrue(type(returned_dict) is dict)
#
#     def test_result_type_accelerometer_obspy(self):
#         returned_dict = rpd_build_sta.build_station(station=self.example_station,
#                                                     sensor_label="accelerometer",
#                                                     highpass_type="obspy",
#                                                     frequency_filter_low=1/100,
#                                                     filter_order=4)
#         self.assertTrue(type(returned_dict) is dict)
#
#     def test_result_type_gyroscope_obspy(self):
#         returned_dict = rpd_build_sta.build_station(station=self.example_station,
#                                                     sensor_label="gyroscope",
#                                                     highpass_type="obspy",
#                                                     frequency_filter_low=1/100,
#                                                     filter_order=4)
#         self.assertTrue(type(returned_dict) is dict)
#
#     def test_result_type_magnetometer_obspy(self):
#         returned_dict = rpd_build_sta.build_station(station=self.example_station,
#                                                     sensor_label="magnetometer",
#                                                     highpass_type="obspy",
#                                                     frequency_filter_low=1/100,
#                                                     filter_order=4)
#         self.assertTrue(type(returned_dict) is dict)
#
#     def test_result_type_health(self):
#         returned_dict = rpd_build_sta.build_station(station=self.example_station,
#                                                     sensor_label="magnetometer",
#                                                     highpass_type="obspy",
#                                                     frequency_filter_low=1/100,
#                                                     filter_order=4)
#         self.assertTrue(type(returned_dict) is dict)
#
#     def test_result_type_location(self):
#         returned_dict = rpd_build_sta.build_station(station=self.example_station,
#                                                     sensor_label="location",
#                                                     highpass_type="obspy",
#                                                     frequency_filter_low=1/100,
#                                                     filter_order=4)
#         self.assertTrue(type(returned_dict) is dict)
#
#     def test_result_type_best_location(self):
#         returned_dict = rpd_build_sta.build_station(station=self.example_station,
#                                                     sensor_label="best_location",
#                                                     highpass_type="obspy",
#                                                     frequency_filter_low=1/100,
#                                                     filter_order=4)
#         self.assertTrue(type(returned_dict) is dict)
#
#     def test_result_type_image(self):
#         returned_dict = rpd_build_sta.build_station(station=self.example_station,
#                                                     sensor_label="image",
#                                                     highpass_type="obspy",
#                                                     frequency_filter_low=1/100,
#                                                     filter_order=4)
#         self.assertTrue(type(returned_dict) is dict)
#
#     def tearDown(self):
#         self.example_station = None
#
#
# class TestStationToDictFromDw(unittest.TestCase):
#
#     def setUp(self) -> None:
#         self.example_station = rdvx_data.get_station("1637610021")[0]
#
#     def test_result_type_audio_only(self):
#         returned_dict = rpd_build_sta.station_to_dict_from_dw(station=self.example_station,
#                                                               sdk_version="3.0",
#                                                               sensor_labels=["audio"],
#                                                               highpass_type="obspy",
#                                                               frequency_filter_low=1/100,
#                                                               filter_order=4)
#         self.assertTrue(type(returned_dict) is dict)
#
#     def test_result_type_barometer_only(self):
#         returned_dict = rpd_build_sta.station_to_dict_from_dw(station=self.example_station,
#                                                               sdk_version="3.0",
#                                                               sensor_labels=["barometer"],
#                                                               highpass_type="obspy",
#                                                               frequency_filter_low=1/100,
#                                                               filter_order=4)
#         self.assertTrue(type(returned_dict) is dict)
#
#     def test_result_type_acceleration_only(self):
#         returned_dict = rpd_build_sta.station_to_dict_from_dw(station=self.example_station,
#                                                               sdk_version="3.0",
#                                                               sensor_labels=["accelerometer"],
#                                                               highpass_type="obspy",
#                                                               frequency_filter_low=1/100,
#                                                               filter_order=4)
#         self.assertTrue(type(returned_dict) is dict)
#
#     def test_result_type_magnetometer_only(self):
#         returned_dict = rpd_build_sta.station_to_dict_from_dw(station=self.example_station,
#                                                               sdk_version="3.0",
#                                                               sensor_labels=["magnetometer"],
#                                                               highpass_type="obspy",
#                                                               frequency_filter_low=1/100,
#                                                               filter_order=4)
#         self.assertTrue(type(returned_dict) is dict)
#
#     def test_result_type_gyroscope_only(self):
#         returned_dict = rpd_build_sta.station_to_dict_from_dw(station=self.example_station,
#                                                               sdk_version="3.0",
#                                                               sensor_labels=["gyroscope"],
#                                                               highpass_type="obspy",
#                                                               frequency_filter_low=1/100,
#                                                               filter_order=4)
#         self.assertTrue(type(returned_dict) is dict)
#
#     def test_result_type_multiple(self):
#         returned_dict = rpd_build_sta.station_to_dict_from_dw(station=self.example_station,
#                                                               sdk_version="3.0",
#                                                               sensor_labels=["audio", "barometer"],
#                                                               highpass_type="obspy",
#                                                               frequency_filter_low=1/100,
#                                                               filter_order=4)
#         self.assertTrue(type(returned_dict) is dict)
#
#     def tearDown(self):
#         self.example_station = None
#
#
# if __name__ == '__main__':
#     unittest.main()
