import unittest
import pandas as pd
import os
import shutil
from redvox.common.data_window import DataWindow
import redpandas.redpd_df as rpd_df
from redpandas.tests import TEST_DATA_DIR


class TestExportDfToParquet(unittest.TestCase):
    def setUp(self) -> None:
        self.df_data = pd.read_parquet(f"{TEST_DATA_DIR}/aud_bar_acc_mag_gyr_loc_soh_clock_sync_df.parquet")

    def test_usual_case(self):
        self.full_output_dir_path_parquet = rpd_df.export_df_to_parquet(df=self.df_data,
                                                                        output_dir_pqt=f"{TEST_DATA_DIR}/TestExportDfToParquet")

        self.assertEqual(self.full_output_dir_path_parquet, f"{TEST_DATA_DIR}/TestExportDfToParquet/Redvox_df.parquet")

    def test_custom_filename_without(self):

        self.full_output_dir_path_parquet = rpd_df.export_df_to_parquet(df=self.df_data,
                                                                        output_dir_pqt=f"{TEST_DATA_DIR}/TestExportDfToParquet",
                                                                        output_filename_pqt='myRedpdParquet1')

        self.assertEqual(self.full_output_dir_path_parquet, f"{TEST_DATA_DIR}/TestExportDfToParquet/myRedpdParquet1.parquet")

    def test_custom_filename_with_dot_parquet(self):

        self.full_output_dir_path_parquet = rpd_df.export_df_to_parquet(df=self.df_data,
                                                                        output_dir_pqt=f"{TEST_DATA_DIR}/TestExportDfToParquet",
                                                                        output_filename_pqt='myRedpdParquet2.parquet')

        self.assertEqual(self.full_output_dir_path_parquet, f"{TEST_DATA_DIR}/TestExportDfToParquet/myRedpdParquet2.parquet")

    def test_custom_filename_with_dot_pqt(self):

        self.full_output_dir_path_parquet = rpd_df.export_df_to_parquet(df=self.df_data,
                                                                        output_dir_pqt=f"{TEST_DATA_DIR}/TestExportDfToParquet",
                                                                        output_filename_pqt='myRedpdParquet3.pqt')

        self.assertEqual(self.full_output_dir_path_parquet, f"{TEST_DATA_DIR}/TestExportDfToParquet/myRedpdParquet3.pqt")

    def tearDown(self) -> None:
        self.df_data = None
        self.full_output_dir_path_parquet = None
        shutil.rmtree(f"{TEST_DATA_DIR}/TestExportDfToParquet", ignore_errors=True)


class TestRedpdDataframe(unittest.TestCase):
    def setUp(self) -> None:
        self.dw: DataWindow = DataWindow.load(file_path=f"{TEST_DATA_DIR}/dw/dw310_skyfall_1_min.json")

    def test_result_type(self):
        self.df = rpd_df.redpd_dataframe(input_dw=self.dw)
        self.assertEqual(type(self.df), pd.DataFrame)

    def test_with_dw_audio_only(self):
        self.df = rpd_df.redpd_dataframe(input_dw=self.dw)
        self.assertEqual(len(self.df.columns), 14)

    def test_with_dw_barometer_only(self):
        self.df = rpd_df.redpd_dataframe(input_dw=self.dw, sensor_labels=["barometer"])
        self.assertEqual(len(self.df.columns), 13)

    def test_with_dw_accelerometer_only(self):
        self.df = rpd_df.redpd_dataframe(input_dw=self.dw, sensor_labels=["accelerometer"])
        self.assertEqual(len(self.df.columns), 13)

    def test_with_dw_gyroscope_only(self):
        self.df = rpd_df.redpd_dataframe(input_dw=self.dw, sensor_labels=["gyroscope"])
        self.assertEqual(len(self.df.columns), 13)

    def test_with_dw_magnetometer_only(self):
        self.df = rpd_df.redpd_dataframe(input_dw=self.dw, sensor_labels=["magnetometer"])
        self.assertEqual(len(self.df.columns), 13)

    def test_with_dw_health_only(self):
        self.df = rpd_df.redpd_dataframe(input_dw=self.dw, sensor_labels=["health"])
        self.assertEqual(len(self.df.columns), 19)

    def test_with_dw_location_only(self):
        self.df = rpd_df.redpd_dataframe(input_dw=self.dw, sensor_labels=["location"])
        self.assertEqual(len(self.df.columns), 21)

    def test_with_dw_synchronization_only(self):
        self.df = rpd_df.redpd_dataframe(input_dw=self.dw, sensor_labels=["synchronization"])
        self.assertEqual(len(self.df.columns), 13)

    def test_with_dw_clock_only(self):
        self.df = rpd_df.redpd_dataframe(input_dw=self.dw, sensor_labels=["clock"])
        self.assertEqual(len(self.df.columns), 15)

    def test_with_dw_sensor_is_not_string_then_do_audio_only(self):
        self.df = rpd_df.redpd_dataframe(input_dw=self.dw, sensor_labels="barometer")
        self.assertEqual(len(self.df.columns), 14)

    def test_with_dw_audio_barometer_accelerometer(self):
        self.df = rpd_df.redpd_dataframe(input_dw=self.dw, sensor_labels=['audio', 'barometer', 'accelerometer'])
        self.assertEqual(len(self.df.columns), 26)

    def tearDown(self) -> None:
        self.df_data = None
        shutil.rmtree(f"{TEST_DATA_DIR}/rpd_files", ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
