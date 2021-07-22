import unittest
import pandas as pd
import os
import shutil
from redvox.common.data_window import DataWindow
import redpandas.redpd_df as rpd_df


class TestExportDfToParquet(unittest.TestCase):
    def setUp(self) -> None:
        self.df_data = pd.read_parquet("./test_data/aud_bar_acc_mag_gyr_loc_soh_clock_sync_df.parquet")

    def test_usual_case(self):
        self.full_output_dir_path_parquet = rpd_df.export_df_to_parquet(df=self.df_data,
                                                                        output_dir_pqt="./test_data")

        self.assertEqual(self.full_output_dir_path_parquet, "./test_data/Redvox_df.parquet")

    def test_custom_filename_without(self):

        self.full_output_dir_path_parquet = rpd_df.export_df_to_parquet(df=self.df_data,
                                                                        output_dir_pqt="./test_data",
                                                                        output_filename_pqt='myRedpdParquet1')

        self.assertEqual(self.full_output_dir_path_parquet, "./test_data/myRedpdParquet1.parquet")

    def test_custom_filename_with_dot_parquet(self):

        self.full_output_dir_path_parquet = rpd_df.export_df_to_parquet(df=self.df_data,
                                                                        output_dir_pqt="./test_data",
                                                                        output_filename_pqt='myRedpdParquet2.parquet')

        self.assertEqual(self.full_output_dir_path_parquet, "./test_data/myRedpdParquet2.parquet")

    def test_custom_filename_with_dot_pqt(self):

        self.full_output_dir_path_parquet = rpd_df.export_df_to_parquet(df=self.df_data,
                                                                        output_dir_pqt="./test_data",
                                                                        output_filename_pqt='myRedpdParquet3.pqt')

        self.assertEqual(self.full_output_dir_path_parquet, "./test_data/myRedpdParquet3.pqt")

    def tearDown(self) -> None:
        self.df_data = None
        self.full_output_dir_path_parquet = None
        try:
            os.remove("./test_data/Redvox_df.parquet")
            os.remove("./test_data/myRedpdParquet1.parquet")
            os.remove("./test_data/myRedpdParquet2.parquet")
            os.remove("./test_data/myRedpdParquet3.pqt")
        except OSError:
            pass


class TestRedpdDataframe(unittest.TestCase):
    def setUp(self) -> None:
        self.dw: DataWindow = DataWindow.from_json_file(base_dir="./test_data",
                                                        file_name="aud_bar_acc_mag_gyr_loc_soh_clock_sync")

    def test_with_dw(self):
        self.df, _, _ = rpd_df.redpd_dataframe(input_dw_or_path=self.dw, show_raw_waveform_plots=False)
        self.assertEqual(type(self.df), pd.DataFrame)

    def test_with_dw_audio_only(self):
        self.df, _, _ = rpd_df.redpd_dataframe(input_dw_or_path=self.dw, show_raw_waveform_plots=False)
        self.assertEqual(len(self.df.columns), 13)

    def test_with_dw_barometer_only(self):
        self.df, _, _ = rpd_df.redpd_dataframe(input_dw_or_path=self.dw, show_raw_waveform_plots=False,
                                               sensor_labels=["barometer"])
        self.assertEqual(len(self.df.columns), 12)

    def test_with_dw_accelerometer_only(self):
        self.df, _, _ = rpd_df.redpd_dataframe(input_dw_or_path=self.dw, show_raw_waveform_plots=False,
                                               sensor_labels=["accelerometer"])
        self.assertEqual(len(self.df.columns), 12)

    def test_with_dw_gyroscope_only(self):
        self.df, _, _ = rpd_df.redpd_dataframe(input_dw_or_path=self.dw, show_raw_waveform_plots=False,
                                               sensor_labels=["gyroscope"])
        self.assertEqual(len(self.df.columns), 12)

    def test_with_dw_magnetometer_only(self):
        self.df, _, _ = rpd_df.redpd_dataframe(input_dw_or_path=self.dw, show_raw_waveform_plots=False,
                                               sensor_labels=["magnetometer"])
        self.assertEqual(len(self.df.columns), 12)

    def test_with_dw_health_only(self):
        self.df, _, _ = rpd_df.redpd_dataframe(input_dw_or_path=self.dw, show_raw_waveform_plots=False,
                                               sensor_labels=["health"])
        self.assertEqual(len(self.df.columns), 18)

    def test_with_dw_location_only(self):
        self.df, _, _ = rpd_df.redpd_dataframe(input_dw_or_path=self.dw, show_raw_waveform_plots=False,
                                               sensor_labels=["location"])
        self.assertEqual(len(self.df.columns), 20)

    def test_with_dw_synchronization_only(self):
        self.df, _, _ = rpd_df.redpd_dataframe(input_dw_or_path=self.dw, show_raw_waveform_plots=False,
                                               sensor_labels=["synchronization"])
        self.assertEqual(len(self.df.columns), 12)

    def test_with_dw_clock_only(self):
        self.df, _, _ = rpd_df.redpd_dataframe(input_dw_or_path=self.dw, show_raw_waveform_plots=False,
                                               sensor_labels=["clock"])
        self.assertEqual(len(self.df.columns), 14)

    def test_with_dw_sensor_is_string(self):
        with self.assertRaises(TypeError): self.df, _, _ = rpd_df.redpd_dataframe(input_dw_or_path=self.dw,
                                                                                  show_raw_waveform_plots=False,
                                                                                  sensor_labels="barometer")

    def test_with_dw_audio_barometer_accelerometer(self):
        self.df, _, _ = rpd_df.redpd_dataframe(input_dw_or_path=self.dw, show_raw_waveform_plots=False,
                                               sensor_labels=['audio', 'barometer', 'accelerometer'])
        self.assertEqual(len(self.df.columns), 25)

    def test_with_dw_export_pickle_no_output(self):
        with self.assertRaises(ValueError): self.df, _, self.path_pkl = rpd_df.redpd_dataframe(input_dw_or_path=self.dw,
                                                                                               export_dw_pickle=True,
                                                                                               show_raw_waveform_plots=False)

    def test_with_dw_export_parquet_no_output(self):
        with self.assertRaises(ValueError): self.df, _, self.path_pkl = rpd_df.redpd_dataframe(input_dw_or_path=self.dw,
                                                                                               export_df_parquet=True,
                                                                                               show_raw_waveform_plots=False)

    def test_with_dw_export_pickle_with_output_dir(self):
        self.df, _, self.path_pkl = rpd_df.redpd_dataframe(input_dw_or_path=self.dw,
                                                           export_dw_pickle=True,
                                                           show_raw_waveform_plots=False,
                                                           output_dir="rpd_files")
        self.assertEqual(self.path_pkl, "rpd_files/Redvox.pkl")

    def test_with_dw_export_parquet_with_output_dir(self):
        self.df, self.path_pqt, _ = rpd_df.redpd_dataframe(input_dw_or_path=self.dw,
                                                           export_df_parquet=True,
                                                           show_raw_waveform_plots=False,
                                                           output_dir="rpd_files")
        self.assertEqual(self.path_pqt, "rpd_files/Redvox_df.parquet")

    def test_with_dw_export_pickle_with_output_dir_with_filename(self):
        self.df, _, self.path_pkl = rpd_df.redpd_dataframe(input_dw_or_path=self.dw,
                                                           export_dw_pickle=True,
                                                           show_raw_waveform_plots=False,
                                                           output_dir="rpd_files",
                                                           output_filename_pkl="myRedPdDF")
        self.assertEqual(self.path_pkl, "rpd_files/myRedPdDF.pkl")

    def test_with_dw_export_parquet_with_output_dir_with_filename(self):
        self.df, self.path_pqt, _ = rpd_df.redpd_dataframe(input_dw_or_path=self.dw,
                                                           export_df_parquet=True,
                                                           show_raw_waveform_plots=False,
                                                           output_dir="rpd_files",
                                                           output_filename_pqt="myRedPdDF")
        self.assertEqual(self.path_pqt, "rpd_files/myRedPdDF.parquet")

    def test_with_input_dir(self):
        self.df, _, _ = rpd_df.redpd_dataframe(input_dw_or_path="./test_data", show_raw_waveform_plots=False)
        self.assertEqual(type(self.df), pd.DataFrame)

    def tearDown(self) -> None:
        self.df_data = None
        self.path_pqt = None
        self.path_pkl = None
        self.full_output_dir_path_parquet = None
        shutil.rmtree("rpd_files", ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
