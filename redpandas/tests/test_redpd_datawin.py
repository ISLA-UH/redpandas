import unittest
import shutil
import redpandas.redpd_datawin as rpd_dw
from redvox.common.data_window import DataWindow
from matplotlib.figure import Figure


class TesteExportDwToPickle(unittest.TestCase):
    def setUp(self) -> None:
        # Load DataWindow
        self.rdvx_data: DataWindow = DataWindow.from_json_file(base_dir="test_data",
                                                               file_name="aud_bar_acc_mag_gyr_loc_soh_clock_sync")

    def test_input_output_but_create_filename(self):
        self.path_dw_pickle = rpd_dw.export_dw_to_pickle(dw=self.rdvx_data,
                                                         output_filename=None,
                                                         output_directory="myFirstExport",
                                                         event_name="Redvox")

        self.assertEqual(self.path_dw_pickle, "myFirstExport/Redvox.pkl")

    def test_input_output_and_input_filename_without_pkl(self):
        self.path_dw_pickle = rpd_dw.export_dw_to_pickle(dw=self.rdvx_data,
                                                         output_filename="ExportedPickle",
                                                         output_directory="myFirstExport",
                                                         event_name="Redvox")

        self.assertEqual(self.path_dw_pickle, "myFirstExport/ExportedPickle.pkl")

    def test_input_output_and_input_filename_with_pkl(self):
        self.path_dw_pickle = rpd_dw.export_dw_to_pickle(dw=self.rdvx_data,
                                                         output_filename="ExportedPickle.pkl",
                                                         output_directory="myFirstExport",
                                                         event_name="Redvox")

        self.assertEqual(self.path_dw_pickle, "myFirstExport/ExportedPickle.pkl")

    def test_input_output_and_input_filename_with_pickle(self):
        self.path_dw_pickle = rpd_dw.export_dw_to_pickle(dw=self.rdvx_data,
                                                         output_filename="ExportedPickle.pickle",
                                                         output_directory="myFirstExport",
                                                         event_name="Redvox")
        self.assertEqual(self.path_dw_pickle, "myFirstExport/ExportedPickle.pkl")

    def tearDown(self) -> None:
        self.rdvx_data = None
        self.path_dw_pickl = None
        # Eliminate folders created in this test
        shutil.rmtree("myFirstExport", ignore_errors=True)


class TestPlotDwMic(unittest.TestCase):
    def setUp(self) -> None:
        # Load DataWindow
        self.rdvx_data: DataWindow = DataWindow.from_json_file(base_dir="test_data",
                                                               file_name="aud_bar_acc_mag_gyr_loc_soh_clock_sync")

    def test_result_is_figure(self):
        self.fig = rpd_dw.plot_dw_mic(self.rdvx_data)
        self.assertEqual(type(self.fig), Figure)

    def tearDown(self) -> None:
        self.rdvx_data = None
        self.fig = None


class TestPlotDwBar(unittest.TestCase):
    def setUp(self) -> None:
        # Load DataWindow
        self.rdvx_data: DataWindow = DataWindow.from_json_file(base_dir="test_data",
                                                               file_name="aud_bar_acc_mag_gyr_loc_soh_clock_sync")

    def test_result_is_figure(self):
        self.fig = rpd_dw.plot_dw_baro(self.rdvx_data)
        self.assertEqual(type(self.fig), Figure)

    def tearDown(self) -> None:
        self.rdvx_data = None
        self.fig = None


if __name__ == '__main__':
    unittest.main()
