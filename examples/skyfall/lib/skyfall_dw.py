"""
Loads DataWindow from Redvox data files or compressed files
This is a helper function for other skyfall example files
"""

# Python libraries
import os.path
import pandas as pd

# RedVox RedPandas and related RedVox modules
from redvox.common.data_window import DataWindow
import redpandas.redpd_df as rpd_df
import redpandas.redpd_datawin as rpd_dw

# Configuration files
from redpandas.redpd_config import DataLoadMethod
from examples.skyfall.skyfall_config_file import skyfall_config


LOADED_DF = None


def dw_main(load_method: DataLoadMethod):
    """
    :return: skyfall dataframe; exits if dataframe can't be found
    """
    global LOADED_DF

    if LOADED_DF is None:
        # Load data options
        if load_method == DataLoadMethod.DATAWINDOW or load_method == DataLoadMethod.PICKLE:
            print("Initiating Conversion from RedVox DataWindow to RedVox RedPandas:")
            if load_method == DataLoadMethod.DATAWINDOW:  # Option A: Create DataWindow object
                print("Constructing RedVox DataWindow...", end=" ")

                rdvx_data = rpd_dw.dw_from_redpd_config(config=skyfall_config)

            else:  # Option B: Load pickle with DataWindow object. Assume compressed
                print("Unpickling existing compressed RedVox DataWindow with JSON...", end=" ")
                rdvx_data: DataWindow = DataWindow.load(os.path.join(skyfall_config.output_dir,
                                                                     skyfall_config.output_filename_pkl_pqt))
            print(f"Done. RedVox SDK version: {rdvx_data.sdk_version()}")

            # For option A or B, begin RedPandas
            LOADED_DF = rpd_df.redpd_dataframe(rdvx_data, skyfall_config.sensor_labels)

        elif load_method == DataLoadMethod.PARQUET:  # Option C: Open dataframe from parquet file
            print("Loading existing RedPandas Parquet...", end=" ")
            LOADED_DF = pd.read_parquet(os.path.join(skyfall_config.output_dir, skyfall_config.pd_pqt_file))
            print(f"Done. RedVox SDK version: {LOADED_DF['redvox_sdk_version'][0]}")

        else:
            print('\nNo data loading method selected.  Data is required to run program; will now exit.')
            exit(1)

    return LOADED_DF
