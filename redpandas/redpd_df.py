"""
Pipeline from RedVox DataWindow to pandas DataFrame parquet.
"""
# Python libraries
import os
from typing import List, Optional

import numpy as np
import pandas as pd
from redvox.common.data_window import DataWindow
import redpandas.redpd_build_station as rpd_build_sta
import redpandas.redpd_scales as rpd_scales


def redpd_dataframe(input_dw: DataWindow,
                    sensor_labels: Optional[List[str]] = None,
                    highpass_type: Optional[str] = 'obspy',
                    frequency_filter_low: Optional[float] = 1./rpd_scales.Slice.T100S,
                    filter_order: Optional[int] = 4,
                    debug: bool = False) -> pd.DataFrame:
    """
    Construct pandas dataframe from RedVox DataWindow. Default sensor extracted is audio, for more options see
    sensor_labels parameter.

    :param input_dw: REQUIRED. Redvox DataWindow
    :param sensor_labels: optional list of strings, list of sensors available ['audio', 'barometer', 'accelerometer',
        'gyroscope', 'magnetometer', 'health', 'location', 'synchronization', 'clock', 'best_location']. For example:
        sensor_labels = ['audio', 'accelerometer']. Default is ["audio"]
    :param highpass_type: optional string, type of highpass applied. One of: 'obspy', 'butter', or 'rc'.
        Default is 'obspy'
    :param frequency_filter_low: optional float, lowest frequency for highpass filter. Default is 100-second periods
    :param filter_order: optional integer, the order of the filter. Default is 4
    :param debug: if True, writes additional information when running.  Default False
    :return: pd.DataFrame
    """
    print("Initiating conversion from RedVox DataWindow to RedPandas:")
    rdvx_data: DataWindow = input_dw

    if type(sensor_labels) is not list:
        if debug:
            print("Labels passed is not a list.  Using audio label only.")
        sensor_labels = ["audio"]
    else:
        for lbl in sensor_labels:
            if lbl not in ['audio', 'barometer', 'accelerometer', 'gyroscope', 'magnetometer', 'health',
                           'location', 'synchronization', 'clock', 'best_location']:
                if debug:
                    print(f"{lbl} not a valid choice.  Valid choices are:\naudio, barometer, accelerometer, "
                          f"gyroscope, magnetometer, health, location, synchronization, clock, best_location")

    # BEGIN RED PANDAS
    print("\nInitiating RedVox Redpandas:")
    df_all_sensors_all_stations = \
        pd.DataFrame([rpd_build_sta.station_to_dict_from_dw(station=station,
                                                            sdk_version=rdvx_data.sdk_version(),
                                                            sensor_labels=sensor_labels,
                                                            highpass_type=highpass_type,
                                                            frequency_filter_low=frequency_filter_low,
                                                            filter_order=filter_order)
                      for station in rdvx_data.stations()])
    # df_all_sensors_all_stations.sort_values(by="station_id", ignore_index=True, inplace=True)

    # Offer glimpse of what the DataFrame contains
    print(f"\nTotal stations in DataFrame: {len(df_all_sensors_all_stations['station_id'])}")
    print(f"Available stations: \n{df_all_sensors_all_stations['station_id'].to_string(index=False)}")
    print(f"Total columns in DataFrame: {len(df_all_sensors_all_stations.columns)}")

    return df_all_sensors_all_stations


def export_df_to_parquet(df: pd.DataFrame,
                         output_dir_pqt: str,
                         output_filename_pqt: Optional[str] = None,
                         event_name: Optional[str] = "Redvox") -> str:
    """
    Export RedPandas DataFrame to parquet

    :param df: input pandas DataFrame. REQUIRED
    :param output_dir_pqt: string, output directory for parquet. REQUIRED
    :param output_filename_pqt: optional string for parquet filename. Default is None
    :param event_name: optional string with name of event. Default is "Redvox"
    :return: string with full path (output directory and filename) of parquet
    """
    for column in df.columns:
        for row in df.index:  # check all rows, look into dtypes
            check = np.shape(df[column][row])
            if len(check) >= 2:
                # Create new columns with shape tuple for future unflattening/reshaping
                df[[f'{column}_ndim']] = df[[f'{column}']].applymap(np.shape)
                # Change tuples to 1D np.array to save it to parquet
                df[[f'{column}_ndim']] = df[[f'{column}_ndim']].applymap(np.asarray)
                # Flatten each row in wf columns
                df[[f'{column}']] = df[[f'{column}']].applymap(np.ravel)
                break
    # Make filename if not given
    if output_filename_pqt is None:
        output_filename_pqt: str = event_name + "_df.parquet"
    elif output_filename_pqt.find(".parquet") == -1 and output_filename_pqt.find(".pqt") == -1:
        output_filename_pqt += ".parquet"

    if not os.path.exists(output_dir_pqt):
        os.makedirs(output_dir_pqt)
    full_output_dir_path_parquet = os.path.join(output_dir_pqt, output_filename_pqt)
    df.to_parquet(full_output_dir_path_parquet)
    print(f"\nExported Parquet RedPandas DataFrame to {full_output_dir_path_parquet}")

    return full_output_dir_path_parquet
