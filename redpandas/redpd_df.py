"""
Pipeline from RedVox DataWindow to pandas DataFrame parquet.
"""

# Python libraries
import os
from typing import List, Optional, Union, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# RedVox and Red Pandas modules
# from redvox.common.data_window import DataWindow
from redvox.common.data_window import DataWindow
import redpandas.redpd_datawin as rpd_dw
import redpandas.redpd_dq as rpd_dq
import redpandas.redpd_build_station as rpd_build_sta
from redpandas.redpd_config import RedpdConfig
import redpandas.redpd_scales as rpd_scales
import redvox.common.date_time_utils as dt_utils


# def redpd_dataframe_from_config(config: RedpdConfig,
#                                 export_dw_pickle: bool = False,
#                                 export_df_parquet: bool = False,
#                                 print_dq: bool = False,
#                                 show_raw_waveform_plots: bool = True,
#                                 highpass_type: str = 'obspy',
#                                 frequency_filter_low: float = 1./rpd_scales.Slice.T100S,
#                                 filter_order: int = 4):
#     """
#     Extract RedVox data, convert to pandas dataframe and save in parquet
#
#     :param config:
#     :param export_dw_pickle: optional bool, export RedVox DataWindow to pickle if True. Default is False
#     :param export_df_parquet: optional bool, export created RedPandas Dataframe to parquet if True. Default is False
#     :param print_dq: optional bool, print data quality statements. Default is False
#     :param show_raw_waveform_plots: optional bool, plot and show raw waveforms if True. Default is True
#     :param highpass_type: optional string, type of highpass applied. One of: {'obspy', 'butter', or 'rc'}. Default is 'obspy'
#     :param frequency_filter_low: optional float, lowest frequency for highpass filter. Default is 100 second periods
#     :param filter_order: optional integer, the order of the filter. Default is 4
#
#     :return: pd.DataFrame, string with output directory for parquet if export_df_parquet = True, string with output directory
#         for pickle if export_dw_pickle = True
#     """
#     df_all_sensors_all_stations, full_path_parquet, full_path_pickle = \
#         redpd_dataframe(input_dw=config.input_dir,
#                         export_dw_pickle=export_dw_pickle,
#                         export_df_parquet=export_df_parquet,
#                         event_name=config.event_name,
#                         print_dq=print_dq,
#                         show_raw_waveform_plots=show_raw_waveform_plots,
#                         output_dir=config.output_dir,
#                         output_filename_pkl=config.dw_file,
#                         output_filename_pqt=config.pd_pqt_file,
#                         station_ids=config.station_ids,
#                         sensor_labels=config.sensor_labels,
#                         start_epoch_s=config.event_start_epoch_s,
#                         end_epoch_s=config.event_end_epoch_s,
#                         highpass_type=highpass_type,
#                         frequency_filter_low=frequency_filter_low,
#                         filter_order=filter_order)
#
#     return df_all_sensors_all_stations, full_path_parquet, full_path_pickle


def redpd_dataframe(input_dw: DataWindow,
                    sensor_labels: Optional[List[str]] = ["audio"],
                    highpass_type: Optional[str] = 'obspy',
                    frequency_filter_low: Optional[float] = 1./rpd_scales.Slice.T100S,
                    filter_order: Optional[int] = 4) -> pd.DataFrame:
    """
    Construct pandas dataframe from RedVox DataWindow. Default sensor extracted is audio, for more options see sensor_labels parameter.

    :param input_dw: REQUIRED. Redvox DataWindow
    :param sensor_labels: optional list of strings, list of sensors available ['audio', 'barometer', 'accelerometer',
        'gyroscope', 'magnetometer', 'health', 'location', 'synchronization', 'best_location']. For example: sensor_labels = ['audio', 'accelerometer'].
        Default is ["audio"]
    :param highpass_type: optional string, type of highpass applied. One of: 'obspy', 'butter', or 'rc'. Default is 'obspy'
    :param frequency_filter_low: optional float, lowest frequency for highpass filter. Default is 100 second periods
    :param filter_order: optional integer, the order of the filter. Default is 4

    :return: pd.DataFrame
    """
    print("Initiating conversion from RedVox DataWindow to RedPandas:")
    rdvx_data: DataWindow = input_dw

    if type(sensor_labels) is not list:
        sensor_labels = ["audio"]

    # BEGIN RED PANDAS
    print("\nInitiating RedVox Redpandas:")
    df_all_sensors_all_stations = pd.DataFrame([rpd_build_sta.station_to_dict_from_dw(station=station,
                                                                                      sdk_version=rdvx_data.sdk_version(),
                                                                                      sensor_labels=sensor_labels,
                                                                                      highpass_type=highpass_type,
                                                                                      frequency_filter_low=frequency_filter_low,
                                                                                      filter_order=filter_order)
                                                for station in rdvx_data.stations()])
    df_all_sensors_all_stations.sort_values(by="station_id", ignore_index=True, inplace=True)

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
    if not os.path.exists(output_dir_pqt):  # make output directory if it doesn't exist
        print(f"Creating output directory: {output_dir_pqt}...")
        os.mkdir(output_dir_pqt)

    key_sensors = {'barometer', 'accelerometer', 'gyroscope', 'magnetometer'}
    for label in key_sensors:
        # Create new columns with shape tuple for future unflattening/reshaping
        if f'{label}_wf_raw' in df.columns:
            # Create new columns with shape tuple for future unflattening/reshaping
            df[[f'{label}_wf_raw_ndim']] = df[[f'{label}_wf_raw']].applymap(np.shape)
            # Change tuples to 1D np.array to save it to parquet
            df[[f'{label}_wf_raw_ndim']] = df[[f'{label}_wf_raw_ndim']].applymap(np.asarray)
            # Flatten each row in wf columns
            df[[f'{label}_wf_raw']] = df[[f'{label}_wf_raw']].applymap(np.ravel)

        if f'{label}_wf_highpass' in df.columns:
            # Create new columns with shape tuple for future unflattening/reshaping
            df[[f'{label}_wf_highpass_ndim']] = df[[f'{label}_wf_highpass']].applymap(np.shape)
            # Change tuples to 1D np.array to save it to parquet
            df[[f'{label}_wf_highpass_ndim']] = df[[f'{label}_wf_highpass_ndim']].applymap(np.asarray)
            # Flatten each row in wf columns
            df[[f'{label}_wf_highpass']] = df[[f'{label}_wf_highpass']].applymap(np.ravel)

        if f'{label}_nans' in df.columns:
            # Create new columns with shape tuple for future unflattening/reshaping
            df[[f'{label}_nans_ndim']] = df[[f'{label}_nans']].applymap(np.shape)
            # Change tuples to 1D np.array to save it to parquet
            df[[f'{label}_nans_ndim']] = df[[f'{label}_nans_ndim']].applymap(np.asarray)
            # Flatten each row in wf columns
            df[[f'{label}_nans']] = df[[f'{label}_nans']].applymap(np.ravel)

    # Make filename if non given
    if output_filename_pqt is None:
        output_filename_pqt: str = event_name + "_df.parquet"

    if output_filename_pqt.find(".parquet") == -1 and output_filename_pqt.find(".pqt") == -1:
        full_output_dir_path_parquet = os.path.join(output_dir_pqt, output_filename_pqt + ".parquet")
    else:
        full_output_dir_path_parquet = os.path.join(output_dir_pqt, output_filename_pqt)

    df.to_parquet(full_output_dir_path_parquet)
    print(f"\nExported Parquet RedPandas DataFrame to {full_output_dir_path_parquet}")

    return full_output_dir_path_parquet
