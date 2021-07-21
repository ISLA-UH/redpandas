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
from redvox.common.data_window import DataWindow
import redpandas.redpd_datawin as rpd_dw
import redpandas.redpd_dq as rpd_dq
import redpandas.redpd_build_station as rpd_build_sta
from redpandas.redpd_config import RedpdConfig
import redpandas.redpd_scales as rpd_scales
import redvox.common.date_time_utils as dt_utils


def redpd_dataframe_from_config(config: RedpdConfig,
                                export_dw_pickle: bool = False,
                                export_df_parquet: bool = False,
                                print_dq: bool = False,
                                show_raw_waveform_plots: bool = True,
                                highpass_type: str = 'obspy',
                                frequency_filter_low: float = 1./rpd_scales.Slice.T100S,
                                filter_order: int = 4):
    """
    Extract RedVox data, convert to pandas dataframe and save in parquet

    :param config:
    :param export_dw_pickle: optional bool, export RedVox DataWindow to pickle if True. Default is False
    :param export_df_parquet: optional bool, export created RedPandas Dataframe to parquet if True. Default is False
    :param print_dq: optional bool, print data quality statements. Default is False
    :param show_raw_waveform_plots: optional bool, plot and show raw waveforms if True. Default is True
    :param highpass_type: optional string, type of highpass applied. One of: {'obspy', 'butter', or 'rc'}. Default is 'obspy'
    :param frequency_filter_low: optional float, lowest frequency for highpass filter. Default is 100 second periods
    :param filter_order: optional integer, the order of the filter. Default is 4

    :return: pd.DataFrame, string with output directory for parquet if export_df_parquet = True, string with output directory
        for pickle if export_dw_pickle = True
    """
    df_all_sensors_all_stations, full_path_parquet, full_path_pickle = \
        redpd_dataframe(input_dw_or_path=config.input_dir,
                        export_dw_pickle=export_dw_pickle,
                        export_df_parquet=export_df_parquet,
                        event_name=config.event_name,
                        print_dq=print_dq,
                        show_raw_waveform_plots=show_raw_waveform_plots,
                        output_dir=config.output_dir,
                        output_filename_pkl=config.dw_file,
                        output_filename_pqt=config.pd_pqt_file,
                        station_ids=config.station_ids,
                        sensor_labels=config.sensor_labels,
                        start_epoch_s=config.event_start_epoch_s,
                        end_epoch_s=config.event_end_epoch_s,
                        highpass_type=highpass_type,
                        frequency_filter_low=frequency_filter_low,
                        filter_order=filter_order)

    return df_all_sensors_all_stations, full_path_parquet, full_path_pickle


def redpd_dataframe(input_dw_or_path: Union[str, DataWindow],
                    export_dw_pickle: Optional[bool] = False,
                    export_df_parquet: Optional[bool] = False,
                    event_name: Optional[str] = "Redvox",
                    print_dq: Optional[bool] = False,
                    show_raw_waveform_plots: bool = True,
                    output_dir: Optional[str] = None,
                    output_filename_pkl: Optional[str] = None,
                    output_filename_pqt: Optional[str] = None,
                    station_ids: Optional[List[str]] = None,
                    sensor_labels: Optional[List[str]] = None,
                    start_epoch_s: Optional[float] = None,
                    end_epoch_s: Optional[float] = None,
                    start_buffer_minutes: Optional[int] = 3,
                    end_buffer_minutes: Optional[int] = 3,
                    debug: Optional[bool] = False,
                    highpass_type: Optional[str] = 'obspy',
                    frequency_filter_low: Optional[float] = 1./rpd_scales.Slice.T100S,
                    filter_order: Optional[int] = 4) -> Tuple[pd.DataFrame, Union[str, None], Union[str, None]]:
    """
    Extract RedVox data from raw or RedVox DataWindow, and construct pandas dataframe. Note:
        - Default sensor extracted is audio, for more options see sensor_labels parameter.
        - Default all stations included, for filtering specific stations see station_ids parameter.
        - To export RedVox DataWindow to pickle, change export_dw_pickle parameter to export_dw_pickle = True
        - To export RedPandas DataFrame to parquet, change export_df_parquet parameter to export_df_parquet = True

    :param input_dw_or_path: REQUIRED. Redvox DataWindow, or string with directory that contains the files to read data from
    :param export_dw_pickle: optional bool, export RedVox DataWindow to pickle if True. Default is False
    :param export_df_parquet: optional bool, export created RedPandas Dataframe to parquet if True. Default is False
    :param event_name: optional string, name of event. Default is "Redvox"
    :param print_dq: optional bool, print data quality statements. Default is False
    :param show_raw_waveform_plots: optional bool, plot and show raw waveforms if True. Default is True
    :param output_dir: optional string, directory to created save pickle/JSON/parquet. Default is None
    :param output_filename_pkl: optional string, name of created parquet and pickle files. Default is None
    :param output_filename_pqt: optional list of strings, list of station ids to filter on. Default is None
    :param station_ids: optional list of strings, list of station ids to filter on. Default is None, so all stations included
    :param sensor_labels: optional list of strings, list of sensors available ['audio', 'barometer', 'accelerometer',
        'gyroscope', 'magnetometer', 'health', 'location', 'image']. For example: sensor_labels = ['audio', 'accelerometer'].
        Default is ["audio"]
    :param start_epoch_s: optional float, start time in epoch s. Default is None
    :param end_epoch_s: optional float, end time in epoch s. Default is None
    :param start_buffer_minutes: float representing the amount of minutes to include before the start datetime
        when filtering data. Default is 3
    :param end_buffer_minutes: float representing the amount of minutes to include after the end datetime
        when filtering data. Default is 3
    :param debug: optional bool, print debug for DataWindow if True. Default is False
    :param highpass_type: optional string, type of highpass applied. One of: 'obspy', 'butter', or 'rc'. Default is 'obspy'
    :param frequency_filter_low: optional float, lowest frequency for highpass filter. Default is 100 second periods
    :param filter_order: optional integer, the order of the filter. Default is 4

    :return: pd.DataFrame, string with output directory for parquet if export_df_parquet = True, string with output directory for pickle
    if export_dw_pickle = True
    """
    print("Initiating conversion from RedVox DataWindow to RedPandas:")

    if sensor_labels is None:
        sensor_labels = ["audio"]

    if export_dw_pickle is True or export_df_parquet is True:
        if type(input_dw_or_path) is str:
            if output_filename_pkl is None:
                output_filename_pkl: str = event_name
            else:
                # make sure the .pkl does not repeat when we save the DataWindow
                output_filename_pkl = output_filename_pkl.replace(".pkl", "")

            # set output dir for DataWindow pickle/JSON and parquet data products
            if output_dir is None:
                output_dw_pqt_dir = os.path.join(input_dw_or_path, "rpd_files")
            else:
                output_dw_pqt_dir = output_dir
        else:
            if output_dir is None:
                raise ValueError("Please provide a directory in output_dir parameter")
            else:
                output_dw_pqt_dir = output_dir

    if type(input_dw_or_path) == str:

        if start_epoch_s is not None and end_epoch_s is not None:
            start_epoch_s = dt_utils.datetime_from_epoch_seconds_utc(start_epoch_s)
            end_epoch_s = dt_utils.datetime_from_epoch_seconds_utc(end_epoch_s)

        # Load signals, create a RedVox DataWindow structure
        print("Loading data with RedVox DataWindow...", end=" ")
        rdvx_data: DataWindow = DataWindow(input_dir=input_dw_or_path,
                                           structured_layout=True,
                                           start_datetime=start_epoch_s,
                                           end_datetime=end_epoch_s,
                                           station_ids=station_ids,
                                           start_buffer_td=dt_utils.timedelta(minutes=start_buffer_minutes),
                                           end_buffer_td=dt_utils.timedelta(minutes=end_buffer_minutes),
                                           apply_correction=True,
                                           debug=debug)
        print("Done.")

    else:
        # Load DataWindow
        rdvx_data: DataWindow = input_dw_or_path

    if export_dw_pickle is True:
        full_path_pickle = rpd_dw.export_dw_to_pickle(api_input_directory=input_dw_or_path,
                                                      dw=rdvx_data,
                                                      output_directory=output_dir,
                                                      output_filename=output_filename_pkl,
                                                      event_name=event_name)
    else:
        full_path_pickle = None

    # Print out basic stats
    if print_dq:
        print("\nDQ/DA LAYER: STATION")
        rpd_dq.station_metadata(rdvx_data)
        print("DQ/DA LAYER: MIC & SYNCH")
        rpd_dq.mic_sync(rdvx_data)
        print("DQ/DA LAYER: SENSOR TIMING")
        rpd_dq.station_channel_timing(rdvx_data)

    # BEGIN RED PANDAS
    print("\nInitiating RedVox Redpandas:")
    df_all_sensors_all_stations = pd.DataFrame([rpd_build_sta.station_to_dict_from_dw(station=station,
                                                                                      sdk_version=rdvx_data.sdk_version,
                                                                                      sensor_labels=sensor_labels,
                                                                                      highpass_type=highpass_type,
                                                                                      frequency_filter_low=frequency_filter_low,
                                                                                      filter_order=filter_order)
                                                for station in rdvx_data.stations])
    df_all_sensors_all_stations.sort_values(by="station_id", ignore_index=True, inplace=True)

    # Offer glimpse of what the DataFrame contains
    print(f"\nTotal stations in DataFrame: {len(df_all_sensors_all_stations['station_id'])}")
    print(f"Available stations: \n{df_all_sensors_all_stations['station_id'].to_string(index=False)}")
    print(f"Total columns in DataFrame: {len(df_all_sensors_all_stations.columns)}")

    if export_df_parquet is True:  # export dataframe to parquet
        full_path_parquet = export_df_to_parquet(df=df_all_sensors_all_stations,
                                                 sensor_labels=sensor_labels,
                                                 output_filename_pqt=output_filename_pqt,
                                                 event_name=event_name,
                                                 output_dir_pqt=output_dw_pqt_dir)
    else:
        full_path_parquet = None

    if show_raw_waveform_plots:
        fig_mic = rpd_dw.plot_dw_mic(data_window=rdvx_data)
        fig_bar = rpd_dw.plot_dw_baro(data_window=rdvx_data)
        plt.show()

    return df_all_sensors_all_stations, full_path_parquet, full_path_pickle


def export_df_to_parquet(df: pd.DataFrame,
                         output_dir_pqt: str,
                         sensor_labels: List[str],
                         output_filename_pqt: Optional[str] = None,
                         event_name: Optional[str] = "Redvox") -> str:
    """
    Export RedPandas DataFrame to parquet

    :param df: input pandas DataFrame. REQUIRED
    :param output_dir_pqt: string, output directory for parquet. REQUIRED
    :param sensor_labels: list of strings, list of sensors available ['audio', 'barometer', 'accelerometer',
        'gyroscope', 'magnetometer', 'health', 'location', 'image']. REQUIRED
    :param output_filename_pqt: optional string for parquet filename. Default is None
    :param event_name: optional string with name of event. Default is "Redvox"

    :return: string with full path (output directory and filename) of parquet
    """

    # Need to flatten to save to parquet
    for label in sensor_labels:
        if label in ['barometer', 'accelerometer', 'gyroscope', 'magnetometer']:

            if f'{label}_wf_raw' in df.columns:  # make sure there is raw data first
                # Create new columns with shape tuple for future unflattening/reshaping
                df[[f'{label}_wf_raw_ndim',
                    f'{label}_wf_highpass_ndim',
                    f'{label}_nans_ndim']] = \
                    df[[f'{label}_wf_raw',
                        f'{label}_wf_highpass',
                        f'{label}_nans']].applymap(np.shape)

                # Change tuples to 1D np.array to save it to parquet
                df[[f'{label}_wf_raw_ndim',
                    f'{label}_wf_highpass_ndim',
                    f'{label}_nans_ndim']] = \
                    df[[f'{label}_wf_raw_ndim',
                        f'{label}_wf_highpass_ndim',
                        f'{label}_nans_ndim']].applymap(np.asarray)

                # Flatten each row in wf columns
                df[[f'{label}_wf_raw',
                    f'{label}_wf_highpass',
                    f'{label}_nans']] = \
                    df[[f'{label}_wf_raw',
                        f'{label}_wf_highpass',
                        f'{label}_nans']].applymap(np.ravel)

    # Export pandas data frame to parquet
    if output_filename_pqt is None:
        output_filename_pqt: str = event_name + "_df.parquet"

    full_output_dir_path_parquet = os.path.join(output_dir_pqt, output_filename_pqt)
    df.to_parquet(full_output_dir_path_parquet)
    print(f"\nExported Parquet RedPandas DataFrame to {full_output_dir_path_parquet}")

    return full_output_dir_path_parquet
