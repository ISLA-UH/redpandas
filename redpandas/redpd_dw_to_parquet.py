"""
Pipeline from RedVox DataWindow to pandas DataFrame parquet.

Last updated: 6 July 2021
"""

# Python libraries
import os
from typing import List, Optional

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


def redpd_dw_to_parquet_from_config(config: RedpdConfig,
                                    create_dw: bool = True,
                                    print_dq: bool = False,
                                    show_raw_waveform_plots: bool = True,
                                    highpass_type: str = 'obspy',
                                    frequency_filter_low: float = 1./rpd_scales.Slice.T100S,
                                    filter_order: int = 4):
    """
    Extract RedVox data, convert to pandas dataframe and save in parquet

    :param config:
    :param create_dw: create RedVox DataWindow to load data. Default if True. If false, open existing pickle file
    :param print_dq: print data quality statements. Default is True
    :param show_raw_waveform_plots: bool = True,
    :param highpass_type: obspy', 'butter', 'rc', default 'obspy'
    :param frequency_filter_low: apply highpass filter. Default is 100 second periods
    :param filter_order: the order of the filter integer. Default is 4
    :return: print data quality statements, build parquet for RedVox data, plot waveforms
    """
    redpd_dw_to_parquet(input_dir=config.input_dir,
                        event_name=config.event_name,
                        create_dw=create_dw,
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


def redpd_dw_to_parquet(input_dir: str,
                        event_name: Optional[str] = "Redvox",
                        create_dw: bool = True,
                        print_dq: bool = False,
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
                        debug: bool = False,
                        highpass_type: str = 'obspy',
                        frequency_filter_low: float = 1./rpd_scales.Slice.T100S,
                        filter_order: int = 4):
    """
    Extract RedVox data, convert to pandas dataframe and save in parquet

    :param input_dir: string, directory that contains the files to read data from.  REQUIRED
    :param event_name: optional string, name of event. Default is "Redvox"
    :param create_dw: create RedVox DataWindow to load data. Default if True. If false, open existing pickle file
    :param print_dq: print data quality statements. Default is True
    :param show_raw_waveform_plots: plot and show raw waveforms. Default is True
    :param output_dir: optional string, directory to created save pickle/JSON/parquet
    :param output_filename_pkl: optional string, name of created parquet and pickle files
    :param output_filename_pqt: optional list of strings, list of station ids to filter on
    :param station_ids: optional list of strings, list of station ids to filter on
    :param sensor_labels: optional list of strings, list of sensors available ['audio', 'barometer', 'accelerometer',
        'gyroscope', 'magnetometer', 'health', 'location', 'image']. Default is "audio"
    :param start_epoch_s: optional float, start time in epoch s. Default is None
    :param end_epoch_s: optional float, end time in epoch s. Default is None
    :param start_buffer_minutes: float representing the amount of minutes to include before the start datetime
        when filtering data. Default is 3
    :param end_buffer_minutes: float representing the amount of minutes to include after the end datetime
        when filtering data. Default is 3
    :param debug: print debug for DataWindow. Default is False
    :param highpass_type: 'obspy', 'butter', 'rc', default 'obspy'
    :param frequency_filter_low: apply highpass filter. Default is 100 second periods
    :param filter_order: the order of the filter integer. Default is 4
    :return: print data quality statements, build parquet for RedVox data, plot waveforms
    """
    print("Initiating conversion from RedVox DataWindow to RedPandas:")

    if sensor_labels is None:
        sensor_labels = ["audio"]

    if output_filename_pkl is None:
        output_filename_pkl: str = event_name
    else:
        # make sure the .pkl does not repeat when we save the DataWindow
        output_filename_pkl = output_filename_pkl.replace(".pkl", "")

    if create_dw:

        # Load signals, create a RedVox DataWindow structure, export to pickle.
        rpd_dw.build(api_input_directory=input_dir,
                     event_name=event_name,
                     output_directory=output_dir,
                     output_filename=output_filename_pkl,
                     redvox_station_ids=station_ids,
                     start_epoch_s=start_epoch_s,
                     end_epoch_s=end_epoch_s,
                     start_buffer_minutes=start_buffer_minutes,
                     end_buffer_minutes=end_buffer_minutes,
                     debug=debug)

    # Import DataWindow
    else:
        print("Unpickling existing compressed RedVox DataWindow with JSON...")

    if output_dir is None:  # set output dir for DataWindow pickle/JSON and parquet
        output_dw_pqt_dir = os.path.join(input_dir, "rpd_files")
    else:
        output_dw_pqt_dir = output_dir

    rdvx_data: DataWindow = DataWindow.from_json_file(base_dir=output_dw_pqt_dir, file_name=output_filename_pkl)

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

    # Need to flatten to save to parquet
    for label in sensor_labels:
        if label in ['barometer', 'accelerometer', 'gyroscope', 'magnetometer']:

            if f'{label}_wf_raw' in df_all_sensors_all_stations.columns:  # make sure there is raw data first
                # Create new columns with shape tuple for future unflattening/reshaping
                df_all_sensors_all_stations[[f'{label}_wf_raw_ndim',
                                             f'{label}_wf_highpass_ndim',
                                             f'{label}_nans_ndim']] = \
                    df_all_sensors_all_stations[[f'{label}_wf_raw',
                                                 f'{label}_wf_highpass',
                                                 f'{label}_nans']].applymap(np.shape)

                # Change tuples to 1D np.array to save it to parquet
                df_all_sensors_all_stations[[f'{label}_wf_raw_ndim',
                                             f'{label}_wf_highpass_ndim',
                                             f'{label}_nans_ndim']] = \
                    df_all_sensors_all_stations[[f'{label}_wf_raw_ndim',
                                                 f'{label}_wf_highpass_ndim',
                                                 f'{label}_nans_ndim']].applymap(np.asarray)

                # Flatten each row in wf columns
                df_all_sensors_all_stations[[f'{label}_wf_raw',
                                             f'{label}_wf_highpass',
                                             f'{label}_nans']] = \
                    df_all_sensors_all_stations[[f'{label}_wf_raw',
                                                 f'{label}_wf_highpass',
                                                 f'{label}_nans']].applymap(np.ravel)

    # Export pandas data frame to parquet
    if output_filename_pqt is None:
        output_filename_pqt: str = event_name + "_df.parquet"

    df_all_sensors_all_stations.to_parquet(os.path.join(output_dw_pqt_dir, output_filename_pqt))
    print("\nExported Parquet RedPandas DataFrame to " + os.path.join(output_dw_pqt_dir, output_filename_pqt))

    # Check that parquet file saves and opens correctly
    df_open = pd.read_parquet(os.path.join(output_dw_pqt_dir, output_filename_pqt))
    print(f"Total stations in DataFrame: \n{len(df_open['station_id'])}")
    print(f"Available stations: {df_open['station_id'].to_string(index=False)}")
    print(f"Total columns in DataFrame: {len(df_open.columns)}")

    # Plot data window waveforms
    if show_raw_waveform_plots:
        rpd_dw.plot_dw_mic(data_window=rdvx_data)
        rpd_dw.plot_dw_baro(data_window=rdvx_data)
        plt.show()


