# Python libraries
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# RedVox and Red Pandas modules
from redvox.common.data_window import DataWindow
import redvox.settings as settings

import redpandas.redpd_datawin as rpd_dw
import redpandas.redpd_dq as rpd_dq
import redpandas.redpd_build_station as rpd_build_sta

# Configuration file
# from examples.skyfall.skyfall_config import EVENT_NAME, INPUT_DIR, EPISODE_START_EPOCH_S, \
#     EPISODE_END_EPOCH_S, STATIONS, PD_PQT_FILE, OUTPUT_DIR, DW_FILE, build_dw_pickle, build_df_parquet, \
#     plot_mic_waveforms, print_datawindow_dq, SENSOR_LABEL

from redpd_config import RedpdConfig


def redpd_dw_to_parquet_from_config(config: RedpdConfig):
    redpd_dw_to_parquet(config.input_dir, config.dw_file)


def redpd_dw_to_parquet(input_dir: str, output_filename_base: str, compress_dw: bool = True,
                        output_dir: Optional[str] = None, sensor_labels: Optional[List[str]] = None):
    """
    Beta workflow for API M pipeline
    Last updated: 8 June 2021
    """
    print("Initiating Conversion from RedVox DataWindow to RedPandas:")

    settings.set_parallelism_enabled(True)

    if not output_dir:
        output_dir = input_dir

    if not sensor_labels:
        sensor_labels = ["audio"]

    pkl_output: str = output_filename_base + ".pkl"
    pd_pqt_output: str = output_filename_base + "_df.parquet"

    if compress_dw:
        # Load signals, create a RedVox DataWindow structure, export to pickle.
        # rpd_dw.build(api_input_directory=skyfall_config.input_dir,
        #              start_epoch_s=skyfall_config.episode_start_epoch_s,
        #              end_epoch_s=skyfall_config.episode_end_epoch_s,
        #              redvox_station_ids=skyfall_config.stations,
        #              event_name=skyfall_config.event_name,
        #              output_directory=skyfall_config.output_dir,
        #              output_filename=skyfall_config.dw_file,
        #              start_buffer_minutes=3.,
        #              end_buffer_minutes=3.,
        #              debug=True)
        rpd_dw.build_ez(api_input_directory=input_dir, pickle_filename=pkl_output)

    # Import DataWindow
    else:
        print("Unpickling existing compressed RedVox DataWindow with JSON...")
    rdvx_data: DataWindow = DataWindow.from_json_file(base_dir=input_dir, file_name=output_filename_base)
    print(f"RedVox SDK version: {rdvx_data.sdk_version}")

    # Print out basic stats
    # if print_dw_quality:
    #     print("\nDQ/DA LAYER: STATION")
    #     rpd_dq.station_metadata(rdvx_data)
    #     print("DQ/DA LAYER: MIC & SYNCH")
    #     rpd_dq.mic_sync(rdvx_data)
    #     print("DQ/DA LAYER: SENSOR TIMING")
    #     rpd_dq.station_channel_timing(rdvx_data)

    # Plot data window waveforms
    # if plot_mic_waveforms:
    #     rpd_dw.plot_dw_mic(data_window=rdvx_data)
    #     rpd_dw.plot_dw_baro(data_window=rdvx_data)
    #     plt.show()

    # BEGIN RED PANDAS
    print("\nInitiating RedVox Redpandas:")
    df_all_sensors_all_stations = pd.DataFrame([rpd_build_sta.station_to_dict_from_dw(station=station,
                                                                                      sdk_version=rdvx_data.sdk_version,
                                                                                      sensor_labels=sensor_labels)
                                                for station in rdvx_data.stations])
    df_all_sensors_all_stations.sort_values(by="station_id", ignore_index=True, inplace=True)

    # Need to flatten to save to parquet
    for label in sensor_labels:
        if label in ['barometer', 'accelerometer', 'gyroscope', 'magnetometer']:

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
    df_all_sensors_all_stations.to_parquet(os.path.join(input_dir, pd_pqt_output))
    print("\nExported Parquet RedPandas DataFrame to " + os.path.join(input_dir, pd_pqt_output))

    # Check that parquet file saves and opens correctly
    df_open = pd.read_parquet(os.path.join(input_dir, pd_pqt_output))
    print("Total stations in DataFrame:", len(df_open['station_id']))
    print("Available stations:", df_open['station_id'])
    print("Total columns in DataFrame:", len(df_open.columns))
    print("Available columns:", df_open.columns)
