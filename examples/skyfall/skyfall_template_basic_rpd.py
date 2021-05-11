# Python libraries
import os.path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# RedVox and Red Pandas modules
from redvox.common.data_window import DataWindowFast
import redpandas.redpd_datawin as rpd_dw
import redpandas.redpd_dq as rpd_dq
import redpandas.redpd_build_station as rpd_build_sta

# Configuration file
from examples.skyfall.skyfall_config import EVENT_NAME, INPUT_DIR, EPISODE_START_EPOCH_S, \
    EPISODE_END_EPOCH_S, STATIONS, PD_PQT_FILE, OUTPUT_DIR, DW_FILE, build_dw_pickle, build_df_parquet, \
    plot_mic_waveforms, print_datawindow_dq, SENSOR_LABEL


if __name__ == "__main__":
    """
    Beta workflow for API M pipeline
    Last updated: 10 May 2021
    """
    print("Initiating Conversion from DataWindow to RedPandas")

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    if build_dw_pickle:
        print("Building Data Window")
        # RECOMMENDED USE
        # Load signals, create a RedVox DataWindow structure, export to pickle.
        rpd_dw.build_fast(api_input_directory=INPUT_DIR,
                          start_epoch_s=EPISODE_START_EPOCH_S,
                          end_epoch_s=EPISODE_END_EPOCH_S,
                          redvox_station_ids=STATIONS,
                          event_name=EVENT_NAME,
                          output_directory=OUTPUT_DIR,
                          output_filename=DW_FILE,
                          start_buffer_minutes=3.,
                          end_buffer_minutes=3.,
                          debug=True)

    # Import DataWindow
    print("Assume compressed, JSON and pickled DW already built")
    print("\nUsing build_dw_pickle =", build_dw_pickle)
    rdvx_data: DataWindowFast = DataWindowFast.from_json_file(base_dir=OUTPUT_DIR,
                                                              file_name=DW_FILE)

    # Print out basic stats
    if print_datawindow_dq:
        print("\nDQ/DA LAYER: STATION")
        rpd_dq.station_metadata(rdvx_data)
        print("DQ/DA LAYER: MIC & SYNCH")
        rpd_dq.mic_synch(rdvx_data)
        print("DQ/DA LAYER: SENSOR TIMING")
        rpd_dq.station_channel_timing(rdvx_data)

    # Plot data window waveforms
    if plot_mic_waveforms:
        rpd_dw.plot_dw_mic(data_window=rdvx_data)
        rpd_dw.plot_dw_baro(data_window=rdvx_data)
        plt.show()

    # BEGIN RED PANDAS
    list_df_stations = []  # list to store dataframes with sensors for one station

    for station in rdvx_data.stations:

        list_df_sensors_per_station = []  # list to store sensor dataframes for one station

        # TODO: SDK version.
        # TODO: app start time - how is it handled, as an enhanced station_id?
        # TODO: Location provider, how to access
        # TODO: Timing information from DQ
        dict_for_station_id = {'station_id': [station.id],
                               'station_make': [station.metadata.make],
                               'station_model': [station.metadata.model],
                               'station_app_version': [station.metadata.app_version],
                               'datawin_sdk_version': ["3.0.0rc31"]}

        df_station_id = pd.DataFrame.from_dict(data=dict_for_station_id)
        list_df_sensors_per_station.append(df_station_id)

        print(f"Prep {station.id}...", end=" ")

        for label in SENSOR_LABEL:
            print(f"{label} sensor...", end=" ")
            df_sensor = rpd_build_sta.build_station(station=station,
                                                    sensor_label=label)
            list_df_sensors_per_station.append(df_sensor)

        print(f"Done.")

        # convert list of sensor dataframes into one station dataframe
        df_all_sensors_one_station = pd.concat(list_df_sensors_per_station, axis=1)
        list_df_stations.append(df_all_sensors_one_station)

    # convert list of station dataframes into one master dataframe to later parquet
    df_all_sensors_all_stations = df_all_sensors_one_station = pd.concat(list_df_stations, axis=0)
    df_all_sensors_all_stations.sort_values(by="station_id", ignore_index=True, inplace=True)  # sort by station id

    if build_df_parquet:
        # Need to flatten to save to parquet
        for label in SENSOR_LABEL:
            if label == 'barometer' or label == 'accelerometer' or label == 'gyroscope' or label == 'magnetometer':

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
        df_all_sensors_all_stations.to_parquet(os.path.join(OUTPUT_DIR, PD_PQT_FILE))
        print("\nExported pandas data frame to " + os.path.join(OUTPUT_DIR, PD_PQT_FILE))

        # Check that parquet file saves and opens correctly
        df_open = pd.read_parquet(os.path.join(OUTPUT_DIR, PD_PQT_FILE))
        print("Total stations:", len(df_open['station_id']))
        print("Available stations names:", df_open['station_id'])
        print("Total available columns:", df_open.columns)

    else:
        print("\nDid not export pandas data frame, must set build_df_parquet = True")
