# Python libraries
import os
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
from examples.skyfall.skyfall_config import EVENT_NAME, INPUT_DIR, EPISODE_START_EPOCH_S, \
    EPISODE_END_EPOCH_S, STATIONS, PD_PQT_FILE, OUTPUT_DIR, DW_FILE, build_dw_pickle, build_df_parquet, \
    plot_mic_waveforms, print_datawindow_dq, SENSOR_LABEL


if __name__ == "__main__":
    """
    Beta workflow for API M pipeline
    Last updated: 1 June 2021
    """
    print('Let the sky fall')
    print("Initiating Conversion from RedVox DataWindow to RedVox RedPandas:")

    settings.set_parallelism_enabled(True)

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    if build_dw_pickle:
        # RECOMMENDED USE
        # Load signals, create a RedVox DataWindow structure, export to pickle.
        rpd_dw.build(api_input_directory=INPUT_DIR,
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
    else:
        print("Unpickling existing compressed RedVox DataWindow with JSON...")
    rdvx_data: DataWindow = DataWindow.from_json_file(base_dir=OUTPUT_DIR, file_name=DW_FILE)
    print(f"RedVox SDK version: {rdvx_data.sdk_version}")

    # Print out basic stats
    if print_datawindow_dq:
        print("\nDQ/DA LAYER: STATION")
        rpd_dq.station_metadata(rdvx_data)
        print("DQ/DA LAYER: MIC & SYNCH")
        rpd_dq.mic_sync(rdvx_data)
        print("DQ/DA LAYER: SENSOR TIMING")
        rpd_dq.station_channel_timing(rdvx_data)

    # Plot data window waveforms
    if plot_mic_waveforms:
        rpd_dw.plot_dw_mic(data_window=rdvx_data)
        rpd_dw.plot_dw_baro(data_window=rdvx_data)
        plt.show()

    # BEGIN RED PANDAS
    print("\nInitiating RedVox Redpandas:")
    df_all_sensors_all_stations = pd.DataFrame([rpd_build_sta.station_to_dict_from_dw(station=station,
                                                                                      sdk_version=rdvx_data.sdk_version,
                                                                                      sensor_labels=SENSOR_LABEL)
                                                for station in rdvx_data.stations])
    df_all_sensors_all_stations.sort_values(by="station_id", ignore_index=True, inplace=True)

    if build_df_parquet:
        # Need to flatten to save to parquet
        for label in SENSOR_LABEL:
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
        df_all_sensors_all_stations.to_parquet(os.path.join(OUTPUT_DIR, PD_PQT_FILE))
        print("\nExported Parquet RedPandas DataFrame to " + os.path.join(OUTPUT_DIR, PD_PQT_FILE))

        # Check that parquet file saves and opens correctly
        df_open = pd.read_parquet(os.path.join(OUTPUT_DIR, PD_PQT_FILE))
        print("Total stations in DataFrame:", len(df_open['station_id']))
        print("Available stations:", df_open['station_id'])
        print("Total columns in DataFrame:", len(df_open.columns))
        print("Available columns:", df_open.columns)

    else:
        print("\nDid not export pandas data frame, must set build_df_parquet = True")
