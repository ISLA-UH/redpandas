# Python libraries
import os.path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# RedVox and Red Pandas modules
import redpandas.redpd_preprocess as rpd_prep
import redpandas.redpd_scales as rpd_scales
from libquantum.plot_templates import plot_time_frequency_reps as pnl

# Configuration file
from examples.skyfall.skyfall_config import EVENT_NAME, OUTPUT_DIR, PD_PQT_FILE

# Verify points to correct config file.
# TODO MAG: build check in case file does not exist!
# TODO MC: highpass three panels
# TODO MC: plot hp in same panel, sqrt(add squares) for power in another panel, top panel TBD
# TODO MC: wiggles for sensors_raw and sensors_highpass

# TODO: decide where this fcn lives
def df_column_unflatten(df: pd.DataFrame,
                        col_wf_label: str,
                        col_ndim_label: str):
    """
    Restores original shape of elements in column

    :param df: pandas DataFrame
    :param col_wf_label: column label for data that needs reshaping, usually waveform arrays.
    :param col_ndim_label: column label with dimensions for reshaping. Elements in column need to be a numpy array.
    :return: original df, replaces column values with reshaped ones
    """

    col_values = df[col_wf_label].to_numpy()
    for index_array in df.index:
        col_values[index_array].shape = (df[col_ndim_label][index_array][0],
                                         df[col_ndim_label][index_array][1])


if __name__ == "__main__":
    """
    Red Pandas time-frequency representation of API900 data
    Last updated: 7 May 2021
    """

    print('Let the sky fall')

    # Label columns in dataframe
    station_label: str = "station_id"

    # Audio columns
    audio_data_label: str = "audio_wf"
    audio_epoch_s_label: str = "audio_epoch_s"
    audio_fs_label: str = "audio_sample_rate_nominal_hz"

    # Barometer columns
    barometer_data_raw_label: str = "barometer_wf_raw"
    barometer_data_highpass_label: str = "barometer_wf_highpass"
    barometer_epoch_s_label: str = "barometer_epoch_s"
    barometer_fs_label: str = "barometer_sample_rate_hz"

    # Accelerometer columns
    accelerometer_data_raw_label: str = "accelerometer_wf_raw"
    accelerometer_epoch_s_label: str = "accelerometer_epoch_s"
    accelerometer_fs_label: str = "accelerometer_sample_rate_hz"

    # Gyroscope columns
    gyroscope_data_raw_label: str = "gyroscope_wf_raw"
    gyroscope_epoch_s_label: str = "gyroscope_epoch_s"
    gyroscope_fs_label: str = "gyroscope_sample_rate_hz"

    # Magnetometer columns
    magnetometer_data_raw_label: str = "magnetometer_wf_raw"
    magnetometer_epoch_s_label: str = "magnetometer_epoch_s"
    magnetometer_fs_label: str = "magnetometer_sample_rate_hz"

    # Health columns
    health_battery_charge_label: str = 'battery_charge_remaining_per'
    health_internal_temp_deg_C_label: str = 'internal_temp_deg_C'
    health_network_type_label: str = 'network_type'
    health_epoch_s_label: str = 'health_epoch_s'

    # Location columns
    location_latitude_label: str = 'location_latitude'
    location_longitude_label: str = 'location_longitude'
    location_altitude_label: str = "location_altitude"
    location_speed_label: str = 'location_speed'
    location_epoch_s_label: str = 'location_epoch_s'

    # Open dataframe from parquet file
    df_skyfall_data = pd.read_parquet(os.path.join(OUTPUT_DIR, PD_PQT_FILE))

    # Start of building plots
    for station in df_skyfall_data.index:
        station_id_str = df_skyfall_data[station_label][station]  # Get the station id

        if audio_data_label and audio_fs_label in df_skyfall_data.columns:

            print('\nmic_sample_rate_hz: ', df_skyfall_data[audio_fs_label][station])
            print('mic_epoch_s_0: ', df_skyfall_data[audio_epoch_s_label][station][0])

            # Frame to mic start and end and plot
            event_reference_time_epoch_s = df_skyfall_data[audio_epoch_s_label][station][0]

        if barometer_data_raw_label and barometer_data_highpass_label and barometer_fs_label in df_skyfall_data.columns:
            # Reshape wf columns
            df_column_unflatten(df=df_skyfall_data,
                                col_wf_label=barometer_data_raw_label,
                                col_ndim_label=barometer_data_raw_label + "_ndim")

            df_column_unflatten(df=df_skyfall_data,
                                col_wf_label=barometer_data_highpass_label,
                                col_ndim_label=barometer_data_highpass_label + "_ndim")

            print('barometer_sample_rate_hz:', df_skyfall_data[barometer_fs_label][station])
            print('barometer_epoch_s_0:', df_skyfall_data[barometer_epoch_s_label][station][0])

            barometer_height_m = \
                rpd_prep.model_height_from_pressure_skyfall(df_skyfall_data[barometer_data_raw_label][station][0])
            baro_height_from_bounder_km = barometer_height_m/1E3

        # Repeat here
        if accelerometer_data_raw_label and accelerometer_fs_label in df_skyfall_data.columns:
            # Reshape wf columns
            df_column_unflatten(df=df_skyfall_data,
                                col_wf_label=accelerometer_data_raw_label,
                                col_ndim_label=accelerometer_data_raw_label + "_ndim")

            print('accelerometer_sample_rate_hz:', df_skyfall_data[accelerometer_fs_label][station])
            print('accelerometer_epoch_s_0:',  df_skyfall_data[accelerometer_epoch_s_label][station][0],
                  df_skyfall_data[accelerometer_epoch_s_label][station][-1])
            # Plot aligned waveforms
            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=df_skyfall_data[accelerometer_data_raw_label][station][2],
                                   wf_panel_2_time=df_skyfall_data[accelerometer_epoch_s_label][station],
                                   wf_panel_1_sig=df_skyfall_data[accelerometer_data_raw_label][station][1],
                                   wf_panel_1_time=df_skyfall_data[accelerometer_epoch_s_label][station],
                                   wf_panel_0_sig=df_skyfall_data[accelerometer_data_raw_label][station][0],
                                   wf_panel_0_time=df_skyfall_data[accelerometer_epoch_s_label][station],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Z, m/$s^2$",
                                   wf_panel_1_units="Y, m/$s^2$",
                                   wf_panel_0_units="X, m/$s^2$",
                                   figure_title=EVENT_NAME + ": Accelerometer raw")

        if accelerometer_data_raw_label and barometer_data_raw_label in df_skyfall_data.columns:
            # Plot aligned waveforms
            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=df_skyfall_data[audio_data_label][station],
                                   wf_panel_2_time=df_skyfall_data[audio_epoch_s_label][station],
                                   wf_panel_1_sig=df_skyfall_data[accelerometer_data_raw_label][station][2],
                                   wf_panel_1_time=df_skyfall_data[accelerometer_epoch_s_label][station],
                                   wf_panel_0_sig=df_skyfall_data[barometer_data_highpass_label][station][0],
                                   wf_panel_0_time=df_skyfall_data[barometer_epoch_s_label][station],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Mic, Bits",
                                   wf_panel_1_units="Acc Z, m/$s^2$",
                                   wf_panel_0_units="Bar, kPa",
                                   figure_title=EVENT_NAME)

            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=df_skyfall_data[audio_data_label][station],
                                   wf_panel_2_time=df_skyfall_data[audio_epoch_s_label][station],
                                   wf_panel_1_sig=df_skyfall_data[accelerometer_data_raw_label][station][2],
                                   wf_panel_1_time=df_skyfall_data[accelerometer_epoch_s_label][station],
                                   wf_panel_0_sig=baro_height_from_bounder_km,
                                   wf_panel_0_time=df_skyfall_data[barometer_epoch_s_label][station],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Mic, Norm",
                                   wf_panel_1_units="Acc Z, m/$s^2$",
                                   wf_panel_0_units="Height (/Bar), km",
                                   figure_title=EVENT_NAME)

        if gyroscope_data_raw_label and gyroscope_epoch_s_label and gyroscope_fs_label in df_skyfall_data.columns:
            # Reshape wf columns
            df_column_unflatten(df=df_skyfall_data,
                                col_wf_label=gyroscope_data_raw_label,
                                col_ndim_label=gyroscope_data_raw_label + "_ndim")

            print('gyroscope_sample_rate_hz:', df_skyfall_data[gyroscope_fs_label][station])
            print('gyroscope_epoch_s_0:', df_skyfall_data[gyroscope_epoch_s_label][station][0],
                  df_skyfall_data[gyroscope_epoch_s_label][station][-1])
            # Plot aligned waveforms
            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=df_skyfall_data[gyroscope_data_raw_label][station][2],
                                   wf_panel_2_time=df_skyfall_data[gyroscope_epoch_s_label][station],
                                   wf_panel_1_sig=df_skyfall_data[gyroscope_data_raw_label][station][1],
                                   wf_panel_1_time=df_skyfall_data[gyroscope_epoch_s_label][station],
                                   wf_panel_0_sig=df_skyfall_data[gyroscope_data_raw_label][station][0],
                                   wf_panel_0_time=df_skyfall_data[gyroscope_epoch_s_label][station],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Z, rad/s",
                                   wf_panel_1_units="Y, rad/s",
                                   wf_panel_0_units="X, rad/s",
                                   figure_title=EVENT_NAME + ": Gyroscope raw")

        if magnetometer_data_raw_label and magnetometer_fs_label in df_skyfall_data.columns:
            # Reshape wf columns
            df_column_unflatten(df=df_skyfall_data,
                                col_wf_label=magnetometer_data_raw_label,
                                col_ndim_label=magnetometer_data_raw_label + "_ndim")

            print('magnetometer_sample_rate_hz:', df_skyfall_data[magnetometer_fs_label][station])
            print('magnetometer_epoch_s_0:', df_skyfall_data[magnetometer_epoch_s_label][station][0],
                  df_skyfall_data[magnetometer_epoch_s_label][station][-1])
            # Plot aligned waveforms
            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=df_skyfall_data[magnetometer_data_raw_label][station][2],
                                   wf_panel_2_time=df_skyfall_data[magnetometer_epoch_s_label][station],
                                   wf_panel_1_sig=df_skyfall_data[magnetometer_data_raw_label][station][1],
                                   wf_panel_1_time=df_skyfall_data[magnetometer_epoch_s_label][station],
                                   wf_panel_0_sig=df_skyfall_data[magnetometer_data_raw_label][station][0],
                                   wf_panel_0_time=df_skyfall_data[magnetometer_epoch_s_label][station],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Z, $\mu$T",
                                   wf_panel_1_units="Y, $\mu$T",
                                   wf_panel_0_units="X, $\mu$T",
                                   figure_title=EVENT_NAME + ": Magnetometer")
            
        if location_latitude_label and location_longitude_label and location_altitude_label and location_speed_label\
                in df_skyfall_data.columns:
            # Range vs reference lat lon
            location_latitude_reference = 35.83728684
            location_longitude_reference = -115.57228988
            print("LAT LON at landing:", location_latitude_reference, location_longitude_reference)
            range_lat = (df_skyfall_data[location_latitude_label][station] - location_latitude_reference) \
                        * rpd_scales.DEGREES_TO_M
            range_lon = (df_skyfall_data[location_longitude_label][station] - location_longitude_reference) \
                        * rpd_scales.DEGREES_TO_M
            range_m = np.sqrt(np.array(range_lat**2 + range_lon**2).astype(np.float64))

            # TODO, TYLER: ADDRESS AND PROVIDE RECOMMENDATIONS
            # location_provider = station.location_sensor().get_data_channel("location_provider")
            # location_bearing = station.location_sensor().get_channel("bearing")
            # print('Why does bearing only have nans?', location_bearing)
            # print('Why does vertical accuracy only have nans?', location_vertical_accuracy)

            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=range_m,
                                   wf_panel_2_time=df_skyfall_data[location_epoch_s_label][station],
                                   wf_panel_1_sig=df_skyfall_data[location_altitude_label][station],
                                   wf_panel_1_time=df_skyfall_data[location_epoch_s_label][station],
                                   wf_panel_0_sig=df_skyfall_data[location_speed_label][station],
                                   wf_panel_0_time=df_skyfall_data[location_epoch_s_label][station],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Range, m",
                                   wf_panel_1_units="Altitude, m",
                                   wf_panel_0_units="Speed, m/s",
                                   figure_title=EVENT_NAME + ": Location Framework")

        if health_battery_charge_label and health_internal_temp_deg_C_label and health_network_type_label \
                and barometer_data_raw_label in df_skyfall_data.columns:

            # TODO, ANTHONY: How to handle these
            # print("\nLocation provider:", location_provider)
            print(f"network_type_epoch_s_0:", df_skyfall_data[health_network_type_label][station][0],
                  f", network_type_epoch_s_end:", df_skyfall_data[health_network_type_label][station][-1])

            # Other interesting fields: Estimated Height ASL, Internal Temp, % Battery
            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=barometer_height_m,
                                   wf_panel_2_time=df_skyfall_data[barometer_epoch_s_label][station],
                                   wf_panel_1_sig=df_skyfall_data[health_internal_temp_deg_C_label][station],
                                   wf_panel_1_time=df_skyfall_data[health_epoch_s_label][station],
                                   wf_panel_0_sig=df_skyfall_data[health_battery_charge_label][station],
                                   wf_panel_0_time=df_skyfall_data[health_epoch_s_label][station],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Baro Z, m",
                                   wf_panel_1_units="Temp C",
                                   wf_panel_0_units="Battery %",
                                   figure_title=EVENT_NAME + ": Station Status")

        # TODO: Convert synchronization to a sensor object. In fact, convert everything to a sensor object.
        # TODO: Get number of synch exchanges per packet
        # synchronization = station.timesync_analysis
        # synchronization_epoch_s = synchronization.get_start_times() * rpd_scales.MICROS_TO_S
        # synchronization_latency_ms = synchronization.get_latencies() * rpd_scales.MICROS_TO_MILLIS
        # synchronization_offset_ms = synchronization.get_offsets() * rpd_scales.MICROS_TO_MILLIS
        # synchronization_best_offset_ms = synchronization.get_best_offset() * rpd_scales.MICROS_TO_MILLIS
        # synchronization_offset_delta_ms = synchronization_offset_ms - synchronization_best_offset_ms
        # # TODO, TYLER: Get number of synch exchanges per packet as a time series
        # synchronization_number_exchanges = synchronization.timesync_data[0].num_tri_messages()
        #
        # pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
        #                        wf_panel_2_sig=synchronization_latency_ms,
        #                        wf_panel_2_time=synchronization_epoch_s,
        #                        wf_panel_1_sig=synchronization_offset_ms,
        #                        wf_panel_1_time=synchronization_epoch_s,
        #                        wf_panel_0_sig=synchronization_offset_delta_ms,
        #                        wf_panel_0_time=synchronization_epoch_s,
        #                        start_time_epoch=event_reference_time_epoch_s,
        #                        wf_panel_2_units="Latency, ms",
        #                        wf_panel_1_units="Offset, s",
        #                        wf_panel_0_units="Offset delta, s",
        #                        figure_title=EVENT_NAME + ": Synchronization Framework")
        # TODO: Address nan padding on location framework

        if health_internal_temp_deg_C_label and barometer_data_raw_label and location_altitude_label \
                in df_skyfall_data.columns:

            # Finally, barometric and altitude estimates
            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=barometer_height_m,
                                   wf_panel_2_time=df_skyfall_data[barometer_epoch_s_label][station],
                                   wf_panel_1_sig=df_skyfall_data[location_altitude_label][station],
                                   wf_panel_1_time=df_skyfall_data[location_epoch_s_label][station],
                                   wf_panel_0_sig=df_skyfall_data[health_internal_temp_deg_C_label][station],
                                   wf_panel_0_time=df_skyfall_data[health_epoch_s_label][station],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Baro Height, m",
                                   wf_panel_1_units="Loc Height, m",
                                   wf_panel_0_units="Temp C",
                                   figure_title=EVENT_NAME + ": Height and Temperature")
        plt.show()
        # FOR API M: All other SOH fields.

