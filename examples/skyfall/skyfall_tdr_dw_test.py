import numpy as np
import matplotlib.pyplot as plt
import pickle
import os.path

# RedVox and Red Pandas modules
# from redvox.common.data_window import DataWindow
from redvox.common.data_window import DataWindowFast
import redvox.common.date_time_utils as dt
import redpandas.redpd_preprocess as rpd_prep
import redpandas.redpd_scales as rpd_scales
import redpandas.redpd_build_station as rpd_build_sta

from libquantum.plot_templates import plot_time_frequency_reps as pnl

# Verify points to correct config file.
# TODO: build check in case file does not exist!

# Configuration file
from examples.skyfall.skyfall_config import EVENT_NAME, INPUT_DIR, OUTPUT_DIR, EPISODE_START_EPOCH_S, \
    EPISODE_END_EPOCH_S, STATIONS, DW_FILE, use_datawindow, is_pickle_serialized


if __name__ == "__main__":
    """
    Red Pandas time-frequency representation of API900 data
    Last updated: 7 May 2021
    """

    print('Let the sky fall')

    # Option A: Open data with DataWindow
    if use_datawindow:
        rdvx_data = DataWindowFast(input_dir=INPUT_DIR,
                                   station_ids=STATIONS,
                                   start_datetime=dt.datetime_from_epoch_seconds_utc(EPISODE_START_EPOCH_S),
                                   end_datetime=dt.datetime_from_epoch_seconds_utc(EPISODE_END_EPOCH_S),
                                   apply_correction=True,
                                   structured_layout=True)
    # Option B: Load pickle
    else:
        if is_pickle_serialized:
            # Load DataWindow structure
            rdvx_data: DataWindowFast = DataWindowFast.from_json_file(base_dir=OUTPUT_DIR,
                                                                      file_name=DW_FILE)

        else:
            with open(os.path.join(INPUT_DIR, DW_FILE), 'rb') as file:
                # Load DataWindow structure
                rdvx_data = pickle.load(file)

    # Start of building plots
    for station in rdvx_data.stations:
        station_id_str = station.id  # Get the station id

        if station.has_audio_data():
            df_audio = rpd_build_sta.build_station(station=station,
                                                   sensor_label='audio')

            print('\nmic_sample_rate_hz: ', df_audio['audio_sample_rate_corrected_hz'][0])
            print('mic_epoch_s_0: ', df_audio['audio_epoch_s'][0][0])

            # Frame to mic start and end and plot
            event_reference_time_epoch_s = df_audio['audio_epoch_s'][0][0]

        if station.has_barometer_sensor():
            df_baro = rpd_build_sta.build_station(station=station, sensor_label='barometer')

            print('barometer_sample_rate_hz:', df_baro['barometer_sample_rate_hz'][0])
            print('barometer_epoch_s_0:', df_baro['barometer_epoch_s'][0][0])

            barometer_height_m = rpd_prep.model_height_from_pressure_skyfall(df_baro['barometer_wf_raw'][0][0])
            baro_height_from_bounder_km = barometer_height_m/1E3

        # Repeat here
        if station.has_accelerometer_sensor():
            df_acc = rpd_build_sta.build_station(station=station, sensor_label='accelerometer')

            print('accelerometer_sample_rate_hz:', df_acc['accelerometer_sample_rate_hz'][0])
            print('accelerometer_epoch_s_0:',  df_acc['accelerometer_epoch_s'][0][0], df_acc['accelerometer_epoch_s'][0][-1])
            # Plot aligned waveforms
            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=df_acc['accelerometer_wf_raw'][0][2],
                                   wf_panel_2_time=df_acc['accelerometer_epoch_s'][0],
                                   wf_panel_1_sig=df_acc['accelerometer_wf_raw'][0][1],
                                   wf_panel_1_time=df_acc['accelerometer_epoch_s'][0],
                                   wf_panel_0_sig=df_acc['accelerometer_wf_raw'][0][0],
                                   wf_panel_0_time=df_acc['accelerometer_epoch_s'][0],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Z, m/$s^2$",
                                   wf_panel_1_units="Y, m/$s^2$",
                                   wf_panel_0_units="X, m/$s^2$",
                                   figure_title=EVENT_NAME + ": Accelerometer raw")

        if station.has_accelerometer_sensor() and station.has_barometer_sensor():
            # Plot aligned waveforms

            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=df_audio['audio_wf'][0],
                                   wf_panel_2_time=df_audio['audio_epoch_s'][0],
                                   wf_panel_1_sig=df_acc['accelerometer_wf_raw'][0][2],
                                   wf_panel_1_time=df_acc['accelerometer_epoch_s'][0],
                                   wf_panel_0_sig=df_baro['barometer_wf_raw'][0][0],
                                   wf_panel_0_time=df_baro['barometer_epoch_s'][0],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Mic, Bits",
                                   wf_panel_1_units="Acc Z, m/$s^2$",
                                   wf_panel_0_units="Bar, kPa",
                                   figure_title=EVENT_NAME)

            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=df_audio['audio_wf'][0],
                                   wf_panel_2_time=df_audio['audio_epoch_s'][0],
                                   wf_panel_1_sig=df_acc['accelerometer_wf_raw'][0][2],
                                   wf_panel_1_time=df_acc['accelerometer_epoch_s'][0],
                                   wf_panel_0_sig=baro_height_from_bounder_km,
                                   wf_panel_0_time=df_baro['barometer_epoch_s'][0],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Mic, Norm",
                                   wf_panel_1_units="Acc Z, m/$s^2$",
                                   wf_panel_0_units="Height (/Bar), km",
                                   figure_title=EVENT_NAME)

        if station.has_gyroscope_sensor():
            df_gyro = rpd_build_sta.build_station(station=station, sensor_label='gyroscope')
            print('gyroscope_sample_rate_hz:', df_gyro['gyroscope_sample_rate_hz'][0])
            print('gyroscope_epoch_s_0:', df_gyro['gyroscope_epoch_s'][0][0], df_gyro['gyroscope_epoch_s'][0][-1])
            # Plot aligned waveforms
            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=df_gyro['gyroscope_wf_raw'][0][2],
                                   wf_panel_2_time=df_gyro['gyroscope_epoch_s'][0],
                                   wf_panel_1_sig=df_gyro['gyroscope_wf_raw'][0][1],
                                   wf_panel_1_time=df_gyro['gyroscope_epoch_s'][0],
                                   wf_panel_0_sig=df_gyro['gyroscope_wf_raw'][0][0],
                                   wf_panel_0_time=df_gyro['gyroscope_epoch_s'][0],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Z, rad/s",
                                   wf_panel_1_units="Y, rad/s",
                                   wf_panel_0_units="X, rad/s",
                                   figure_title=EVENT_NAME + ": Gyroscope raw")

        if station.has_magnetometer_sensor():
            df_mag = rpd_build_sta.build_station(station=station, sensor_label='magnetometer')
            print('magnetometer_sample_rate_hz:', df_mag['magnetometer_sample_rate_hz'][0])
            print('magnetometer_epoch_s_0:', df_mag['magnetometer_epoch_s'][0][0], df_mag['magnetometer_epoch_s'][0][-1])
            # Plot aligned waveforms
            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=df_mag['magnetometer_wf_raw'][0][2],
                                   wf_panel_2_time=df_mag['magnetometer_epoch_s'][0],
                                   wf_panel_1_sig=df_mag['magnetometer_wf_raw'][0][1],
                                   wf_panel_1_time=df_mag['magnetometer_epoch_s'][0],
                                   wf_panel_0_sig=df_mag['magnetometer_wf_raw'][0][0],
                                   wf_panel_0_time=df_mag['magnetometer_epoch_s'][0],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Z, $\mu$T",
                                   wf_panel_1_units="Y, $\mu$T",
                                   wf_panel_0_units="X, $\mu$T",
                                   figure_title=EVENT_NAME + ": Magnetometer")

        if station.has_location_sensor():

            df_loc = rpd_build_sta.build_station(station=station, sensor_label='location')

            # Range vs reference lat lon
            location_latitude_reference = 35.83728684
            location_longitude_reference = -115.57228988
            print("LAT LON at landing:", location_latitude_reference, location_longitude_reference)
            range_lat = (df_loc['location_latitude'][0] - location_latitude_reference) * rpd_scales.DEGREES_TO_M
            range_lon = (df_loc['location_longitude'][0] - location_longitude_reference) * rpd_scales.DEGREES_TO_M
            range_m = np.sqrt(np.array(range_lat**2 + range_lon**2).astype(np.float64))

            # TODO, TYLER: ADDRESS AND PROVIDE RECOMMENDATIONS
            location_provider = station.location_sensor().get_data_channel("location_provider")
            # location_bearing = station.location_sensor().get_channel("bearing")
            # print('Why does bearing only have nans?', location_bearing)
            # print('Why does vertical accuracy only have nans?', location_vertical_accuracy)

            pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                                   wf_panel_2_sig=range_m,
                                   wf_panel_2_time=df_loc['location_epoch_s'][0],
                                   wf_panel_1_sig=df_loc['location_altitude'][0],
                                   wf_panel_1_time=df_loc['location_epoch_s'][0],
                                   wf_panel_0_sig=df_loc['location_speed'][0],
                                   wf_panel_0_time=df_loc['location_epoch_s'][0],
                                   start_time_epoch=event_reference_time_epoch_s,
                                   wf_panel_2_units="Range, m",
                                   wf_panel_1_units="Altitude, m",
                                   wf_panel_0_units="Speed, m/s",
                                   figure_title=EVENT_NAME + ": Location Framework")

        if station.has_health_sensor():
            df_soh = rpd_build_sta.build_station(station=station, sensor_label='soh')

            # TODO, ANTHONY: How to handle these
            # print("\nLocation provider:", location_provider)
            print(f"network_type_epoch_s_0:", df_soh['network_type'][0][0],
                  f", network_type_epoch_s_end:", df_soh['network_type'][0][-1])


        # TODO: Convert synchronization to a sensor object. In fact, convert everything to a sensor object.
        # TODO: Get number of synch exchanges per packet
        synchronization = station.timesync_analysis
        synchronization_epoch_s = synchronization.get_start_times() * rpd_scales.MICROS_TO_S
        synchronization_latency_ms = synchronization.get_latencies() * rpd_scales.MICROS_TO_MILLIS
        synchronization_offset_ms = synchronization.get_offsets() * rpd_scales.MICROS_TO_MILLIS
        synchronization_best_offset_ms = synchronization.get_best_offset() * rpd_scales.MICROS_TO_MILLIS
        synchronization_offset_delta_ms = synchronization_offset_ms - synchronization_best_offset_ms
        # TODO, TYLER: Get number of synch exchanges per packet as a time series
        synchronization_number_exchanges = synchronization.timesync_data[0].num_tri_messages()

        pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                               wf_panel_2_sig=synchronization_latency_ms,
                               wf_panel_2_time=synchronization_epoch_s,
                               wf_panel_1_sig=synchronization_offset_ms,
                               wf_panel_1_time=synchronization_epoch_s,
                               wf_panel_0_sig=synchronization_offset_delta_ms,
                               wf_panel_0_time=synchronization_epoch_s,
                               start_time_epoch=event_reference_time_epoch_s,
                               wf_panel_2_units="Latency, ms",
                               wf_panel_1_units="Offset, s",
                               wf_panel_0_units="Offset delta, s",
                               figure_title=EVENT_NAME + ": Synchronization Framework")
        # TODO: Address nan padding on location framework

        # Other interesting fields: Estimated Height ASL, Internal Temp, % Battery
        pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                               wf_panel_2_sig=barometer_height_m,
                               wf_panel_2_time=df_baro['barometer_epoch_s'][0],
                               wf_panel_1_sig=df_soh['internal_temp_deg_C'][0],
                               wf_panel_1_time=df_soh['health_epoch_s'][0],
                               wf_panel_0_sig=df_soh['battery_charge_remaining_per'][0],
                               wf_panel_0_time=df_soh['health_epoch_s'][0],
                               start_time_epoch=event_reference_time_epoch_s,
                               wf_panel_2_units="Baro Z, m",
                               wf_panel_1_units="Temp C",
                               wf_panel_0_units="Battery %",
                               figure_title=EVENT_NAME + ": Station Status")

        # Finally, barometric and altitude estimates
        pnl.plot_wf_wf_wf_vert(redvox_id=station_id_str,
                               wf_panel_2_sig=barometer_height_m,
                               wf_panel_2_time=df_baro['barometer_epoch_s'][0],
                               wf_panel_1_sig=df_loc['location_altitude'][0],
                               wf_panel_1_time=df_loc['location_epoch_s'][0],
                               wf_panel_0_sig=df_soh['internal_temp_deg_C'][0],
                               wf_panel_0_time=df_soh['health_epoch_s'][0],
                               start_time_epoch=event_reference_time_epoch_s,
                               wf_panel_2_units="Baro Height, m",
                               wf_panel_1_units="Loc Height, m",
                               wf_panel_0_units="Temp C",
                               figure_title=EVENT_NAME + ": Height and Temperature")
        plt.show()
        # FOR API M: All other SOH fields.

