"""
Skyfall TDR RPD
"""
# todo: address possible invalid values in building plots section
# Python libraries
import matplotlib.pyplot as plt
import datetime as dtime

# RedVox RedPandas and related RedVox modules
import redpandas.redpd_preprocess as rpd_prep
import redpandas.redpd_plot.wiggles as rpd_plot
import redpandas.redpd_geospatial as rpd_geo
from redpandas.redpd_scales import METERS_TO_KM
from quantum_inferno.plot_templates import plot_base as pbase, plot_templates as pnl

# Configuration files
from redpandas.redpd_config import DataLoadMethod
import examples.skyfall.lib.skyfall_dw as sf_dw
from examples.skyfall.skyfall_config_file import skyfall_config, \
    ref_latitude_deg, ref_longitude_deg, ref_altitude_m, ref_epoch_s


def main():
    """
    RedVox RedPandas time-domain representation of API900 data. Example: Skyfall.
    Last updated: November 2021
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
    accelerometer_data_highpass_label: str = "accelerometer_wf_highpass"
    accelerometer_epoch_s_label: str = "accelerometer_epoch_s"
    accelerometer_fs_label: str = "accelerometer_sample_rate_hz"

    # Gyroscope columns
    gyroscope_data_raw_label: str = "gyroscope_wf_raw"
    gyroscope_data_highpass_label: str = "gyroscope_wf_highpass"
    gyroscope_epoch_s_label: str = "gyroscope_epoch_s"
    gyroscope_fs_label: str = "gyroscope_sample_rate_hz"

    # Magnetometer columns
    magnetometer_data_raw_label: str = "magnetometer_wf_raw"
    magnetometer_data_highpass_label: str = "magnetometer_wf_highpass"
    magnetometer_epoch_s_label: str = "magnetometer_epoch_s"
    magnetometer_fs_label: str = "magnetometer_sample_rate_hz"

    # Health columns
    health_battery_charge_label: str = 'battery_charge_remaining_per'
    health_internal_temp_deg_c_label: str = 'internal_temp_deg_C'
    health_network_type_label: str = 'network_type'
    health_epoch_s_label: str = 'health_epoch_s'

    # Location columns
    location_latitude_label: str = 'location_latitude'
    location_longitude_label: str = 'location_longitude'
    location_altitude_label: str = "location_altitude"
    location_speed_label: str = 'location_speed'
    location_epoch_s_label: str = 'location_epoch_s'
    location_provider_label: str = 'location_provider'

    # Synchronization columns
    synchronization_epoch_label: str = 'synchronization_epoch_s'
    synchronization_latency_label: str = 'synchronization_latency_ms'
    synchronization_offset_label: str = 'synchronization_offset_ms'
    synchronization_best_offset_label: str = 'synchronization_best_offset_ms'
    synchronization_offset_delta_label: str = 'synchronization_offset_delta_ms'
    synchronization_number_exchanges_label: str = 'synchronization_number_exchanges'

    # Load data options
    # RECOMMENDED: tdr_load_method="datawindow" in config file
    df_skyfall_data = sf_dw.dw_main(skyfall_config.tdr_load_method)

    # Start of building plots
    print("\nInitiating time-domain representation of Skyfall:")
    for station in df_skyfall_data.index:
        station_id_str = df_skyfall_data[station_label][station]  # Get the station id

        if audio_data_label and audio_fs_label in df_skyfall_data.columns:
            print('mic_sample_rate_hz: ', df_skyfall_data[audio_fs_label][station])
            print('mic_epoch_s_0: ', df_skyfall_data[audio_epoch_s_label][station][0])

            # Frame to mic start and end and plot
            event_reference_time_epoch_s = df_skyfall_data[audio_epoch_s_label][station][0]
        else:
            print(f"Missing Audio for station {station_id_str}.")
            continue  # skip to next iteration if audio data missing

        if barometer_data_raw_label and barometer_data_highpass_label and barometer_fs_label in df_skyfall_data.columns:
            if skyfall_config.tdr_load_method == DataLoadMethod.PARQUET:
                # Reshape wf columns
                rpd_prep.df_column_unflatten(df=df_skyfall_data,
                                             col_wf_label=barometer_data_raw_label,
                                             col_ndim_label=barometer_data_raw_label + "_ndim")

                rpd_prep.df_column_unflatten(df=df_skyfall_data,
                                             col_wf_label=barometer_data_highpass_label,
                                             col_ndim_label=barometer_data_highpass_label + "_ndim")

            print('barometer_sample_rate_hz:', df_skyfall_data[barometer_fs_label][station])
            print('barometer_epoch_s_0:', df_skyfall_data[barometer_epoch_s_label][station][0])

            barometer_height_m = \
                rpd_geo.bounder_model_height_from_pressure(df_skyfall_data[barometer_data_raw_label][station][0])
            baro_height_from_bounder_km = barometer_height_m * METERS_TO_KM
        else:
            barometer_height_m = 0  # set as a precaution
            baro_height_from_bounder_km = 0  # set as a precaution

        # Repeat here
        if accelerometer_data_raw_label and accelerometer_fs_label and accelerometer_data_highpass_label \
                in df_skyfall_data.columns:
            if skyfall_config.tdr_load_method == DataLoadMethod.PARQUET:
                # Reshape wf columns
                rpd_prep.df_column_unflatten(df=df_skyfall_data,
                                             col_wf_label=accelerometer_data_raw_label,
                                             col_ndim_label=accelerometer_data_raw_label + "_ndim")
                rpd_prep.df_column_unflatten(df=df_skyfall_data,
                                             col_wf_label=accelerometer_data_highpass_label,
                                             col_ndim_label=accelerometer_data_highpass_label + "_ndim")

            print('accelerometer_sample_rate_hz:', df_skyfall_data[accelerometer_fs_label][station])
            print('accelerometer_epoch_s_0:',  df_skyfall_data[accelerometer_epoch_s_label][station][0],
                  df_skyfall_data[accelerometer_epoch_s_label][station][-1])

            # Plot 3c acceleration raw waveforms
            pnl_wfb = pbase.WaveformPlotBase(station_id=station_id_str,
                                             figure_title=skyfall_config.event_name + ": Accelerometer raw",
                                             figure_title_show=False, # for press
                                             start_time_epoch=event_reference_time_epoch_s,
                                             label_panel_show=True,  # for press
                                             labels_fontweight='bold')
            pnl_a = pbase.WaveformPanel(sig=df_skyfall_data[accelerometer_data_raw_label][station][0],
                                        time=df_skyfall_data[accelerometer_epoch_s_label][station],
                                        label="Acc X, m/$s^2$")
            pnl_b = pbase.WaveformPanel(sig=df_skyfall_data[accelerometer_data_raw_label][station][1],
                                        time=df_skyfall_data[accelerometer_epoch_s_label][station],
                                        label="Acc Y, m/$s^2$")
            pnl_c = pbase.WaveformPanel(sig=df_skyfall_data[accelerometer_data_raw_label][station][2],
                                        time=df_skyfall_data[accelerometer_epoch_s_label][station],
                                        label="Acc Z, m/$s^2$")
            fig_3c_acc_raw = pnl.plot_wf_3_vert(pnl_wfb, pnl_a, pnl_b, pnl_c)

        if accelerometer_data_raw_label and barometer_data_raw_label in df_skyfall_data.columns:
            # Plot aligned waveforms for sensor payload
            pnl_wfb.figure_title = skyfall_config.event_name + ": with Acc and Bar Highpass"
            pnl_a.sig = df_skyfall_data[barometer_data_highpass_label][station][0]
            pnl_a.time = df_skyfall_data[barometer_epoch_s_label][station]
            pnl_a.label = "Bar hp, kPa"
            pnl_b.sig = df_skyfall_data[accelerometer_data_highpass_label][station][2]
            pnl_b.time = df_skyfall_data[accelerometer_epoch_s_label][station]
            pnl_b.label = "Acc Z hp, m/$s^2$"
            pnl_c.sig = df_skyfall_data[audio_data_label][station]
            pnl_c.time = df_skyfall_data[audio_epoch_s_label][station]
            pnl_c.label = "Mic, Norm,"
            fig_3c_aab_hp = pnl.plot_wf_3_vert(pnl_wfb, pnl_a, pnl_b, pnl_c)

            pnl_wfb.figure_title = skyfall_config.event_name
            pnl_a.sig = baro_height_from_bounder_km
            pnl_a.label = "Bar Z Height, km"
            pnl_b.sig = df_skyfall_data[accelerometer_data_raw_label][station][2]
            pnl_b.label = "Acc Z, m/$s^2$"
            fig_3c_aab_raw = pnl.plot_wf_3_vert(pnl_wfb, pnl_a, pnl_b, pnl_c)

        if gyroscope_data_raw_label and gyroscope_fs_label and gyroscope_data_highpass_label \
                in df_skyfall_data.columns:

            if skyfall_config.tdr_load_method == DataLoadMethod.PARQUET:
                # Reshape wf columns
                rpd_prep.df_column_unflatten(df=df_skyfall_data,
                                             col_wf_label=gyroscope_data_raw_label,
                                             col_ndim_label=gyroscope_data_raw_label + "_ndim")

                rpd_prep.df_column_unflatten(df=df_skyfall_data,
                                             col_wf_label=gyroscope_data_highpass_label,
                                             col_ndim_label=gyroscope_data_highpass_label + "_ndim")

            print('gyroscope_sample_rate_hz:', df_skyfall_data[gyroscope_fs_label][station])
            print('gyroscope_epoch_s_0:', df_skyfall_data[gyroscope_epoch_s_label][station][0],
                  df_skyfall_data[gyroscope_epoch_s_label][station][-1])
            # Plot 3c raw gyroscope waveforms
            pnl_wfb.figure_title = skyfall_config.event_name + ": Gyroscope raw"
            pnl_a.sig = df_skyfall_data[gyroscope_data_raw_label][station][0]
            pnl_a.time = df_skyfall_data[gyroscope_epoch_s_label][station]
            pnl_a.label = "Gyr X, rad/s"
            pnl_b.sig = df_skyfall_data[gyroscope_data_raw_label][station][1]
            pnl_b.time = df_skyfall_data[gyroscope_epoch_s_label][station]
            pnl_b.label = "Gyr Y, rad/s"
            pnl_c.sig = df_skyfall_data[gyroscope_data_raw_label][station][2]
            pnl_c.time = df_skyfall_data[gyroscope_epoch_s_label][station]
            pnl_c.label = "Gyr Z, rad/s"
            fig_3c_gyr_raw = pnl.plot_wf_3_vert(pnl_wfb, pnl_a, pnl_b, pnl_c)

        if magnetometer_data_raw_label and magnetometer_fs_label and magnetometer_data_highpass_label \
                in df_skyfall_data.columns:
            if skyfall_config.tdr_load_method == DataLoadMethod.PARQUET:
                # Reshape wf columns
                rpd_prep.df_column_unflatten(df=df_skyfall_data,
                                             col_wf_label=magnetometer_data_raw_label,
                                             col_ndim_label=magnetometer_data_raw_label + "_ndim")
                rpd_prep.df_column_unflatten(df=df_skyfall_data,
                                             col_wf_label=magnetometer_data_highpass_label,
                                             col_ndim_label=magnetometer_data_highpass_label + "_ndim")

            print('magnetometer_sample_rate_hz:', df_skyfall_data[magnetometer_fs_label][station])
            print('magnetometer_epoch_s_0:', df_skyfall_data[magnetometer_epoch_s_label][station][0],
                  df_skyfall_data[magnetometer_epoch_s_label][station][-1])
            # Plot 3c magnetometer raw waveforms
            pnl_wfb.figure_title = skyfall_config.event_name + ": Magnetometer raw"
            pnl_a.sig = df_skyfall_data[magnetometer_data_raw_label][station][0]
            pnl_a.time = df_skyfall_data[magnetometer_epoch_s_label][station]
            pnl_a.label = "Mag X, $\mu$T"
            pnl_b.sig = df_skyfall_data[magnetometer_data_raw_label][station][1]
            pnl_b.time = df_skyfall_data[magnetometer_epoch_s_label][station]
            pnl_b.label = "Mag Y, $\mu$T"
            pnl_c.sig = df_skyfall_data[magnetometer_data_raw_label][station][2]
            pnl_c.time = df_skyfall_data[magnetometer_epoch_s_label][station]
            pnl_c.label = "Mag Z, $\mu$T"
            fig_3c_mag_raw = pnl.plot_wf_3_vert(pnl_wfb, pnl_a, pnl_b, pnl_c)

        if location_latitude_label and location_longitude_label and location_altitude_label and location_speed_label \
                in df_skyfall_data.columns:

            print("Bounder End EPOCH:", ref_epoch_s)
            print("Bounder End LAT LON ALT:", ref_latitude_deg, ref_longitude_deg, ref_altitude_m)

            # Compute ENU projections
            df_range_z_speed = \
                rpd_geo.compute_t_r_z_speed(unix_s=df_skyfall_data[location_epoch_s_label][station],
                                            lat_deg=df_skyfall_data[location_latitude_label][station],
                                            lon_deg=df_skyfall_data[location_longitude_label][station],
                                            alt_m=df_skyfall_data[location_altitude_label][station],
                                            ref_unix_s=ref_epoch_s,
                                            ref_lat_deg=ref_latitude_deg,
                                            ref_lon_deg=ref_longitude_deg,
                                            ref_alt_m=ref_altitude_m)

            # Plot location framework
            pnl_wfb.figure_title = skyfall_config.event_name + ": Location Framework"
            pnl_a.sig = df_skyfall_data[location_speed_label][station]
            pnl_a.time = df_skyfall_data[location_epoch_s_label][station]
            pnl_a.label = "Speed, m/s"
            pnl_b.sig = df_range_z_speed['Z_m']*METERS_TO_KM
            pnl_b.time = df_skyfall_data[location_epoch_s_label][station]
            pnl_b.label = "Altitude, km"
            pnl_c.sig = df_range_z_speed['Range_m']*METERS_TO_KM
            pnl_c.time = df_skyfall_data[location_epoch_s_label][station]
            pnl_c.label = "Range, km"
            fig_3c_loc_raw = pnl.plot_wf_3_vert(pnl_wfb, pnl_a, pnl_b, pnl_c)

            if location_epoch_s_label and location_altitude_label and barometer_epoch_s_label and \
                    barometer_data_raw_label in df_skyfall_data.columns:

                print(df_skyfall_data.columns)

                plt.figure()
                time_bar = df_skyfall_data[barometer_epoch_s_label][station] - skyfall_config.event_start_epoch_s
                # time_bounder = bounder_loc['Epoch_s'] - skyfall_config.event_start_epoch_s
                time_loc = df_skyfall_data[location_epoch_s_label][station] - skyfall_config.event_start_epoch_s

                ax1 = plt.subplot(211)
                plt.semilogy(time_bar, df_skyfall_data[barometer_data_raw_label][station][0], 'midnightblue',
                             label='Barometer kPa')
                # plt.semilogy(time_bounder, bounder_loc['Pres_kPa'], 'g', label='Bounder kPa')
                plt.ylabel('Pressure, kPa')
                plt.legend(loc='lower right')
                plt.xlim([0, 1800])
                plt.text(0.01, 0.9, "(b)", transform=ax1.transAxes,  fontweight='bold')
                ax1.set_xticklabels([])
                plt.grid(True)

                ax2 = plt.subplot(212)
                plt.plot(time_loc, df_skyfall_data[location_altitude_label][station] * METERS_TO_KM, 'r',
                         label='Location sensor')
                plt.plot(time_bar, barometer_height_m * METERS_TO_KM, 'midnightblue', label='Barometer Z')
                # plt.plot(time_bounder, bounder_loc['Alt_m'] * METERS_TO_KM, 'g', label='Bounder Z')
                plt.ylabel('Height, km')
                time_str = dtime.datetime.utcfromtimestamp(skyfall_config.event_start_epoch_s).strftime(
                    '%Y-%m-%d %H:%M:%S')
                plt.xlabel(f"Time (s) from UTC {time_str}")
                plt.legend(loc='upper right')
                plt.xlim([0, 1800])
                plt.text(0.01, 0.05, "(a)", transform=ax2.transAxes,  fontweight='bold')
                plt.grid(True)
                plt.tight_layout()

        if health_battery_charge_label and health_internal_temp_deg_c_label and health_network_type_label \
                and barometer_data_raw_label and location_provider_label in df_skyfall_data.columns:

            print(f"location_provider_epoch_s_0: {df_skyfall_data[location_provider_label][station][0]}",
                  f", location_provider_epoch_s_end: {df_skyfall_data[location_provider_label][station][-1]}")
            print(f"network_type_epoch_s_0: {df_skyfall_data[health_network_type_label][station][0]}",
                  f", network_type_epoch_s_end: {df_skyfall_data[health_network_type_label][station][-1]}")

            # Other interesting fields: Estimated Height ASL, Internal Temp, % Battery
            pnl_wfb.figure_title = skyfall_config.event_name + ": Station Status"
            pnl_a.sig = df_skyfall_data[health_battery_charge_label][station]
            pnl_a.time = df_skyfall_data[health_epoch_s_label][station]
            pnl_a.label = "Battery %"
            pnl_b.sig = df_skyfall_data[health_internal_temp_deg_c_label][station]
            pnl_b.time = df_skyfall_data[health_epoch_s_label][station]
            pnl_b.label = "Temp, $^oC$"
            pnl_c.sig = barometer_height_m
            pnl_c.time = df_skyfall_data[barometer_epoch_s_label][station]
            pnl_c.label = "Bar Z Height, m"
            fig_3c_sts_raw = pnl.plot_wf_3_vert(pnl_wfb, pnl_a, pnl_b, pnl_c)

        if synchronization_epoch_label and synchronization_latency_label and synchronization_offset_label \
                and synchronization_best_offset_label and synchronization_offset_delta_label and \
                synchronization_number_exchanges_label in df_skyfall_data.columns:

            # Plot synchronization framework
            pnl_wfb.figure_title = skyfall_config.event_name + ": Synchronization Framework"
            pnl_a.sig = df_skyfall_data[synchronization_offset_delta_label][station]
            pnl_a.time = df_skyfall_data[synchronization_epoch_label][station]
            pnl_a.label = "Offset delta, s"
            pnl_b.sig = df_skyfall_data[synchronization_offset_label][station]
            pnl_b.time = df_skyfall_data[synchronization_epoch_label][station]
            pnl_b.label = "Offset, s"
            pnl_c.sig = df_skyfall_data[synchronization_latency_label][station]
            pnl_c.time = df_skyfall_data[synchronization_epoch_label][station]
            pnl_c.label = "Latency, ms"
            fig_3c_syn_raw = pnl.plot_wf_3_vert(pnl_wfb, pnl_a, pnl_b, pnl_c)

            # Plot synchronization framework
            pnl_a.sig = df_skyfall_data[location_altitude_label][station] * METERS_TO_KM
            pnl_a.time = df_skyfall_data[location_epoch_s_label][station]
            pnl_a.label = "Height, km"
            fig_3c_lsy_raw = pnl.plot_wf_3_vert(pnl_wfb, pnl_a, pnl_b, pnl_c)

        # Plot sensor wiggles
        sensor_column_label_list = [audio_data_label, barometer_data_highpass_label,
                                    accelerometer_data_highpass_label, gyroscope_data_highpass_label,
                                    magnetometer_data_highpass_label]

        sensor_epoch_column_label_list = [audio_epoch_s_label, barometer_epoch_s_label,
                                          accelerometer_epoch_s_label, gyroscope_epoch_s_label,
                                          magnetometer_epoch_s_label]

        print("Wiggle pandas ...")
        print('Sensor labels: ', sensor_column_label_list)
        print('Sensor epochs: ', sensor_epoch_column_label_list)

        rpd_plot.plot_wiggles_pandas(df=df_skyfall_data,
                                     sig_wf_label=sensor_column_label_list,
                                     sig_timestamps_label=sensor_epoch_column_label_list,
                                     sig_id_label='station_id',
                                     station_id_str='1637610021',
                                     fig_title_show=True,
                                     fig_title='sensor waveforms',
                                     show_figure=True)

        plt.show()


if __name__ == "__main__":
    main()
