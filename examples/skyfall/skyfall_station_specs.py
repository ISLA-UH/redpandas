# Python libraries
import os

import csv

# RedVox and Red Pandas modules
from redvox.common.data_window import DataWindowFast

import redpandas.redpd_datawin as rpd_dw
import redpandas.redpd_dq as rpd_dq
import redpandas.redpd_build_station as rpd_build_sta
from redvox.api1000.wrapped_redvox_packet.station_information import OsType
import redvox.common.date_time_utils as dt
from redvox.common.station import Station

# Configuration file
from examples.skyfall.skyfall_config import EVENT_NAME, INPUT_DIR, EPISODE_START_EPOCH_S, \
    EPISODE_END_EPOCH_S, STATIONS, PD_PQT_FILE, OUTPUT_DIR, DW_FILE, build_dw_pickle, build_df_parquet, \
    plot_mic_waveforms, print_datawindow_dq, SENSOR_LABEL


def station_specs_to_csv(data_window, export_file):
    station: Station
    with open(export_file, 'w', newline='') as csvfile:
        # writer = csv.writer(csvfile, delimiter=' ',
        #                         quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer = csv.writer(csvfile, delimiter=',')

        for station in data_window.stations:
            # Station Make, Model, etc
            writer.writerow(["Description", "Field", "Value"])
            writer.writerow(["Station ID", "station.id", station.id])
            writer.writerow(["Make", "station.metadata.make", station.metadata.make])
            writer.writerow(["Model", "station.metadata.model", station.metadata.model])
            writer.writerow(["OS", "station.metadata.os", OsType(station.metadata.os).name])
            writer.writerow(["OS Version", "station.metadata.os_version", station.metadata.os_version])
            writer.writerow(["App Version", "station.metadata.app_version", station.metadata.app_version])
            writer.writerow(["SDK Version", "station.metadata.os_version", data_window.sdk_version])
            writer.writerow([])

            writer.writerow(["Description", "Field", "Value"])
            writer.writerow(["Station ID", "station.id", station.id])

            # if station.start_timestamp > 0:
            #     writer.writerow("STATION SPECS FOR ID: "
            #           f"{station.id}\n"
            #           f"App start time: "
            #           f"{dt.datetime_from_epoch_microseconds_utc(station.start_timestamp)}\n"
            #           f"Station first time stamp: "
            #           f"{dt.datetime_from_epoch_microseconds_utc(station.first_data_timestamp)}\n"
            #           f"Station last time stamp: "
            #           f"{dt.datetime_from_epoch_microseconds_utc(station.last_data_timestamp)}\n")
            # else:
            #     writer.writerow(f"STATION SPECS FOR ID: "
            #           f"{station.id}\n"
            #           f"App start time not available\n"
            #           f"Station first time stamp: "
            #           f"{dt.datetime_from_epoch_microseconds_utc(station.first_data_timestamp)}\n"
            #           f"Station last time stamp: "
            #           f"{dt.datetime_from_epoch_microseconds_utc(station.last_data_timestamp)}\n")
        #
        # print(f"Station Metadata:\n"
        #       f"Make: "
        #       f"{station.metadata.make}\n"
        #       f"Model: "
        #       f"{station.metadata.model}\n"
        #       f"OS: "
        #       f"{OsType(station.metadata.os).name}\n"
        #       f"OS version: "
        #       f"{station.metadata.os_version}\n"
        #       f"App Version: "
        #       f"{station.metadata.app_version}\n")
        #
        # if station.has_audio_data():
        #     print(f"\nAudio Sensor:\n"
        #           f"Model: "
        #           f"{station.audio_sensor().name}\n"
        #           f"Sample rate, Hz: "
        #           f"{station.audio_sensor().sample_rate_hz}\n"
        #           f"Sample interval, seconds: "
        #           f"{station.audio_sensor().sample_interval_s}\n"
        #           f"Sample interval standard dev, seconds: "
        #           f"{station.audio_sensor().sample_interval_std_s}\n")
        #
        # if station.has_barometer_data():
        #     print(f"Barometer Sensor:\n"
        #           f"Model: "
        #           f"{station.barometer_sensor().name}\n"
        #           f"Sample rate, Hz: "
        #           f"{station.barometer_sensor().sample_rate_hz}\n"
        #           f"Sample interval, seconds: "
        #           f"{station.barometer_sensor().sample_interval_s}\n"
        #           f"Sample interval standard dev, seconds: "
        #           f"{station.barometer_sensor().sample_interval_std_s}\n")
        # if station.has_accelerometer_data():
        #     print(f"Accelerometer Sensor:\n"
        #           f"Model: "
        #           f"{station.accelerometer_sensor().name}\n"
        #           f"Sample rate, Hz: "
        #           f"{station.accelerometer_sensor().sample_rate_hz}\n"
        #           f"Sample interval, seconds: "
        #           f"{station.accelerometer_sensor().sample_interval_s}\n"
        #           f"Sample interval standard dev, seconds: "
        #           f"{station.accelerometer_sensor().sample_interval_std_s}\n")
        # if station.has_magnetometer_data():
        #     print(f"Magnetometer Sensor:\n"
        #           f"Model: "
        #           f"{station.magnetometer_sensor().name}\n"
        #           f"Sample rate, Hz: "
        #           f"{station.magnetometer_sensor().sample_rate_hz}\n"
        #           f"Sample interval, seconds: "
        #           f"{station.magnetometer_sensor().sample_interval_s}\n"
        #           f"Sample interval standard dev, seconds: "
        #           f"{station.magnetometer_sensor().sample_interval_std_s}\n")
        # if station.has_gyroscope_data():
        #     print(f"Gyroscope Sensor:\n"
        #           f"Model: "
        #           f"{station.gyroscope_sensor().name}\n"
        #           f"Sample rate, Hz: "
        #           f"{station.gyroscope_sensor().sample_rate_hz}\n"
        #           f"Sample interval, seconds: "
        #           f"{station.gyroscope_sensor().sample_interval_s}\n"
        #           f"Sample interval standard dev, seconds: "
        #           f"{station.gyroscope_sensor().sample_interval_std_s}\n")
        # if station.has_location_sensor():
        #     print(f"Location Sensor:\n"
        #           f"Model: "
        #           f"{station.location_sensor().name}\n"
        #           f"Sample rate, Hz: "
        #           f"{station.location_sensor().sample_rate_hz}\n"
        #           f"Sample interval, seconds: "
        #           f"{station.location_sensor().sample_interval_s}\n"
        #           f"Sample interval standard dev, seconds: "
        #           f"{station.location_sensor().sample_interval_std_s}\n"
        #           f"Number of GPS Points, Samples: "
        #           f"{station.location_sensor().num_samples()}\n")


if __name__ == "__main__":
    """
    Beta workflow for API M pipeline
    Last updated: 17 May 2021
    """
    print('Let the sky fall')
    print("Plot station information")

    rdvx_data: DataWindowFast = DataWindowFast.from_json_file(base_dir=OUTPUT_DIR,
                                                              file_name=DW_FILE)
    print("\nSave Station specs to file")
    station_file = "skyfall_station.csv"
    station_specs_to_csv(rdvx_data, station_file)

