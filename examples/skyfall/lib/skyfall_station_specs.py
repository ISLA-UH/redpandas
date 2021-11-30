import os
import csv

# RedVox and Red Pandas modules
from redvox.common.data_window import DataWindow

import redpandas.redpd_datawin as rpd_dw
import redpandas.redpd_dq as rpd_dq
from redvox.api1000.wrapped_redvox_packet.station_information import OsType
import redvox.common.date_time_utils as dt
from redvox.common.station import Station

# Configuration file
from examples.skyfall.skyfall_config_file import skyfall_config


def station_specs_to_csv(data_window: DataWindow,
                         export_file: str) -> None:
    """
    Export station spects to CSV

    :param data_window: RedVox DataWindow object
    :param export_file: full path to directory which to export file
    :return: save CSV to output directory provided
    """
    station: Station
    with open(export_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        for station in data_window.stations():
            # Station Make, Model, etc
            writer.writerow(["Description", "Field", "Value"])
            writer.writerow(["Station ID", "station.id", station.id])
            writer.writerow(["Make", "station.metadata().make", station.metadata().make])
            writer.writerow(["Model", "station.metadata().model", station.metadata().model])
            writer.writerow(["OS", "station.metadata().os", OsType(station.metadata().os).name])
            writer.writerow(["OS Version", "station.metadata().os_version", station.metadata().os_version])
            writer.writerow(["App Version", "station.metadata().app_version", station.metadata().app_version])
            writer.writerow(["SDK Version", "station.sdk_version()", data_window.sdk_version()])
            writer.writerow([])

            writer.writerow(["Station and Event Date"])
            writer.writerow([])
            writer.writerow(["Description", "Field", "Epoch s", "Human UTC"])
            if station.start_date() > 0:
                writer.writerow(["Station Start Date", "station.start_timestamp()",
                                 str(station.start_date()/1E6),
                                 dt.datetime_from_epoch_microseconds_utc(station.first_data_timestamp())])
            else:
                writer.writerow(["Station Start Date", "station.start_timestamp()",
                                 "N/A",
                                 "N/A before v2.6.3"])
            writer.writerow(["Event Start Date", "station.first_data_timestamp()",
                             str(station.first_data_timestamp()/1E6),
                             dt.datetime_from_epoch_microseconds_utc(station.first_data_timestamp())])
            writer.writerow(["Event End Date", "station.first_data_timestamp()",
                             str(station.last_data_timestamp()/1E6),
                             dt.datetime_from_epoch_microseconds_utc(station.last_data_timestamp())])

            writer.writerow([])
            writer.writerow(["Station Sensors"])
            if station.has_audio_data():
                writer.writerow([])
                writer.writerow(["audio"])
                writer.writerow(["Description", "Field", "Value"])
                writer.writerow(["Sensor Name", "station.audio_sensor().name",
                                 station.audio_sensor().name])
                writer.writerow(["Nominal Rate Hz", "station.audio_sample_rate_nominal_hz()",
                                 station.audio_sample_rate_nominal_hz()])
                writer.writerow(["Sample Rate Hz", "station.audio_sensor().sample_rate_hz()",
                                 station.audio_sensor().sample_rate_hz()])
                writer.writerow(["Sample Interval s", "station.audio_sensor().sample_interval_s()",
                                 station.audio_sensor().sample_interval_s()])
                writer.writerow(["Interval Dev s", "station.audio_sensor().sample_interval_std_s()",
                                 station.audio_sensor().sample_interval_std_s()])

            if station.has_barometer_data():
                writer.writerow([])
                writer.writerow(["barometer"])
                writer.writerow(["Description", "Field", "Value"])
                writer.writerow(["Sensor Name", "station.barometer_sensor().name",
                                 station.barometer_sensor().name])
                writer.writerow(["Sample Rate Hz", "station.barometer_sensor().sample_rate_hz()",
                                 station.barometer_sensor().sample_rate_hz()])
                writer.writerow(["Sample Interval s", "station.barometer_sensor().sample_interval_s()",
                                 station.barometer_sensor().sample_interval_s()])
                writer.writerow(["Interval Dev s", "station.barometer_sensor().sample_interval_std_s()",
                                 station.barometer_sensor().sample_interval_std_s()])

            if station.has_accelerometer_data():
                writer.writerow([])
                writer.writerow(["accelerometer"])
                writer.writerow(["Description", "Field", "Value"])
                writer.writerow(["Sensor Name", "station.accelerometer_sensor().name",
                                 station.accelerometer_sensor().name])
                writer.writerow(["Sample Rate Hz", "station.accelerometer_sensor().sample_rate_hz()",
                                 station.accelerometer_sensor().sample_rate_hz()])
                writer.writerow(["Sample Interval s", "station.accelerometer_sensor().sample_interval_s()",
                                 station.accelerometer_sensor().sample_interval_s()])
                writer.writerow(["Interval Dev s", "station.accelerometer_sensor().sample_interval_std_s()",
                                 station.accelerometer_sensor().sample_interval_std_s()])

            if station.has_gyroscope_data():
                writer.writerow([])
                writer.writerow(["gyroscope"])
                writer.writerow(["Description", "Field", "Value"])
                writer.writerow(["Sensor Name", "station.gyroscope_sensor().name",
                                 station.gyroscope_sensor().name])
                writer.writerow(["Sample Rate Hz", "station.gyroscope_sensor().sample_rate_hz()",
                                 station.gyroscope_sensor().sample_rate_hz()])
                writer.writerow(["Sample Interval s", "station.gyroscope_sensor().sample_interval_s()",
                                 station.gyroscope_sensor().sample_interval_s()])
                writer.writerow(["Interval Dev s", "station.gyroscope_sensor().sample_interval_std_s()",
                                 station.gyroscope_sensor().sample_interval_std_s()])

            if station.has_magnetometer_data():
                writer.writerow([])
                writer.writerow(["magnetometer"])
                writer.writerow(["Description", "Field", "Value"])
                writer.writerow(["Sensor Name", "station.magnetometer_sensor().name",
                                 station.magnetometer_sensor().name])
                writer.writerow(["Sample Rate Hz", "station.magnetometer_sensor().sample_rate_hz()",
                                 station.magnetometer_sensor().sample_rate_hz()])
                writer.writerow(["Sample Interval s", "station.magnetometer_sensor().sample_interval_s()",
                                 station.magnetometer_sensor().sample_interval_s()])
                writer.writerow(["Interval Dev s", "station.magnetometer_sensor().sample_interval_std_s()",
                                 station.magnetometer_sensor().sample_interval_std_s()])

            if station.has_location_data():
                writer.writerow([])
                writer.writerow(["location"])
                writer.writerow(["Description", "Field", "Value"])
                writer.writerow(["Sensor Name", "station.location_sensor().name",
                                 station.location_sensor().name])
                writer.writerow(["Sample Rate Hz", "station.location_sensor().sample_rate_hz()",
                                 station.location_sensor().sample_rate_hz()])
                writer.writerow(["Sample Interval s", "station.location_sensor().sample_interval_s()",
                                 station.location_sensor().sample_interval_s()])
                writer.writerow(["Interval Dev s", "station.location_sensor().sample_interval_std_s()",
                                 station.location_sensor().sample_interval_std_s()])


def main():
    """
    Beta workflow for API M pipeline
    """

    print("Print and save station information")

    rdvx_data = rpd_dw.dw_from_redpd_config(config=skyfall_config)
    rpd_dq.station_metadata(rdvx_data)

    print("\nSave Station specs to file")
    csv_station_file = skyfall_config.event_name + "_station.csv"
    csv_station_full_path = os.path.join(skyfall_config.output_dir, csv_station_file)

    station_specs_to_csv(rdvx_data, csv_station_full_path)


if __name__ == "__main__":
    main()
