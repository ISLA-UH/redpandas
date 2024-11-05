"""
DQ/DA/UQ statistics/metrics.
"""
from dataclasses import dataclass
from dataclasses_json import dataclass_json

import numpy as np

# RedVox modules
# from redvox.common.data_window import DataWindow
from redvox.common.data_window import DataWindow
from redvox.common.date_time_utils import MICROSECONDS_IN_SECOND
import redvox.common.date_time_utils as dt
# from redvox.common.station import Station
from redvox.common.data_window import Station
from redvox.api1000.wrapped_redvox_packet.station_information import OsType


@dataclass_json
@dataclass
class AudioDq:
    sample_rate: float
    sample_rate_is_fixed: bool
    sample_interval_seconds: float
    sample_interval_std: float
    first_sample_ts_microseconds: float
    last_sample_ts_microseconds: float


@dataclass_json
@dataclass
class StationDq:
    audio_dq: AudioDq
    timing_dq: int
    # etc


def mic_sync(data_window: DataWindow) -> None:
    """
    Print Audio Sensor information
    :param data_window: RedVox DataWindow object
    :return: print statements with Mic and Clock specs
    """
    station: Station
    for station in data_window.stations():
        if station.has_audio_data():
            print(f"{station.id()} Audio Sensor (All timestamps are in microseconds since epoch UTC):\n"
                  f"mic sample rate in hz: {station.audio_sensor().sample_rate_hz()}\n"
                  f"is mic sample rate constant: {station.audio_sensor().is_sample_rate_fixed()}\n"
                  f"mic sample interval in seconds: {station.audio_sensor().sample_interval_s()}\n"
                  f"mic sample interval std dev: {station.audio_sensor().sample_interval_std_s()}\n"
                  f"the first data timestamp: {station.audio_sensor().first_data_timestamp()}\n"
                  f"the last data timestamp:  {station.audio_sensor().last_data_timestamp()}\n"
                  f"the data as an ndarray: {station.audio_sensor().samples()}\n"
                  f"the number of data samples: {station.audio_sensor().num_samples()}\n"
                  f"the names of the dataframe columns: {station.audio_sensor().data_channels()}\n")

            mic_sample_rate_nominal_hz = station.audio_sample_rate_nominal_hz()
            mic_sample_rate_hz = station.audio_sensor().sample_rate_hz()
            # Unaltered time can have nans if packets missing or edges are truncated
            mic_unaltered_time_s = \
                station.audio_sensor().unaltered_data_timestamps() / MICROSECONDS_IN_SECOND
            # Corrected time should have no nans
            mic_corrected_time_s = \
                station.audio_sensor().data_timestamps() / MICROSECONDS_IN_SECOND

            print(f"\nMIC AND CLOCK SPECS: Station ID {station.id()}")
            if any(np.isnan(mic_corrected_time_s)) > 0:
                print(f'SYNCH WARNING: Have nans in data_timestamps\n'
                      f'Number Indices: {np.count_nonzero(np.isnan(mic_corrected_time_s))}')
            else:
                print('No nans in corrected data_timestamps')
            if any(np.isnan(mic_unaltered_time_s)) > 0:
                print(f'Nans in unaltered_data_timestamps\n'
                      f'Number Indices: {np.count_nonzero(np.isnan(mic_unaltered_time_s))}')

            print(f'App start time: {station.start_date()}')
            print(f'Clock model start time: {station.timesync_data().offset_model().start_time}')
            if np.abs(station.timesync_data().offset_model().intercept) == 0:
                print('ZERO OFFSET, NO CORRECTION')
            else:
                print(f'Offset, microseconds: {station.timesync_data().offset_model().intercept}')
                print(f'Mean best latency: {station.timesync_data().offset_model().mean_latency}')
                print(f'Best latency std dev: {station.timesync_data().offset_model().std_dev_latency}')
                print(f'Number bins: {station.timesync_data().offset_model().k_bins}')
                print(f'Min number of samples: {station.timesync_data().offset_model().n_samples}')

            if np.abs(station.timesync_data().offset_model().slope) == 0:
                print('NO SLOPE, CONSTANT OFFSET')
            else:
                print(f'Slope: {station.timesync_data().offset_model().slope}')
                print(f'Regression score: {station.timesync_data().offset_model().score}')

            print(f'Nominal sample rate, Hz: {mic_sample_rate_nominal_hz}')
            print(f'Corrected sample rate, Hz: {mic_sample_rate_hz}')

            # Sample rate check
            mic_sample_interval_from_dt = np.mean(np.diff(station.audio_sensor().data_timestamps()))
            mic_sample_rate_from_dt = MICROSECONDS_IN_SECOND / mic_sample_interval_from_dt

            print(f'Sample rate from dif mic time: {mic_sample_rate_from_dt}')
            sample_rate_percent_error = ((mic_sample_rate_from_dt - mic_sample_rate_nominal_hz)
                                         / mic_sample_rate_nominal_hz) * 100.
            print(f"Percent sample rate computation error: {sample_rate_percent_error:.2f} %")
        else:
            # There should ALWAYS be mic data.
            print(f'NO MIC DATA IN STATION {station.id()}, SOMETHING IS AMISS')
            continue


def station_channel_timing(data_window: DataWindow) -> None:
    """
    Print RedVox DataWindow Station channel time information for Audio, Barometer, Accelerometer, Gyroscope and
    Magnetometer sensors.

    :param data_window: RedVox DataWindow object
    :return: print statement with Station timing and sensors specs
    """
    station: Station
    for station in data_window.stations():
        print(f"STATION CHANNEL TIMING FOR ID {station.id()}")
        if station.first_data_timestamp() > 0:
            print(f'App start time: '
                  f'{dt.datetime_from_epoch_microseconds_utc(station.first_data_timestamp())}')
        else:
            print('App start time not available')
        print(f'Station first time stamp: '
              f'{dt.datetime_from_epoch_microseconds_utc(station.first_data_timestamp())}')
        print(f'Station last time stamp: '
              f'{dt.datetime_from_epoch_microseconds_utc(station.last_data_timestamp())}')

        if station.has_audio_data():
            print(f"\naudio Sensor:\n"
                  f"audio first data timestamp: "
                  f"{dt.datetime_from_epoch_microseconds_utc(station.audio_sensor().first_data_timestamp())}\n"
                  f"audio last data timestamp: "
                  f"{dt.datetime_from_epoch_microseconds_utc(station.audio_sensor().last_data_timestamp())}\n")
        else:
            print("WARNING: NO MIC DATA - INSPECT ISSUE")
            continue

        print("Print Data Window edge time differences for all sensors; verify they are zero")
        if station.has_barometer_data():
            barometer_first_timestamp_delta = (station.barometer_sensor().first_data_timestamp() -
                                               station.audio_sensor().first_data_timestamp())
            barometer_last_timestamp_delta = (station.barometer_sensor().last_data_timestamp() -
                                              station.audio_sensor().last_data_timestamp())
            print(f"barometer Sensor:\n"
                  f"barometer first data timestamp diff from mic: "
                  f"{barometer_first_timestamp_delta}\n"
                  f"barometer last data timestamp diff from mic: "
                  f"{barometer_last_timestamp_delta}\n")

        if station.has_accelerometer_data():
            accelerometer_first_timestamp_delta = (station.accelerometer_sensor().first_data_timestamp() -
                                                   station.audio_sensor().first_data_timestamp())
            accelerometer_last_timestamp_delta = (station.accelerometer_sensor().last_data_timestamp() -
                                                  station.audio_sensor().last_data_timestamp())
            print(f"accelerometer Sensor:\n"
                  f"accelerometer first data timestamp diff from mic: "
                  f"{accelerometer_first_timestamp_delta}\n"
                  f"accelerometer last data timestamp diff from mic: "
                  f"{accelerometer_last_timestamp_delta}\n")

        if station.has_magnetometer_data():
            magnetometer_first_timestamp_delta = (station.magnetometer_sensor().first_data_timestamp() -
                                                  station.audio_sensor().first_data_timestamp())
            magnetometer_last_timestamp_delta = (station.magnetometer_sensor().last_data_timestamp() -
                                                 station.audio_sensor().last_data_timestamp())
            print(f"magnetometer Sensor:\n"
                  f"magnetometer first data timestamp diff from mic: "
                  f"{magnetometer_first_timestamp_delta}\n"
                  f"magnetometer last data timestamp diff from mic: "
                  f"{magnetometer_last_timestamp_delta}\n")

        if station.has_gyroscope_data():
            gyroscope_first_timestamp_delta = (station.gyroscope_sensor().first_data_timestamp() -
                                               station.audio_sensor().first_data_timestamp())
            gyroscope_last_timestamp_delta = (station.gyroscope_sensor().last_data_timestamp() -
                                              station.audio_sensor().last_data_timestamp())
            print(f"gyroscope Sensor:\n"
                  f"gyroscope first data timestamp diff from mic: "
                  f"{gyroscope_first_timestamp_delta}\n"
                  f"gyroscope last data timestamp diff from mic: "
                  f"{gyroscope_last_timestamp_delta}\n")


def station_metadata(data_window: DataWindow) -> None:
    """
    Print RedVox DataWindow Station metadata

    :param data_window: RedVox DataWindow object
    :return: print statements with Station specs
    """
    station: Station
    for station in data_window.stations():
        if station.first_data_timestamp() > 0:
            print(f"STATION SPECS FOR ID: " 
                  f"{station.id()}\n"
                  f"App start time: "
                  f"{dt.datetime_from_epoch_microseconds_utc(station.first_data_timestamp())}\n"
                  f"Station first time stamp: "
                  f"{dt.datetime_from_epoch_microseconds_utc(station.first_data_timestamp())}\n"
                  f"Station last time stamp: "
                  f"{dt.datetime_from_epoch_microseconds_utc(station.last_data_timestamp())}\n")
        else:
            print(f"STATION SPECS FOR ID: "
                  f"{station.id()}\n"
                  f"App start time not available\n"
                  f"Station first time stamp: "
                  f"{dt.datetime_from_epoch_microseconds_utc(station.first_data_timestamp())}\n"
                  f"Station last time stamp: "
                  f"{dt.datetime_from_epoch_microseconds_utc(station.last_data_timestamp())}\n")

        print(f"Station Metadata:\n"
              f"Make: "
              f"{station.metadata().make}\n"
              f"Model: "
              f"{station.metadata().model}\n"
              f"OS: "
              f"{OsType(station.metadata().os).name}\n"
              f"OS version: "
              f"{station.metadata().os_version}\n"
              f"App Version: "
              f"{station.metadata().app_version}\n")

        if station.has_audio_data():
            print(f"\nAudio Sensor:\n"
                  f"Model: "
                  f"{station.audio_sensor().name}\n"
                  f"Sample rate, Hz: "
                  f"{station.audio_sensor().sample_rate_hz()}\n"
                  f"Sample interval, seconds: "
                  f"{station.audio_sensor().sample_interval_s()}\n"
                  f"Sample interval standard dev, seconds: "
                  f"{station.audio_sensor().sample_interval_std_s()}\n")
        if station.has_barometer_data():
            print(f"Barometer Sensor:\n"
                  f"Model: "
                  f"{station.barometer_sensor().name}\n"
                  f"Sample rate, Hz: "
                  f"{station.barometer_sensor().sample_rate_hz()}\n"
                  f"Sample interval, seconds: "
                  f"{station.barometer_sensor().sample_interval_s()}\n"
                  f"Sample interval standard dev, seconds: "
                  f"{station.barometer_sensor().sample_interval_std_s()}\n")
        if station.has_accelerometer_data():
            print(f"Accelerometer Sensor:\n"
                  f"Model: "
                  f"{station.accelerometer_sensor().name}\n"
                  f"Sample rate, Hz: "
                  f"{station.accelerometer_sensor().sample_rate_hz()}\n"
                  f"Sample interval, seconds: "
                  f"{station.accelerometer_sensor().sample_interval_s()}\n"
                  f"Sample interval standard dev, seconds: "
                  f"{station.accelerometer_sensor().sample_interval_std_s()}\n")
        if station.has_magnetometer_data():
            print(f"Magnetometer Sensor:\n"
                  f"Model: "
                  f"{station.magnetometer_sensor().name}\n"
                  f"Sample rate, Hz: "
                  f"{station.magnetometer_sensor().sample_rate_hz()}\n"
                  f"Sample interval, seconds: "
                  f"{station.magnetometer_sensor().sample_interval_s()}\n"
                  f"Sample interval standard dev, seconds: "
                  f"{station.magnetometer_sensor().sample_interval_std_s()}\n")
        if station.has_gyroscope_data():
            print(f"Gyroscope Sensor:\n"
                  f"Model: "
                  f"{station.gyroscope_sensor().name}\n"
                  f"Sample rate, Hz: "
                  f"{station.gyroscope_sensor().sample_rate_hz()}\n"
                  f"Sample interval, seconds: "
                  f"{station.gyroscope_sensor().sample_interval_s()}\n"
                  f"Sample interval standard dev, seconds: "
                  f"{station.gyroscope_sensor().sample_interval_std_s()}\n")
        if station.has_location_sensor():
            print(f"Location Sensor:\n"
                  f"Model: "
                  f"{station.location_sensor().name}\n"
                  f"Sample rate, Hz: "
                  f"{station.location_sensor().sample_rate_hz()}\n"
                  f"Sample interval, seconds: "
                  f"{station.location_sensor().sample_interval_s()}\n"
                  f"Sample interval standard dev, seconds: "
                  f"{station.location_sensor().sample_interval_std_s()}\n"
                  f"Number of GPS Points, Samples: "
                  f"{station.location_sensor().num_samples()}\n")
