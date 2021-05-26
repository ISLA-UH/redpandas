"""
This module contains general utilities that can work with values containing nans.
"""
# TODO MC: finish build station luminosity

from typing import List

import numpy as np
from redvox.common.station import Station

# RedPandas library
import redpandas.redpd_preprocess as rpd_prep
import redpandas.redpd_scales as rpd_scales


def station_to_dict_from_dw(
        station: Station,
        sdk_version: str,
        sensor_labels: List[str]):
    """
    converts information from a station object created by a data window into a dictionary easily converted into
    a dataframe
    :param station: RDVX station object
    :param sdk_version: version of Redvox SDK used to create the Station object
    :param sensor_labels: the names of the sensors to extract
    :return: a dictionary ready for conversion into a dataframe
    """
    sensors = {"station_id": station.id,
               'station_start_date_epoch_micros': station.start_timestamp,
               'station_make': station.metadata.make,
               'station_model': station.metadata.model,
               'station_app_version': station.metadata.app_version,
               'redvox_sdk_version': sdk_version}

    print(f"Prep Station {station.id}...", end=" ")
    for label in sensor_labels:
        print(f"{label} sensor...", end=" ")
        df_sensor = build_station(station=station, sensor_label=label)
        if len(df_sensor.values()) > 0:
            sensors.update(df_sensor)
    print(f"Done.")
    return sensors


def sensor_uneven(station: Station, sensor_label: str):
    """
    ID nans, sample rate, epoch, raw of uneven sensor

    :param station: RDVX Station object
    :param sensor_label: one of: ['barometer', 'accelerometer', 'gyroscope', 'magnetometer']
    :return: sensor sample rate (Hz), timestamps, raw data and nans in sensor.
    """

    # default parameters
    sensor_sample_rate_hz = None
    sensor_epoch_s = None
    sensor_raw = None
    sensor_nans = None

    if eval('station.has_' + sensor_label + '_data()'):
        sensor_dw = eval('station.' + sensor_label + '_sensor()')
        sensor_sample_rate_hz = sensor_dw.sample_rate_hz
        sensor_epoch_s = sensor_dw.data_timestamps() * rpd_scales.MICROS_TO_S
        sensor_raw = sensor_dw.samples()
        sensor_nans = np.argwhere(np.isnan(sensor_raw))
    else:
        print(f'Station {station.id} has no {sensor_label} data.')

    return sensor_sample_rate_hz, sensor_epoch_s, sensor_raw, sensor_nans


# Build station modules
def build_station(station: Station,
                  sensor_label: str,
                  highpass_type: str = 'obspy',
                  frequency_filter_low: float = 1./rpd_scales.Slice.T100S,
                  filter_order: int = 4) -> dict:
    """
    Obtain sensor data from RDVX station

    :param station: RDVX Station object
    :param sensor_label: one of: ['audio', 'barometer', 'accelerometer', 'gyroscope', 'magnetometer',
    'health', 'location', 'image']
    :param highpass_type: 'obspy', 'butter', 'rc', default 'obspy'
    :param frequency_filter_low: todo what is this 100s default
    :param filter_order: todo what is this Default = 4
    :return: dictionary with sensor name, sample rate, timestamps, data (raw and highpassed)
    """
    if sensor_label == 'mic' or sensor_label == 'microphone' or sensor_label == 'audio':
        return audio_wf_time_build_station(station=station, mean_type='simple', raw=False)

    elif sensor_label == 'location' or sensor_label == 'loc':
        return location_build_station(station=station)

    elif sensor_label == 'clock':
        return clock_build_station(station=station)

    elif sensor_label == 'synchronization' or sensor_label == 'sync':
        return synchronization_build_station(station=station)

    elif sensor_label == 'health' or sensor_label == 'soh':
        return state_of_health_build_station(station=station)

    elif sensor_label == 'image' or sensor_label == 'im':
        return image_build_station(station=station)

    else:
        sensor_sample_rate_hz, sensor_epoch_s, sensor_raw, sensor_nans = sensor_uneven(station=station,
                                                                                       sensor_label=sensor_label)

        list_sensor_highpass = []
        if sensor_sample_rate_hz:
            for index_dimension, _ in enumerate(sensor_raw):
                sensor_waveform_highpass, _ = rpd_prep.highpass_from_diff(sensor_waveform=sensor_raw[index_dimension],
                                                                          sensor_epoch_s=sensor_epoch_s,
                                                                          sample_rate=sensor_sample_rate_hz,
                                                                          highpass_type=highpass_type,
                                                                          frequency_filter_low=frequency_filter_low,
                                                                          filter_order=filter_order)
                # print(sensor_waveform_highpass)
                list_sensor_highpass.append(sensor_waveform_highpass)

            return {f'{sensor_label}_sensor_name': eval('station.' + sensor_label + '_sensor()').name,
                    f'{sensor_label}_sample_rate_hz': sensor_sample_rate_hz,
                    f'{sensor_label}_epoch_s': sensor_epoch_s,
                    f'{sensor_label}_wf_raw': sensor_raw,
                    f'{sensor_label}_wf_highpass': np.array(list_sensor_highpass),
                    f'{sensor_label}_nans': sensor_nans}
        else:
            print(f"{sensor_label} doesn't exist in the station.")
            return {}


# Modules for specific sensors
def audio_wf_time_build_station(station: Station,
                                mean_type: str = "simple",
                                raw: bool = False) -> dict:
    """
    Builds mic waveform and times if it exists
    :param station: RDVX Station object
    :param mean_type: todo: under development
    :param raw: if false (default), boolean or nan mean removed
    :return: dictionary with sensor name, sample rate, timestamps, audio data
    """
    if station.has_audio_data():
        mic_wf_raw = station.audio_sensor().get_data_channel("microphone")
        mic_epoch_s = station.audio_sensor().data_timestamps() * rpd_scales.MICROS_TO_S
        mic_nans = np.array(np.argwhere(np.isnan(mic_wf_raw)))

        if raw:
            mic_wf = np.array(mic_wf_raw)
        else:
            if mean_type == "simple":
                # Simple demean and replace nans with zeros. OK for mic, not OK for all other DC-biased sensors
                mic_wf = rpd_prep.demean_nan(mic_wf_raw)
            else:
                # Remove linear trend
                mic_wf = rpd_prep.detrend_nan(mic_wf_raw)

        return {'audio_sensor_name': station.audio_sensor().name,
                'audio_sample_rate_nominal_hz': station.audio_sample_rate_nominal_hz,
                'audio_sample_rate_corrected_hz': station.audio_sensor().sample_rate_hz,
                'audio_epoch_s': mic_epoch_s,
                'audio_wf_raw': mic_wf_raw,
                'audio_wf': mic_wf,
                'audio_nans': mic_nans.tolist()}
    else:
        print(f'Station {station.id} has no audio data.')
        return {}


def location_build_station(station: Station) -> dict:
    """
    Obtains location data from RedVox station if it exists
    :param station: RDVX Station object
    :return: dictionary with sensor name, sample rate, timestamps, latitude, longitude, altitude, bearing, speed,
    horizontal accuracy, vertical accuracy, bearing accuracy, speed accuracy, and location provider.
    """
    if station.has_location_data():
        return {'location_sensor_name': station.location_sensor().name,
                'location_sample_rate_hz': station.location_sensor().sample_rate_hz,
                'location_epoch_s': station.location_sensor().data_timestamps() * rpd_scales.MICROS_TO_S,
                'location_latitude': station.location_sensor().get_data_channel("latitude"),
                'location_longitude': station.location_sensor().get_data_channel("longitude"),
                'location_altitude': station.location_sensor().get_data_channel("altitude"),
                'location_bearing': station.location_sensor().get_data_channel("bearing"),
                'location_speed': station.location_sensor().get_data_channel("speed"),
                'location_horizontal_accuracy':
                    station.location_sensor().get_data_channel("horizontal_accuracy"),
                'location_vertical_accuracy': station.location_sensor().get_data_channel("vertical_accuracy"),
                'location_bearing_accuracy': station.location_sensor().get_data_channel("bearing_accuracy"),
                'location_speed_accuracy': station.location_sensor().get_data_channel("speed_accuracy"),
                'location_provider': station.location_sensor().get_data_channel("location_provider")}
    else:
        print(f'Station {station.id} has no location data.')
        return {}


def state_of_health_build_station(station: Station) -> dict:
    """
    Obtains state of health data from RedVox station if it exists
    :param station: RDVX Station object
    :return: dictionary with sensor name, sample rate, timestamps, batt. charge, batt. current strength,
    internal temp., network type, network strength, power state, available ram and disk, and cell service state
    """
    if station.has_health_data():
        return {'health_sensor_name': station.health_sensor().name,
                'health_sample_rate_hz': station.health_sensor().sample_rate_hz,
                'health_epoch_s': station.health_sensor().data_timestamps() * rpd_scales.MICROS_TO_S,
                'battery_charge_remaining_per':
                    station.health_sensor().get_data_channel('battery_charge_remaining'),
                'battery_current_strength_mA':
                    station.health_sensor().get_data_channel('battery_current_strength'),
                'internal_temp_deg_C': station.health_sensor().get_data_channel('internal_temp_c'),
                'network_type': station.health_sensor().get_data_channel('network_type'),
                'network_strength_dB': station.health_sensor().get_data_channel('network_strength'),
                'power_state': station.health_sensor().get_data_channel('power_state'),
                'available_ram_byte': station.health_sensor().get_data_channel('avail_ram'),
                'available_disk_byte': station.health_sensor().get_data_channel('avail_disk'),
                'cell_service_state': station.health_sensor().get_data_channel('cell_service')}
    else:
        print(f'Station {station.id} has no health data.')
        return {}


def image_build_station(station: Station) -> dict:
    """
    Obtains images from RedVox station if it exists
    :param station: RDVX Station object
    :return: dictionary with sensor name, sample rate, timestamps, image (bytes), and image codec.
    """
    if station.has_image_data():
        return {'image_sensor_name': station.image_sensor().name,
                'image_sample_rate_hz': station.image_sensor().sample_rate_hz,
                'image_epoch_s': station.image_sensor().data_timestamps() * rpd_scales.MICROS_TO_S,
                'image_bytes': station.image_sensor().get_data_channel('image'),
                'image_codec': station.image_sensor().get_data_channel('image_codec')}
    else:
        print(f'Station {station.id} has no image data.')
        return {}


def synchronization_build_station(station: Station) -> dict:
    """
    Obtains time sync data from RedVox station if it exists
    :param station: RDVX Station object
    :return: dictionary with synchronization start time (s), synchronization latency (ms), synchronization offset (ms),
    synchronization best offset (ms), synchronization offset delta (ms), and synchronization number exchanges.
    """
    if station.has_timesync_data():
        synchronization = station.timesync_analysis
        return {'synchronization_epoch_s': synchronization.get_start_times() * rpd_scales.MICROS_TO_S,
                'synchronization_latency_ms': synchronization.get_latencies() * rpd_scales.MICROS_TO_MILLIS,
                'synchronization_offset_ms': synchronization.get_offsets() * rpd_scales.MICROS_TO_MILLIS,
                'synchronization_best_offset_ms': synchronization.get_best_offset() * rpd_scales.MICROS_TO_MILLIS,
                'synchronization_offset_delta_ms': synchronization.get_offsets() * rpd_scales.MICROS_TO_MILLIS -
                                                    synchronization.get_best_offset() * rpd_scales.MICROS_TO_MILLIS,
                'synchronization_number_exchanges': synchronization.timesync_data[0].num_tri_messages()}
    else:
        print(f'Station {station.id} has no time sync data.')
        return {}


def clock_build_station(station: Station) -> dict:
    """
    Obtains clock model data from the station if it exists
    :param station: RDVX Station object
    :return: dictionary with clock start time (s), clock latency (ms), clock best latency (ms), clock offset (s),
    clock number bins, clock number samples, clock offset slope, and clock offset model score.
    """
    if station.has_timesync_data():
        print('App start time s:', station.start_timestamp)
        clock = station.timesync_analysis.offset_model
        return {'clock_start_time_epoch_s': clock.start_time * rpd_scales.MICROS_TO_S,
                'clock_best_latency_ms': clock.mean_latency * rpd_scales.MICROS_TO_MILLIS,
                'clock_best_latency_std_ms': clock.std_dev_latency * rpd_scales.MICROS_TO_MILLIS,
                'clock_offset_s': clock.intercept * rpd_scales.MICROS_TO_S,
                'clock_number_bins': clock.k_bins,
                'clock_number_samples': clock.n_samples,
                'clock_offset_slope': clock.slope,
                'clock_offset_model_score': clock.score}
    else:
        print(f'Station {station.id} has no timesync analysis.')
        return {}


def light_build_station(station: Station) -> dict:
    """
    Obtains luminosity data from RedVox station if it exists
    :param station: RDVX Station object
    :return:
    """
    if station.has_light_data():

        return {'light_sensor_name': station.light_sensor().name,
                'light_sample_rate_hz': station.light_sensor().sample_rate_hz,
                'light_epoch_s': station.light_sensor().data_timestamps() * rpd_scales.MICROS_TO_S}
    else:
        print(f'Station {station.id} has no luminosity data.')
        return {}
