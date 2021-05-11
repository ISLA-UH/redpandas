from enum import Enum

import numpy as np
import pandas as pd
from redvox.api1000.wrapped_redvox_packet.station_information import NetworkType, PowerState, CellServiceState
from redvox.common.station_raw import StationRaw

import pipeline_m.redpd_preprocess as rpd_prep
import pipeline_m.redpd_scales as rpd_scales


"""
This module contains general utilities that can work with values containing nans.
"""
# TODO: Does synchronization still work
# TODO: build luminosity
# TODO: Build station timing

# Define classes
class NormType(Enum):
    """
    Enumeration of normalization types.
    """
    MAX: str = "max"
    L1: str = "l1"
    L2: str = "l2"
    OTHER: str = "other"


def sensor_uneven(station: StationRaw,
                  sensor_label: str):
    """
    ID nans, sample rate, epoch, raw of uneven sensor
    :param station:
    :param sensor_label:
    :return:
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


# TODO: Clean up
# def sensor_uneven_3c(sensor_x, sensor_y, sensor_z, sensor_epoch, sensor_sample_rate):
#
#     mag_waveform = [sensor_x, sensor_y, sensor_z]
#     mag_epoch_diff = sensor_epoch[:-1]
#
#     mag_diff = np.diff(mag_waveform)
#
#     mag_waveform_dp_filtered_all = []
#     mag_waveform_reconstruct_all = []
#
#     for vector in range(len(mag_waveform)):
#
#         # condition for when the record is too short for high pass at 100 seconds.
#         # Needs to be at least 200 seconds
#         if sensor_epoch[-1] - sensor_epoch[0] < 200:
#             raise ValueError('Sensor record is too short for performing scalogram computations.' +
#                              'Please provide a record longer than 200 seconds (3 minutes and 20 seconds).')
#
#         # apply high pass filter at 100 seconds for sample rate calculated from the mag sensor.
#         mag_waveform_dp_filtered = obspy.signal.filter.highpass(mag_diff[vector], 0.01, sensor_sample_rate, corners=4,
#                                                                 zerophase=True)
#         mag_waveform_dp_filtered_all.append(mag_waveform_dp_filtered)
#
#         # mag data reconstruct reconstruct mag from dP: P(0) = 0, P(i) = dP(i) + P(i-1)
#         mag_waveform_reconstruct = np.zeros(len(mag_epoch_diff))
#         mag_waveform_reconstruct[0] = 0
#
#         # loop over the time series and reconstruct each data point
#         for j in range(1, len(mag_waveform_reconstruct)):
#             mag_waveform_reconstruct[j] = mag_waveform_dp_filtered[j] + mag_waveform_reconstruct[j - 1]
#
#         mag_waveform_reconstruct_all.append(mag_waveform_reconstruct)
#
#     # calculate and subtract the mean of each vector from each vector to remove the "DC offset".
#     # Loop through each vector.
#     mag_waveform_mean_removed_all = []
#
#     for vector in range(len(mag_waveform_reconstruct_all)):
#         vector_mean = np.mean(mag_waveform_reconstruct_all[vector])
#
#         # subtract the mean of each vector from each value in that vector
#         mag_waveform_mean_removed = []
#         for j in range(len(mag_waveform_reconstruct_all[vector])):
#             mag_waveform_mean_removed = mag_waveform_reconstruct_all[vector] - vector_mean
#         mag_waveform_mean_removed_all.append(mag_waveform_mean_removed)
#
#     # calculate/ display the mean of each vector AFTER removing the mean
#     mag_x_mean_removed = mag_waveform_mean_removed_all[0]
#     mag_y_mean_removed = mag_waveform_mean_removed_all[1]
#     mag_z_mean_removed = mag_waveform_mean_removed_all[2]
#
#     # calculate the max of all the components for use in the plot limits later on.
#     mag_x_max_of_mean = np.max(mag_x_mean_removed)
#     mag_y_max_of_mean = np.max(mag_y_mean_removed)
#     mag_z_max_of_mean = np.max(mag_z_mean_removed)
#     max_all_components = [mag_x_max_of_mean, mag_y_max_of_mean, mag_z_max_of_mean]
#     max_final = np.round(np.max(max_all_components))
#     min_final = -max_final
#
#     # plot setup for displaying the mean removed signal from each component
#     df = pd.DataFrame({'x': mag_epoch_diff, 'x-component': mag_x_mean_removed, 'y-component': mag_y_mean_removed,
#                        'z-component': mag_z_mean_removed})
#
#     # Calculate mag intensity from all three components
#     mag_intensity_all = []
#     for j in range(len(mag_x_mean_removed)):
#         mag_intensity = np.sqrt(((mag_x_mean_removed[j]) ** 2) + ((mag_y_mean_removed[j]) ** 2) +
#                                 ((mag_z_mean_removed[j]) ** 2))
#         mag_intensity_all.append(mag_intensity)
#
#     # plot setup for mag intensity plot
#     df2 = pd.DataFrame({'x': mag_epoch_diff, 'y': mag_intensity_all})
#
#     # # calculate the fft
#     # mag_sample_rate = [mag_sample_rate]
#     # mag_intensity_array = [np.array(mag_intensity_all)]
#     # time_fft, frequency_fft, energy_fft, snr_fft = spectra.spectra_fft(mag_intensity_array, mag_sample_rate,
#     #                                                                    redvox_id, minimum_frequency=0.1)
#     #
#     # # calculate fft limits for plotting
#     # max_energy = (np.amax(energy_fft))
#     # min_energy = (np.amin(energy_fft))

# Build station modules
def build_station(station: StationRaw,
                  sensor_label: str,
                  highpass_type: str = 'obspy',
                  frequency_filter_low: float = 1./rpd_scales.Slice.T100S,
                  filter_order: int = 4) -> pd.DataFrame:

    """
    Obtain sensor data from RDVX station
    :param station: RDVX DataWindow station object
    :param sensor_label: {'audio', 'barometer', 'accelerometer', 'gyroscope', 'magnetometer',
    'health', 'location', 'image'}
    :param highpass_type: 'obspy', 'butter', 'rc'
    :param frequency_filter_low: 100s default
    :param filter_order: Default is 4.
    :return: pd.DataFrame with with sensor name, sample rate, timestamps, data (raw and highpassed)
    """
    if sensor_label == 'mic' or sensor_label == 'microphone' or sensor_label == 'audio':
        df_sensor = audio_wf_time_build_station(station=station,
                                                mean_type='simple',
                                                raw=False)

    elif sensor_label == 'location' or sensor_label == 'loc':
        df_sensor = location_build_station(station=station)

    elif sensor_label == 'synchronization' or sensor_label == 'synch':
        df_sensor = synchronization_build_station(station=station)

    elif sensor_label == 'health' or sensor_label == 'soh':
        df_sensor = state_of_health_build_station(station=station)

    elif sensor_label == 'image' or sensor_label == 'im':
        df_sensor = image_build_station(station=station)

    else:
        sensor_sample_rate_hz, sensor_epoch_s, sensor_raw, sensor_nans = sensor_uneven(station=station,
                                                                                       sensor_label=sensor_label)

        list_sensor_highpass = []
        for index_dimension, _ in enumerate(sensor_raw):
            sensor_waveform_highpass, _ = rpd_prep.highpass_from_diff(sensor_waveform=sensor_raw[index_dimension],
                                                                      sensor_epoch_s=sensor_epoch_s,
                                                                      sample_rate=sensor_sample_rate_hz,
                                                                      highpass_type=highpass_type,
                                                                      frequency_filter_low=frequency_filter_low,
                                                                      filter_order=filter_order)
            # print(sensor_waveform_highpass)
            list_sensor_highpass.append(sensor_waveform_highpass)

        dict_for_df_sensor = {f'{sensor_label}_sensor_name': eval('station.' + sensor_label + '_sensor()').name,
                              f'{sensor_label}_sample_rate_hz': [sensor_sample_rate_hz],
                              f'{sensor_label}_epoch_s': [sensor_epoch_s],
                              f'{sensor_label}_wf_raw': [sensor_raw],
                              f'{sensor_label}_wf_highpass': [np.array(list_sensor_highpass)],
                              f'{sensor_label}_nans': [sensor_nans]}

        df_sensor = pd.DataFrame.from_dict(data=dict_for_df_sensor)

    return df_sensor


# Modules for specific sensors
def audio_wf_time_build_station(station: StationRaw,
                                mean_type: str = "simple",
                                raw: bool = False) -> pd.DataFrame:
    """
    Builds mic waveform and times
    :param station: RDVX DataWindow station object
    :param mean_type: under development
    :param raw: if false (default), boolean or nan mean removed
    :return: pd.DataFrame with sensor name, sample rate, timestamps, audio data
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

        dict_for_df_mic = {'audio_sensor_name': station.audio_sensor().name,
                           'audio_sample_rate_nominal_hz': [station.audio_sample_rate_nominal_hz],
                           'audio_sample_rate_corrected_hz': [station.audio_sensor().sample_rate_hz],
                           'audio_epoch_s': [mic_epoch_s],
                           'audio_wf_raw': [mic_wf_raw],
                           'audio_wf': [mic_wf],
                           'audio_nans': [mic_nans.tolist()]}

        df_audio = pd.DataFrame.from_dict(data=dict_for_df_mic)
        return df_audio
    else:
        print(f'Station {station.id} has no audio data.')


def location_build_station(station: StationRaw) -> pd.DataFrame:
    """
    Obtains location data from station
    :param station: RDVX DataWindow station object
    :return: pd.DataFrame with sensor name, sample rate, timestamps, latitude, longitude, altitude, bearing, speed,
    horizontal accuracy, vertical accuracy, bearing accuracy, speed accuracy, and location provider.
    """
    if station.has_location_data():
        dict_for_loc = {'location_sensor_name': [station.location_sensor().name],
                        'location_sample_rate_hz': [station.location_sensor().sample_rate_hz],
                        'location_epoch_s': [station.location_sensor().data_timestamps() * rpd_scales.MICROS_TO_S],
                        'location_latitude': [station.location_sensor().get_data_channel("latitude")],
                        'location_longitude': [station.location_sensor().get_data_channel("longitude")],
                        'location_altitude': [station.location_sensor().get_data_channel("altitude")],
                        'location_bearing': [station.location_sensor().get_data_channel("bearing")],
                        'location_speed': [station.location_sensor().get_data_channel("speed")],
                        'location_horizontal_accuracy':
                            [station.location_sensor().get_data_channel("horizontal_accuracy")],
                        'location_vertical_accuracy': [station.location_sensor().get_data_channel("vertical_accuracy")],
                        'location_bearing_accuracy': [station.location_sensor().get_data_channel("bearing_accuracy")],
                        'location_speed_accuracy': [station.location_sensor().get_data_channel("speed_accuracy")]}
                        # 'location_provider': [station.location_sensor().get_data_channel("location_provider")]}

        df_loc = pd.DataFrame.from_dict(data=dict_for_loc)
        return df_loc
    else:
        print(f'Station {station.id} has no location data.')


def state_of_health_build_station(station: StationRaw) -> pd.DataFrame:
    """
    Obtains state of health data from station
    :param station: RDVX DataWindow station object
    :return: pd.DataFrame with sensor name, sample rate, timestamps, batt. charge, batt. current strength,
    internal temp., network type, network strength, power state, available ram and disk, and cell service state
    """
    if station.has_health_data():

        dict_enum = {'network_type': NetworkType,
                     'power_state': PowerState,
                     'cell_service': CellServiceState}

        list_all_enum_names = []
        for enum_type in dict_enum:
            list_enum_element_name = []
            for power_element in station.health_sensor().get_data_channel(enum_type):
                try:
                    list_enum_element_name.append(dict_enum[enum_type](power_element).name)

                except ValueError:
                    list_enum_element_name.append('Nan')
            list_all_enum_names.append(np.array(list_enum_element_name))

        dict_for_soh = {'health_sensor_name': [station.health_sensor().name],
                        'health_sample_rate_hz': [station.health_sensor().sample_rate_hz],
                        'health_epoch_s': [station.health_sensor().data_timestamps() * rpd_scales.MICROS_TO_S],
                        'battery_charge_remaining_per':
                            [station.health_sensor().get_data_channel('battery_charge_remaining')],
                        'battery_current_strength_mA':
                            [station.health_sensor().get_data_channel('battery_current_strength')],
                        'internal_temp_deg_C': [station.health_sensor().get_data_channel('internal_temp_c')],
                        'network_type': [list_all_enum_names[0]],
                        'network_strength_dB': [station.health_sensor().get_data_channel('network_strength')],
                        'power_state': [list_all_enum_names[1]],
                        'available_ram_byte': [station.health_sensor().get_data_channel('avail_ram')],
                        'available_disk_byte': [station.health_sensor().get_data_channel('avail_disk')],
                        'cell_service_state': [list_all_enum_names[2]]}

        df_soh = pd.DataFrame.from_dict(data=dict_for_soh)
        return df_soh

    else:
        print(f'Station {station.id} has no health data.')


def image_build_station(station: StationRaw) -> pd.DataFrame:
    """
    Obtains images from station
    :param station: RDVX DataWindow station object
    :return: pd.DataFrame with sensor name, sample rate, timestamps, image (bytes), and image codec.
    """
    if station.has_image_data():

        dict_for_im = {'image_sensor_name': [station.image_sensor().name],
                       'image_sample_rate_hz': [station.image_sensor().sample_rate_hz],
                       'image_epoch_s': [station.image_sensor().data_timestamps() * rpd_scales.MICROS_TO_S],
                       'image_bytes': [station.image_sensor().get_data_channel('image')],
                       'image_codec': [station.image_sensor().get_data_channel('image_codec')]}

        df_im = pd.DataFrame.from_dict(data=dict_for_im)
        return df_im
    else:
        print(f'Station {station.id} has no image data.')


def synchronization_build_station(station: StationRaw) -> pd.DataFrame:

    synchronization = station.timesync_analysis
    dict_for_syn = {'synchronization_epoch_s': [synchronization.get_start_times() * rpd_scales.MICROS_TO_S],
                    'synchronization_latency_ms': [synchronization.get_latencies() * rpd_scales.MICROS_TO_MILLIS],
                    'synchronization_offset_ms': [synchronization.get_offsets() * rpd_scales.MICROS_TO_MILLIS],
                    'synchronization_best_offset_ms': [synchronization.get_best_offset() * rpd_scales.MICROS_TO_MILLIS],
                    'synchronization_offset_delta_ms': [synchronization.get_offsets() * rpd_scales.MICROS_TO_MILLIS -
                                                        synchronization.get_best_offset() * rpd_scales.MICROS_TO_MILLIS]}

    df_syn = pd.DataFrame.from_dict(data=dict_for_syn)
    return df_syn


