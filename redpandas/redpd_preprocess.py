from enum import Enum
from typing import Tuple

import numpy as np
from scipy import signal

from redvox.common import date_time_utils as dt
from redvox.common.station import Station
from redvox.common.station_raw import StationRaw
import pipeline_m.redpd_iterator as rdp_iter
import pipeline_m.redpd_scales as rpd_scales
import pipeline_m.redpd_build_station as rpd_build_sta

import matplotlib.pyplot as plt

import obspy.signal.filter
import pandas as pd

"""
This module contains general utilities that can work with values containing nans.
"""


# Define classes
class NormType(Enum):
    """
    Enumeration of normalization types.
    """
    MAX: str = "max"
    L1: str = "l1"
    L2: str = "l2"
    OTHER: str = "other"


# Auxiliary modules for building stations
def datetime_now_epoch_s() -> float:
    """
    Returns the invocation Unix time in seconds
    :return: The current epoch timestamp as seconds since the epoch UTC
    """
    return dt.datetime_to_epoch_seconds_utc(dt.now())


def datetime_now_epoch_micros() -> float:
    """
    Returns the invocation Unix time in microseconds
    :return: The current epoch timestamp as microseconds since the epoch UTC
    """
    return dt.datetime_to_epoch_microseconds_utc(dt.now())


def normalize(sig: np.ndarray, scaling: float = 1., norm_type: NormType = NormType.MAX) -> np.ndarray:
    """
    Scale a 1D time series
    :param sig: time series signature
    :param scaling: scaling parameter, division
    :param norm_type: {'max', l1, l2}, optional
    :return: The scaled series
    """
    if norm_type == NormType.MAX:
        return sig / np.nanmax(np.abs(sig))
    elif norm_type == NormType.L1:
        return sig / np.nansum(sig)
    elif norm_type == NormType.L2:
        return sig / np.sqrt(np.nansum(sig * sig))
    else:  # Must be NormType.Other
        return sig / scaling


def demean_nan(sig: np.ndarray) -> np.ndarray:
    """
    Detrend and normalize a 1D time series
    :param sig: time series with (possibly) non-zero mean
    :return: Detrended and normalized time series
    """""
    return np.nan_to_num(sig - np.nanmean(sig))


def detrend_nan(sig: np.ndarray) -> np.ndarray:
    """
    Detrend and normalize a 1D time series
    :param sig: time series with (possibly) non-zero mean
    :return: Detrended and normalized time series
    """""
    return signal.detrend(demean_nan(sig))


def demean_nan_norm(sig: np.ndarray, scaling: float = 1., norm_type: NormType = NormType.MAX) -> np.ndarray:
    """
    Detrend and normalize a 1D time series
    :param sig: time series with (possibly) non-zero mean
    :param scaling: scaling parameter, division
    :param norm_type: {'max', l1, l2}, overrides scikit default of 'l2' by 'max'
    :return: The detrended and denormalized series.
    """""
    sig_detrend = demean_nan(sig)
    return normalize(sig_detrend, scaling=scaling, norm_type=norm_type)


def demean_nan_matrix(sig: np.ndarray) -> np.ndarray:
    """
    Detrend and normalize a matrix of time series
    :param sig: time series with (possibly) non-zero mean
    :return: The detrended and normalized signature
    """""
    return np.nan_to_num(np.subtract(sig.transpose(), np.nanmean(sig, axis=1))).transpose()


def taper_tukey(sig_or_time: np.ndarray, fraction_cosine: float) -> np.ndarray:
    """
    Constructs a symmetric Tukey window with the same dimensions as a time or signal numpy array.
    fraction_cosine = 0 is a rectangular window, 1 is a Hann window
    :param sig_or_time: input signal or time
    :param fraction_cosine: fraction of the window inside the cosine tapered window, shared between the head and tail
    :return: tukey taper window amplitude
    """
    number_points = np.size(sig_or_time)
    amplitude = signal.windows.tukey(M=number_points, alpha=fraction_cosine, sym=True)
    return amplitude


def pad_reflection_symmetric(sig_wf):
    """
    Apply reflection transformation
    :param sig_wf:
    :return:
    """
    number_points_to_flip_per_edge = int(len(sig_wf)//2)
    wf_folded = np.pad(np.copy(sig_wf),
                       (number_points_to_flip_per_edge, number_points_to_flip_per_edge),
                       'reflect')
    wf_folded *= taper_tukey(wf_folded, fraction_cosine=0.5)
    return wf_folded, number_points_to_flip_per_edge


def filter_reflection_highpass(sig_wf, filter_cutoff_hz, sample_rate_hz):
    """
    Apply filter
    :param sig_wf:
    :param filter_cutoff_hz:
    :param sample_rate_hz:
    :param filter_type:
    :return:
    """
    wf_folded, number_points_to_flip_per_edge = pad_reflection_symmetric(sig_wf)

    sig_folded_filtered = highpass_obspy(sensor_wf=wf_folded,
                                         frequency_low_hz=filter_cutoff_hz,
                                         sample_rate_hz=sample_rate_hz)
    sig_filtered = sig_folded_filtered[number_points_to_flip_per_edge:-number_points_to_flip_per_edge]

    return sig_filtered


def height_asl_from_pressure_below10km(bar_waveform: np.ndarray) -> np.ndarray:
    """
    Simple model for troposphere
    :param bar_waveform: barometric pressure in kPa
    :return: height ASL in m
    """
    mg_rt = 0.00012  # Molar mass of air x gravity / (gas constant x standard temperature)
    elevation_m = -np.log(bar_waveform/rpd_scales.Slice.PREF_KPA)/mg_rt
    return elevation_m


def model_height_from_pressure_skyfall(pressure_kPa):
    """
    Returns empirical height in m from input pressure
    :param pressure_kPa: barometric pressure in kPa
    :return: height in m
    """
    pressure_ref_kPa = 101.325
    scaled_pressure = -np.log(pressure_kPa/pressure_ref_kPa)
    # Empirical model constructed from
    # c, stats = np.polynomial.polynomial.polyfit(poly_x, bounder_loc['Alt_m'], 8, full=True)
    c = [1.52981286e+02, 7.39552295e+03, 2.44663285e+03, -3.57402081e+03, 2.02653051e+03,
         -6.26581722e+02, 1.11758211e+02, -1.08674469e+01, 4.46784010e-01]
    elevation_m = np.polynomial.polynomial.polyval(scaled_pressure, c, tensor=False)
    return elevation_m


def rc_high_pass_signal(sig, sample_rate, highpass_cutoff):
    highpass_signal = np.array([[high]
                                for high
                                in rdp_iter.rc_iterator_high_pass(sig, sample_rate, highpass_cutoff)])
    return highpass_signal


# "Traditional" solution, up to Nyquist
def bandpass_butter_uneven(sensor_wf, filter_order, frequency_cut_low_hz, sample_rate_hz):
    # Frequencies are scaled by Nyquist, with 1 = Nyquist
    # filter_order = 4,
    nyquist = 0.5 * sample_rate_hz
    edge_low = frequency_cut_low_hz / nyquist
    edge_high = 0.5
    [b, a] = signal.butter(N=filter_order, Wn=[edge_low, edge_high], btype='bandpass')
    sensor_bandpass = signal.filtfilt(b, a, np.copy(sensor_wf))
    return sensor_bandpass


def highpass_obspy(sensor_wf, frequency_low_hz, sample_rate_hz, filter_order=4):
    sensor_highpass = obspy.signal.filter.highpass(np.copy(sensor_wf),
                                                   frequency_low_hz,
                                                   sample_rate_hz, corners=filter_order,
                                                   zerophase=True)
    return sensor_highpass


def xcorr_uneven(sig_x: np.ndarray, sig_ref: np.ndarray):
    """
    Variation of cross-correlation function cross_stas.xcorr_all for unevenly sampled data
    with identical sampling and duration.
    :param sig_x: processed signal
    :param sig_ref: reference signal
    :param xcorr_mode: 'same', 'full', and 'valid'. Valid is the default setting for corrcoeff
    :return: cross-correlation metrics
    """
    nx = len(sig_x)
    nref = len(sig_ref)
    if nx > nref:
        print('Vectors must have equal sampling and lengths')
    elif nx < nref:
        print('Vectors must have equal sampling and lengths')
    elif nx == nref:
        """Cross correlation is centered in the middle of the record and has length NX"""
        # Fastest, o(NX) and can use FFT solution
        if nx % 2 == 0:
            xcorr_indexes = np.arange(-int(nx/2), int(nx/2))
        else:
            xcorr_indexes = np.arange(-int(nx/2), int(nx/2)+1)

        xcorr = signal.correlate(sig_ref, sig_x, mode='same')
        # Normalize
        xcorr /= nx * sig_x.std() * sig_ref.std()
        xcorr_offset_index = np.argmax(np.abs(xcorr))
        xcorr_offset_samples = xcorr_indexes[xcorr_offset_index]
        xcorr_peak = xcorr[xcorr_offset_index]

    else:
        print('One of the waveforms is broken')
        return

    return xcorr, xcorr_indexes, xcorr_peak, xcorr_offset_index, xcorr_offset_samples


def highpass_from_diff(sensor_waveform: np.ndarray,
                       sensor_epoch_s: np.ndarray,
                       sample_rate: int or float,
                       highpass_type: str = 'obspy',
                       frequency_filter_low: float = 1./rpd_scales.Slice.T100S,
                       filter_order: int = 4) -> (np.ndarray, float):
    """
    Preprocess barometer data:
    - remove nans and DC offset by getting the differential pressure in kPa
    - apply highpass filter at 100 second periods
    - reconstruct Pressure in kPa from differential pressure: P(i) = dP(i) + P(i-1)
    :param sensor_waveform:
    :param sensor_epoch_s:
    :param sample_rate:
    :param highpass_type: 'obspy', 'butter', 'rc'
    :param frequency_filter_low: 100s default
    :param filter_order: Default is 4.
    :zero phase filters are acausal
    :return:
    """

    # Apply diff to remove DC offset; difference of nans is a nan
    # Replace nans with zeros, otherwise most things don't run
    # Using gradient instead of diff seems to fix off by zero issue!
    sensor_waveform_grad_dm = demean_nan(np.gradient(sensor_waveform))

    # Override default high pass at 100 seconds if signal is too short
    # May be able to zero pad ... with ringing. Or fold as needed.
    if sensor_epoch_s[-1] - sensor_epoch_s[0] < 2/frequency_filter_low:
        frequency_filter_low = 2/(sensor_epoch_s[-1] - sensor_epoch_s[0])
        print('Default 100s highpass override. New highpass period = ', 1/frequency_filter_low)

    # Fold edges of wf
    sensor_waveform_fold, number_points_folded = pad_reflection_symmetric(sensor_waveform_grad_dm)

    if highpass_type == "obspy":
        # Zero phase, acausal
        sensor_waveform_dp_filtered = \
            obspy.signal.filter.highpass(corners=filter_order,
                                         data=np.copy(sensor_waveform_fold),
                                         freq=frequency_filter_low,
                                         df=sample_rate,
                                         zerophase=True)

    elif highpass_type == "butter":
        [b, a] = signal.butter(N=filter_order,
                               Wn=frequency_filter_low,
                               fs=sample_rate,
                               btype='highpass',
                               output='ba')
        # Zero phase, acausal
        sensor_waveform_dp_filtered = signal.filtfilt(b, a, sensor_waveform_fold)

    elif highpass_type == "rc":
        # RC is slow and not zero-phase, does not need a taper to work (but it doesn't hurt)
        sensor_waveform_dp_filtered = \
            rc_high_pass_signal(sig=np.copy(sensor_waveform_fold),
                                sample_rate=sample_rate,
                                highpass_cutoff=frequency_filter_low)

    else:
        raise Exception("No filter selected. Type 'obspy', 'butter', or 'rc'.")

    # Cut fold edges of wf
    sensor_waveform_dp_filtered = sensor_waveform_dp_filtered[number_points_folded:-number_points_folded]

    # Reconstruct Function dP: P(0), P(i) = dP(i) + P(i-1)
    sensor_waveform_reconstruct = np.zeros((len(sensor_waveform_dp_filtered)))
    # Initialize
    sensor_waveform_reconstruct[0] = sensor_waveform_dp_filtered[0]

    for i in range(1, len(sensor_waveform_dp_filtered) - 1):
        sensor_waveform_reconstruct[i] = sensor_waveform_dp_filtered[i] + sensor_waveform_reconstruct[i-1]

    return sensor_waveform_reconstruct, frequency_filter_low


# TODO: eventually eliminated them, sub them in for pandas version
# Modules for building sensor stations
# def audio_wf_time_build_station(station: StationRaw,
#                                 mean_type: str = "simple",
#                                 raw: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
#     """
#     Builds mic waveform and times
#     :param station: the station with data
#     :param mean_type: under development
#     :param raw: if false (default), boolean or nan mean removed
#     :return:
#     """
#     mic_sample_rate_hz = station.audio_sensor().sample_rate_hz
#     # mic_wf_raw = station.audio_sensor().get_data_channel("microphone")
#     mic_wf_raw = station.audio_sensor().samples()
#     mic_epoch_s = station.audio_sensor().data_timestamps() * rpd_scales.MICROS_TO_S
#
#     if raw:
#         mic_wf = np.array(mic_wf_raw)
#     else:
#         if mean_type == "simple":
#             # Simple demean and replace nans with zeros. OK for mic, not OK for all other DC-biased sensors
#             mic_wf = demean_nan(mic_wf_raw)
#         else:
#             # Remove linear trend
#             mic_wf = detrend_nan(mic_wf_raw)
#
#     return mic_wf, mic_epoch_s, mic_sample_rate_hz
#
#
# def barometer_wf_time_build_station(station: StationRaw,
#                                     raw: bool = True,
#                                     highpass_type: str = 'obspy',
#                                     frequency_filter_low: float = 1./rpd_scales.Slice.T100S,
#                                     filter_order: int = 4) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
#     """
#     Obtains barometer data from a station.
#     :param station: RDVX station
#     :param raw: if false (default), highpass filter passed
#     :param frequency_filter_low: Default is 100s
#     :param highpass_type: 'obspy', 'butter', 'rc'
#     :param filter_order: Default is 4
#
#     :return: The barometer data, the timestamps, the estimated sample rate, and the indexes of the nans.
#     The nan indexes should be preserved throughout the computation and used in all the plots.
#     """
#     barometer_sample_rate_hz = station.barometer_sensor().sample_rate_hz
#     # barometer_raw = station.barometer_sensor().get_data_channel("pressure")
#     barometer_raw = station.barometer_sensor().samples()
#     barometer_epoch_s = station.barometer_sensor().data_timestamps() * rpd_scales.MICROS_TO_S
#     barometer_nans = np.argwhere(np.isnan(barometer_raw))
#     if raw:
#         barometer_wf = np.array(barometer_raw)
#     else:
#         barometer_wf, _ = rpd_build_sta.highpass_from_diff_for_build_station(sensor_waveform=barometer_raw,
#                                              sensor_epoch_s=barometer_epoch_s,
#                                              sample_rate=barometer_sample_rate_hz,
#                                              highpass_type=highpass_type,
#                                              frequency_filter_low=frequency_filter_low,
#                                              filter_order=filter_order)
#
#     return barometer_wf, barometer_epoch_s, barometer_sample_rate_hz, barometer_nans
#
#
# def accelerometer_wf_time_build_station(station: StationRaw,
#                                         mean_type: str = "simple",
#                                         raw: bool = False):
#     """
#     Obtains accelerometer data from a station
#     :param station: RDVX station
#     :param mean_type: under development
#     :param raw: if false (default), boolean or nan mean removed
#     :return: the accelerometer data, the timestamps, and the estimated sample rate
#     """
#     accelerometer_sample_rate_hz = station.accelerometer_sensor().sample_rate_hz
#     # accelerometer_x_raw = station.accelerometer_sensor().get_data_channel('accelerometer_x')
#     # accelerometer_y_raw = station.accelerometer_sensor().get_data_channel('accelerometer_y')
#     # accelerometer_z_raw = station.accelerometer_sensor().get_data_channel('accelerometer_z')
#     acceleration_wf_raw = station.accelerometer_sensor().samples()
#     accelerometer_epoch_s = station.accelerometer_sensor().data_timestamps() * rpd_scales.MICROS_TO_S
#
#     if raw:
#         accelerometer_wf = acceleration_wf_raw
#     else:
#         if mean_type == "simple":
#             # Demeans and replaces nans with zeros for 3C sensors
#             acceleration_wf_list = []
#             for index_direction in enumerate(acceleration_wf_raw):
#                 acc_detrend = np.nan_to_num(np.subtract(acceleration_wf_raw[index_direction],
#                                                         np.nanmean(acceleration_wf_raw[index_direction])))
#                 acceleration_wf_list.append(acc_detrend)
#
#             accelerometer_wf = np.array(acceleration_wf_list)
#
#         else:
#             # Placeholder for diff solution with nans
#             pass
#
#     return accelerometer_wf, accelerometer_epoch_s, accelerometer_sample_rate_hz
#
#
# def gyroscope_wf_time_build_station(station: StationRaw,
#                                     mean_type: str = "simple",
#                                     raw: bool = False):
#     """
#     Obtains gyroscope data from a station
#     :param station: RDVX station
#     :param mean_type: under development
#     :param raw: if false (default), boolean or nan mean removed
#     :return: the gyroscope data, the timestamps, and the estimated sample rate
#     """
#     gyroscope_sample_rate_hz = station.gyroscope_sensor().sample_rate_hz
#     # gyroscope_x_raw = station.gyroscope_sensor().get_data_channel('gyroscope_x')
#     # gyroscope_y_raw = station.gyroscope_sensor().get_data_channel('gyroscope_y')
#     # gyroscope_z_raw = station.gyroscope_sensor().get_data_channel('gyroscope_z')
#     gyroscope_wf_raw = station.gyroscope_sensor().samples()
#     gyroscope_epoch_s = station.gyroscope_sensor().data_timestamps() * rpd_scales.MICROS_TO_S
#
#     if raw:
#         gyroscope_wf = gyroscope_wf_raw
#     else:
#         if mean_type == "simple":
#             # Demeans and replaces nans with zeros for 3C sensors
#             gyro_wf_list = []
#             for index_direction in enumerate(gyroscope_wf_raw):
#                 acc_detrend = np.nan_to_num(np.subtract(gyroscope_wf_raw[index_direction],
#                                                         np.nanmean(gyroscope_wf_raw[index_direction])))
#                 gyro_wf_list.append(acc_detrend)
#
#             gyroscope_wf = np.array(gyro_wf_list)
#
#         else:
#             # Placeholder for diff solution with nans
#             pass
#
#     return gyroscope_wf, gyroscope_epoch_s, gyroscope_sample_rate_hz
#
#
# def magnetometer_wf_time_build_station(station: StationRaw,
#                                        mean_type: str = "simple",
#                                        raw: bool = False):
#     """
#     Obtains magnetometer data from a station
#     :param station: RDVX station
#     :param mean_type: under development
#     :param raw: if false (default), boolean or nan mean removed
#     :return: the magnetometer data, the timestamps, and the estimated sample rate
#     """
#     magnetometer_sample_rate_hz = station.magnetometer_sensor().sample_rate_hz
#     # magnetometer_x_raw = station.magnetometer_sensor().get_data_channel('magnetometer_x')
#     # magnetometer_y_raw = station.magnetometer_sensor().get_data_channel('magnetometer_y')
#     # magnetometer_z_raw = station.magnetometer_sensor().get_data_channel('magnetometer_z')
#     magnetometer_wf_raw = station.magnetometer_sensor().samples()
#     magnetometer_epoch_s = station.magnetometer_sensor().data_timestamps() * rpd_scales.MICROS_TO_S
#
#     if raw:
#         magnetometer_wf = magnetometer_wf_raw
#
#     else:
#         if mean_type == "simple":
#             # Demeans and replaces nans with zeros for 3C sensors
#             mag_wf_list = []
#             for index_direction in enumerate(magnetometer_wf_raw):
#                 acc_detrend = np.nan_to_num(np.subtract(magnetometer_wf_raw[index_direction],
#                                                         np.nanmean(magnetometer_wf_raw[index_direction])))
#                 mag_wf_list.append(acc_detrend)
#
#             magnetometer_wf = np.array(mag_wf_list)
#
#         else:
#             # Placeholder for diff solution with nans
#             pass
#
#     return magnetometer_wf, magnetometer_epoch_s, magnetometer_sample_rate_hz
#
# #
# # # location build station?
# # def location_build_station(station: StationRaw):
# #     """
# #     Obtains location data from station
# #     :param station: RDVX station
# #     :return: timestamps, sample rate in Hz, latitude, longitude, altitude, bearing, speed, horizontal accuracy,
# #     vertical accuracy, bearing accuracy, speed accuracy
# #     """
# #
# #     location_sample_rate_hz = station.location_sensor().sample_rate_hz
# #     location_latitude = station.location_sensor().get_data_channel("latitude")
# #     location_longitude = station.location_sensor().get_data_channel("longitude")
# #     location_altitude = station.location_sensor().get_data_channel("altitude")
# #     location_bearing = station.location_sensor().get_data_channel("bearing")
# #     location_speed = station.location_sensor().get_data_channel("speed")
# #     location_horizontal_accuracy = station.location_sensor().get_data_channel("horizontal_accuracy")
# #     location_vertical_accuracy = station.location_sensor().get_data_channel("vertical_accuracy")
# #     location_bearing_accuracy = station.location_sensor().get_data_channel("bearing_accuracy")
# #     location_speed_accuracy = station.location_sensor().get_data_channel("speed_accuracy")
# #     location_provider = station.location_sensor().get_data_channel("location_provider")
# #     location_epoch_s = station.location_sensor().data_timestamps() * rpd_scales.MICROS_TO_S
# #
# #     return location_latitude, location_longitude, location_altitude, location_bearing, location_speed, \
# #            location_horizontal_accuracy, location_vertical_accuracy, location_bearing_accuracy, \
# #            location_speed_accuracy, location_provider, location_epoch_s, location_sample_rate_hz
# #
# #
# # # soh build station?
# # def state_of_health_build_station(station: StationRaw):
# #     """
# #
# #     :param station: RDVX station
# #     :return:
# #     """
# #
# #     health_sample_rate_hz = station.health_sensor().sample_rate_hz
# #     battery_charge_remaining_per = station.health_sensor().get_data_channel('battery_charge_remaining')
# #     battery_current_strength_mA = station.health_sensor().get_data_channel('battery_current_strength')
# #     internal_temp_deg_C = station.health_sensor().get_data_channel('internal_temp_c')
# #     network_type = station.health_sensor().get_data_channel('network_type')
# #     network_strength_dB = station.health_sensor().get_data_channel('network_strength')
# #     power_state = station.health_sensor().get_data_channel('power_state')
# #     available_ram_byte = station.health_sensor().get_data_channel('avail_ram')
# #     available_disk_byte = station.health_sensor().get_data_channel('avail_disk')
# #     cell_service_state = station.health_sensor().get_data_channel('cell_service')
# #     health_epoch_s = station.health_sensor().data_timestamps() * rpd_scales.MICROS_TO_S
# #
# #     return health_sample_rate_hz, battery_charge_remaining_per, battery_current_strength_mA, internal_temp_deg_C, \
# #            network_type, network_strength_dB, power_state, available_ram_byte, available_disk_byte, \
# #            cell_service_state, health_epoch_s
#
# # synchornization build station?
