"""
Functions to extract and process geospatial data.
"""

import os
import numpy as np
import pandas as pd
import pymap3d as pm
from typing import Any

from redpandas.redpd_scales import EPSILON, NANOS_TO_S, DEGREES_TO_METERS, PRESSURE_SEA_LEVEL_KPA


def redvox_loc(df_pqt_path: str) -> pd.DataFrame:
    """
    Extract the location, temperature, and DC pressure payload from the microphones

    :param df_pqt_path: path/to/parquet file with data stored in a pd.DataFrame
    :return: pd. DataFrame with columns {'station_id', 'location_epoch_s', 'location_latitude', 'location_longitude',
    'location_altitude', 'location_speed', 'location_horizontal_accuracy', 'barometer_epoch_s', 'barometer_wf_raw'}
    """
    # Check
    if not os.path.exists(df_pqt_path):
        raise FileNotFoundError(f"Input file does not exist, check path: {df_pqt_path}")

    df = pd.read_parquet(df_pqt_path)
    print('Read parquet with pandas DataFrame')

    # Extract selected fields
    loc_fields = ['station_id',
                  'location_epoch_s',
                  'location_latitude',
                  'location_longitude',
                  'location_altitude',
                  'location_speed',
                  'location_horizontal_accuracy',
                  'barometer_epoch_s',
                  'barometer_wf_raw']
    df_loc = df[loc_fields]

    return df_loc


def bounder_data(path_bounder_csv: str, file_bounder_csv: str, file_bounder_parquet: str) -> None:
    """
    Load data from balloon-based Bounder platform

    :param path_bounder_csv: path/to/bounder csv and parquet files
    :param file_bounder_csv: name bounder csv file
    :param file_bounder_parquet: name bounder parquet file
    :return: save as parquet
    """
    # Event-specific start date and curated file
    # Bounder Skyfall starts at 13:45:00, end at 14:16:00
    yyyymmdd = "2020-10-27 "
    rows = np.arange(5320, 7174)

    input_path = os.path.join(path_bounder_csv, file_bounder_csv)
    print('Input', input_path)
    output_path = os.path.join(path_bounder_csv, file_bounder_parquet)

    df = pd.read_csv(input_path, usecols=[5, 6, 7, 8, 9, 10, 11], skiprows=lambda x: x not in rows,
                     names=['Pres_kPa', 'Temp_C', 'Batt_V', 'Lon_deg', 'Lat_deg', 'Alt_m', 'Time_hhmmss'])
    dtime = pd.to_datetime(yyyymmdd + df['Time_hhmmss'], origin='unix')

    # Convert datetime to unix nanoseconds, then to seconds
    # dtime_unix_s = dtime.astype('int64')*NANOS_TO_S  # Deprecated
    dtime_unix_s = dtime.view('int64')*NANOS_TO_S  # Python 3.9

    skyfall_bounder_loc = df.filter(['Lat_deg', 'Lon_deg', 'Alt_m', 'Pres_kPa', 'Temp_C', 'Batt_V'])
    skyfall_bounder_loc.insert(0, 'Epoch_s', dtime_unix_s)
    skyfall_bounder_loc.insert(1, 'Datetime', dtime)

    print(skyfall_bounder_loc['Epoch_s'])
    # Save to parquet
    skyfall_bounder_loc.to_parquet(output_path)


def bounder_model_height_from_pressure(pressure_kpa: np.ndarray) -> np.ndarray:
    """
    Returns empirical height in m from input pressure

    :param pressure_kpa: Atmospheric pressure in kPa
    :return: Height in m above WGS84 Geoid
    """
    # Empirical model constructed from
    c = [1.52981286e+02, 7.39552295e+03, 2.44663285e+03, -3.57402081e+03, 2.02653051e+03,
         -6.26581722e+02, 1.11758211e+02, -1.08674469e+01, 4.46784010e-01]
    return np.polynomial.polynomial.polyval(-np.log(pressure_kpa / PRESSURE_SEA_LEVEL_KPA), c, tensor=False)


def compute_t_xyz_uvw(unix_s: Any,
                      lat_deg: Any,
                      lon_deg: Any,
                      alt_m: Any,
                      ref_unix_s: Any,
                      ref_lat_deg: Any,
                      ref_lon_deg: Any,
                      ref_alt_m: Any,
                      geodetic_type: str = 'enu') -> pd.DataFrame:
    """
    Compute time and location relative to a reference value; compute speed.

    :param unix_s: target timestamp
    :param lat_deg: target geodetic latitude
    :param lon_deg: target geodetic longitude
    :param alt_m: target altitude above ellipsoid (meters)
    :param ref_unix_s: observer timestamp
    :param ref_lat_deg: observer geodetic latitude
    :param ref_lon_deg: observer geodetic longitude
    :param ref_alt_m: observer altitude above geodetic ellipsoid (meters)
    :param geodetic_type: 'enu' or 'ned'
    :return: pandas DataFrame with columns: {'T_s', 'X_m', 'Y_m', 'Z_m', 'U_mps', 'V_mps', 'W_mps', 'Speed_mps'}
    """
    if geodetic_type == 'enu':
        x_m, y_m, z_m = pm.geodetic2enu(lat=lat_deg, lon=lon_deg, h=alt_m,
                                        lat0=ref_lat_deg, lon0=ref_lon_deg, h0=ref_alt_m)
    elif geodetic_type == 'ned':
        y_m, x_m, z_m = pm.geodetic2ned(lat=lat_deg, lon=lon_deg, h=alt_m,
                                        lat0=ref_lat_deg, lon0=ref_lon_deg, h0=ref_alt_m)
    else:
        x_m = (lon_deg - ref_lon_deg).astype(float) * DEGREES_TO_METERS
        y_m = (lat_deg - ref_lat_deg).astype(float) * DEGREES_TO_METERS
        z_m = (alt_m - ref_alt_m).astype(float)
    t_s = (unix_s - ref_unix_s).astype(float)

    # Speed in mps. Compute diff, add EPSILON to avoid divide by zero on repeat values
    u_mps = np.gradient(x_m) / (np.gradient(t_s)+EPSILON)
    v_mps = np.gradient(y_m) / (np.gradient(t_s)+EPSILON)
    w_mps = np.gradient(z_m) / (np.gradient(t_s)+EPSILON)
    speed_mps = np.sqrt(u_mps**2 + v_mps**2 + w_mps**2)

    t_xyzuvw_s_m = pd.DataFrame(data={'T_s': t_s,
                                      'X_m': x_m,
                                      'Y_m': y_m,
                                      'Z_m': z_m,
                                      'U_mps': u_mps,
                                      'V_mps': v_mps,
                                      'W_mps': w_mps,
                                      'Speed_mps': speed_mps})
    return t_xyzuvw_s_m


def compute_t_r_z_speed(unix_s: Any,
                        lat_deg: Any,
                        lon_deg: Any,
                        alt_m: Any,
                        ref_unix_s: Any,
                        ref_lat_deg: Any,
                        ref_lon_deg: Any,
                        ref_alt_m: Any,
                        geodetic_type: str = 'enu') -> pd.DataFrame:
    """
    Compute time and location relative to a reference value; compute speed.

    :param unix_s: target timestamp
    :param lat_deg: target geodetic latitude
    :param lon_deg: target geodetic longitude
    :param alt_m: target altitude above ellipsoid (meters)
    :param ref_unix_s: observer timestamp
    :param ref_lat_deg: observer geodetic latitude
    :param ref_lon_deg: observer geodetic longitude
    :param ref_alt_m: observer altitude above geodetic ellipsoid (meters)
    :param geodetic_type: 'enu' or 'ned'
    :return: pandas DataFrame with columns: {'Elapsed_s', 'Range_m', 'Z_m', 'LatLon_speed_mps'}
    """
    if geodetic_type == 'enu':
        x_m, y_m, z_m = pm.geodetic2enu(lat=lat_deg, lon=lon_deg, h=alt_m,
                                        lat0=ref_lat_deg, lon0=ref_lon_deg, h0=ref_alt_m)
    elif geodetic_type == 'ned':
        y_m, x_m, z_m = pm.geodetic2ned(lat=lat_deg, lon=lon_deg, h=alt_m,
                                        lat0=ref_lat_deg, lon0=ref_lon_deg, h0=ref_alt_m)
    else:
        x_m = (lon_deg - ref_lon_deg).astype(float) * DEGREES_TO_METERS
        y_m = (lat_deg - ref_lat_deg).astype(float) * DEGREES_TO_METERS
        z_m = (alt_m - ref_alt_m).astype(float)
    t_s = (unix_s - ref_unix_s).astype(float)

    # Speed in mps. Compute diff, add EPSILON to avoid divide by zero on repeat values
    u_mps = np.gradient(x_m) / (np.gradient(t_s)+EPSILON)
    v_mps = np.gradient(y_m) / (np.gradient(t_s)+EPSILON)
    w_mps = np.gradient(z_m) / (np.gradient(t_s)+EPSILON)
    range_m = np.sqrt(x_m**2 + y_m**2)
    speed_mps = np.ma.sqrt(u_mps**2 + v_mps**2 + w_mps**2)
    time_range_z_speed_s_m = pd.DataFrame(data={'Elapsed_s': t_s,
                                                'Range_m': range_m,
                                                'Z_m': z_m,
                                                'LatLon_speed_mps': speed_mps})
    return time_range_z_speed_s_m
