import os
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

import pymap3d as pm
from redpandas.redpd_scales import EPSILON, METERS_TO_KM

import redvox.common.date_time_utils as dt


def redvox_loc(DF_PICKLE_PATH):
    """
    Extract the location, temperature, and DC pressure payload from the microphones
    :param path_redvox:
    :param stations: List of strings [' ', ' ']
    :param redvox_loc_filename:
    :return: save to parquet
    """
    # Check
    if not os.path.exists(DF_PICKLE_PATH):
        print("Input file does not exist, check path:")
        print(DF_PICKLE_PATH)
        exit()

    df = pd.read_parquet(DF_PICKLE_PATH)
    print('Read parquet data frame')

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


# ENU, NED, or ECEF, depending on reference frame
def geodetic_ecef(lat_lon_alt):
    # Earth Centered, Earth Fixed (ECEF) coordinate frame re World Geodetic System (WGS84) ellipsoid
    # lat_lon_alt = latitude and longitude in decimal degrees, altitude above the ellipsoid in m
    # Returns
    # xyz_ned_m = tuple with North, East, Down in meters
    # Default WGS84 ellipsoid
    xyz_ecef_m = pm.geodetic2ecef(*lat_lon_alt, deg=True)
    return xyz_ecef_m


def geodetic_enu(lat_lon_alt, lat_lon_alt_ref):
    # East, North, Up coordinate frame. OK for ground based sensors and ground vehicles.
    # lat_lon_alt = latitude and longitude in decimal degrees, altitude above the ellipsoid in m
    # lat_lon_alt_ref = reference latitude, longitude and altitude, set as xyz = 000
    # Returns
    # xyz_enu_m = tuple with East, North, and Up in meters
    # Default WGS84 ellipsoid
    xyz_enu_m = pm.geodetic2enu(*lat_lon_alt, *lat_lon_alt_ref, deg=True)
    return xyz_enu_m


def geodetic_ned(lat_lon_alt, lat_lon_alt_ref):
    # North, East, Down coordinate frame. OK for marine, aircraft, and spaceships with gravity down
    # lat_lon_alt = latitude and longitude in decimal degrees, altitude above the ellipsoid in m
    # lat_lon_alt_ref = reference latitude, longitude and altitude, set as xyz = 000
    # Returns
    # xyz_ned_m = tuple with North, East, Down in meters
    # Default WGS84 ellipsoid
    xyz_ned_m = pm.geodetic2ned(*lat_lon_alt, *lat_lon_alt_ref, deg=True)
    return xyz_ned_m


# Distance
def pythag_dist(x1, x2, y1, y2) -> float:
    x_dist = x2 - x1
    y_dist = y2 - y1
    dist = np.sqrt(x_dist ** 2 + y_dist ** 2)
    return dist


def euclid_dist(x1, x2, y1, y2, z1, z2) -> float:
    x_dist = x2 - x1
    y_dist = y2 - y1
    z_dist = z2 - z1
    dist = np.sqrt(x_dist ** 2 + y_dist ** 2 + z_dist ** 2)
    return dist



def bounder_data(path_bounder_cvs: str, bounder_filename: str):
    """
    Load data from balloon-based Bounder platform
    :param path_bounder_cvs:
    :param bounder_filename:
    :return:
    """
    # 2020-10-27T13:45:13.132 start time of first RedVox data packet
    # Event-specific start date
    yyyymmdd = "2020-10-27 "
    # Skyfall start at 13:45:00, end at 14:16:00
    # Manual, but can be automated. CSV has been cleaned so can now load all.
    rows = np.arange(5320, 7174)

    df = pd.read_csv(path_bounder_cvs, usecols=[5, 6, 7, 8, 9, 10, 11], skiprows=lambda x: x not in rows,
                     names=['Pres_kPa', 'Temp_C', 'Batt_V', 'Lon_deg', 'Lat_deg', 'Alt_m', 'Time_hhmmss'])
    dtime = pd.to_datetime(yyyymmdd + df['Time_hhmmss'])
    dtime_unix_s = dtime.astype('int64')/1E9

    skyfall_bounder_loc = df.filter(['Lat_deg', 'Lon_deg', 'Alt_m', 'Pres_kPa', 'Temp_C', 'Batt_V'])
    skyfall_bounder_loc.insert(0, 'Epoch_s', dtime_unix_s)
    skyfall_bounder_loc.insert(1, 'Datetime', dtime)

    # Save to parquet
    skyfall_bounder_loc.to_parquet(bounder_filename)


def model_height_from_pressure(pressure_kPa):
    """
    Returns empirical height in m from input pressure
    :param pressure_kPa:
    :return:
    """
    pressure_ref_kPa = 101.325
    scaled_pressure = -np.log(pressure_kPa/pressure_ref_kPa)
    # Empirical model constructed from
    # c, stats = np.polynomial.polynomial.polyfit(poly_x, bounder_loc['Alt_m'], 8, full=True)
    c = [1.52981286e+02, 7.39552295e+03, 2.44663285e+03, -3.57402081e+03, 2.02653051e+03,
         -6.26581722e+02, 1.11758211e+02, -1.08674469e+01, 4.46784010e-01]
    elevation_m = np.polynomial.polynomial.polyval(scaled_pressure, c, tensor=False)
    return elevation_m


def compute_bounder_t_xyz_uvw(unix_s, lat_deg, lon_deg, alt_m):
    """
    Assuming no movement at the end
    :param unix_micros:
    :param lat_deg:
    :param lon_deg:
    :param alt_m:
    :return:
    """
    # Convert to XY for computation of "speed" from derivative
    x_m = (lon_deg - lon_deg.iloc[-1]).astype(float)*111000
    y_m = (lat_deg - lat_deg.iloc[-1]).astype(float)*111000
    z_m = (alt_m - alt_m.iloc[-1]).astype(float)
    t_s = (unix_s - unix_s.iloc[-1]).astype(float)

    # Speed in mps. Compute diff and add zero at terminus (at rest)
    u_mps = np.append(np.diff(x_m)/(np.diff(t_s)+EPSILON), 0)
    v_mps = np.append(np.diff(y_m)/(np.diff(t_s)+EPSILON), 0)
    w_mps = np.append(np.diff(z_m)/(np.diff(t_s)+EPSILON), 0)

    speed_mps = np.sqrt(u_mps**2 + v_mps**2 + w_mps**2)

    txyzuvw = pd.DataFrame(data={'T_s': t_s,
                                 'X_m': x_m,
                                 'Y_m': y_m,
                                 'Z_m': z_m,
                                 'U_mps': u_mps,
                                 'V_mps': v_mps,
                                 'W_mps': w_mps,
                                 'Speed_mps': speed_mps})
    return txyzuvw


def compute_phone_t_xyz_uvw(unix_s, lat_deg, lon_deg, alt_m):
    """
    Assuming no movement at the end
    :param unix_micros:
    :param lat_deg:
    :param lon_deg:
    :param alt_m:
    :return:
    """
    # Convert to XY for computation of "speed" from derivative
    x_m = (lon_deg - lon_deg[-1]).astype(float)*111000
    y_m = (lat_deg - lat_deg[-1]).astype(float)*111000
    z_m = (alt_m - alt_m[-1]).astype(float)
    t_s = (unix_s - unix_s[-1]).astype(float)

    # Speed in mps. Compute diff and add zero at terminus (at rest)
    u_mps = np.append(np.diff(x_m)/(np.diff(t_s)+EPSILON), 0)
    v_mps = np.append(np.diff(y_m)/(np.diff(t_s)+EPSILON), 0)
    w_mps = np.append(np.diff(z_m)/(np.diff(t_s)+EPSILON), 0)

    speed_mps = np.sqrt(u_mps**2 + v_mps**2 + w_mps**2)

    txyzuvw = pd.DataFrame(data={'T_s': t_s,
                                 'X_m': x_m,
                                 'Y_m': y_m,
                                 'Z_m': z_m,
                                 'U_mps': u_mps,
                                 'V_mps': v_mps,
                                 'W_mps': w_mps,
                                 'Speed_mps': speed_mps})
    return txyzuvw