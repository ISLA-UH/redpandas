import os
import numpy as np
import pandas as pd
import pymap3d as pm
from redpandas.redpd_scales import EPSILON, NANOS_TO_S, DEGREES_TO_METERS, PRESSURE_SEA_LEVEL_KPA


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


def bounder_data(path_bounder_csv, file_bounder_csv: str, file_bounder_parquet: str):
    """
    Load data from balloon-based Bounder platform
    :param path_bounder_csv:
    :param file_bounder_csv:
    :param bounder_filename:
    :return:
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
    dtime_unix_s = dtime.astype('int64')*NANOS_TO_S

    skyfall_bounder_loc = df.filter(['Lat_deg', 'Lon_deg', 'Alt_m', 'Pres_kPa', 'Temp_C', 'Batt_V'])
    skyfall_bounder_loc.insert(0, 'Epoch_s', dtime_unix_s)
    skyfall_bounder_loc.insert(1, 'Datetime', dtime)

    print(skyfall_bounder_loc['Epoch_s'])
    # Save to parquet
    skyfall_bounder_loc.to_parquet(output_path)


def bounder_model_height_from_pressure(pressure_kPa):
    """
    Returns empirical height in m from input pressure
    :param pressure_kPa:
    :return:
    """
    pressure_ref_kPa = PRESSURE_SEA_LEVEL_KPA
    scaled_pressure = -np.log(pressure_kPa/pressure_ref_kPa)
    # Empirical model constructed from
    # c, stats = np.polynomial.polynomial.polyfit(poly_x, bounder_loc['Alt_m'], 8, full=True)
    c = [1.52981286e+02, 7.39552295e+03, 2.44663285e+03, -3.57402081e+03, 2.02653051e+03,
         -6.26581722e+02, 1.11758211e+02, -1.08674469e+01, 4.46784010e-01]
    elevation_m = np.polynomial.polynomial.polyval(scaled_pressure, c, tensor=False)
    return elevation_m


def compute_t_xyz_uvw(unix_s, lat_deg, lon_deg, alt_m,
                      ref_unix_s, ref_lat_deg, ref_lon_deg, ref_alt_m,
                      geodetic_type: str = 'enu'):
    """
    Compute time and location relative to a reference value; compute speed.
    :param unix_s:
    :param lat_deg:
    :param lon_deg:
    :param alt_m:
    :param ref_unix_s:
    :param ref_lat_deg:
    :param ref_lon_deg:
    :param ref_alt_m:
    :param geodetic_type:
    :return:
    """

    if geodetic_type == 'enu':
        x_m, y_m, z_m = pm.geodetic2enu(lat=lat_deg, lon=lon_deg, h=alt_m,
                                        lat0=ref_lat_deg, lon0=ref_lon_deg, h0=ref_alt_m)
        t_s = (unix_s - ref_unix_s).astype(float)
    elif geodetic_type == 'ned':
        y_m, x_m, z_m = pm.geodetic2ned(lat=lat_deg, lon=lon_deg, h=alt_m,
                                        lat0=ref_lat_deg, lon0=ref_lon_deg, h0=ref_alt_m)
        t_s = (unix_s - ref_unix_s).astype(float)
    else:
        x_m = (lon_deg - ref_lon_deg).astype(float) * DEGREES_TO_METERS
        y_m = (lat_deg - ref_lat_deg).astype(float) * DEGREES_TO_METERS
        z_m = (alt_m - ref_alt_m).astype(float)
        t_s = (unix_s - ref_unix_s).astype(float)

    # Speed in mps. Compute diff, add EPSILON to avoid divide by zero on repeat values
    u_mps = np.gradient(x_m)/(np.gradient(t_s)+EPSILON)
    v_mps = np.gradient(y_m)/(np.gradient(t_s)+EPSILON)
    w_mps = np.gradient(z_m)/(np.gradient(t_s)+EPSILON)

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


def compute_t_r_z_speed(unix_s, lat_deg, lon_deg, alt_m,
                        ref_unix_s, ref_lat_deg, ref_lon_deg, ref_alt_m,
                        geodetic_type: str = 'enu'):
    """
    Compute time and location relative to a reference value; compute speed.
    :param unix_s:
    :param lat_deg:
    :param lon_deg:
    :param alt_m:
    :param ref_unix_s:
    :param ref_lat_deg:
    :param ref_lon_deg:
    :param ref_alt_m:
    :param geodetic_type:
    :return:
    """

    if geodetic_type == 'enu':
        x_m, y_m, z_m = pm.geodetic2enu(lat=lat_deg, lon=lon_deg, h=alt_m,
                                        lat0=ref_lat_deg, lon0=ref_lon_deg, h0=ref_alt_m)
        t_s = (unix_s - ref_unix_s).astype(float)
    elif geodetic_type == 'ned':
        y_m, x_m, z_m = pm.geodetic2ned(lat=lat_deg, lon=lon_deg, h=alt_m,
                                        lat0=ref_lat_deg, lon0=ref_lon_deg, h0=ref_alt_m)
        t_s = (unix_s - ref_unix_s).astype(float)
    else:
        x_m = (lon_deg - ref_lon_deg).astype(float) * DEGREES_TO_METERS
        y_m = (lat_deg - ref_lat_deg).astype(float) * DEGREES_TO_METERS
        z_m = (alt_m - ref_alt_m).astype(float)
        t_s = (unix_s - ref_unix_s).astype(float)

    # Speed in mps. Compute diff, add EPSILON to avoid divide by zero on repeat values
    u_mps = np.gradient(x_m)/(np.gradient(t_s)+EPSILON)
    v_mps = np.gradient(y_m)/(np.gradient(t_s)+EPSILON)
    w_mps = np.gradient(z_m)/(np.gradient(t_s)+EPSILON)

    range_m = np.sqrt(x_m**2 + y_m**2)
    speed_mps = np.ma.sqrt(u_mps**2 + v_mps**2 + w_mps**2)
    time_range_z_speed_s_m = pd.DataFrame(data={'Elapsed_s': t_s,
                                                'Range_m': range_m,
                                                'Z_m': z_m,
                                                'LatLon_speed_mps': speed_mps})
    return time_range_z_speed_s_m