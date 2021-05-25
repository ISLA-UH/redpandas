import numpy as np
import pandas as pd
from typing import List
import redvox.common.date_time_utils as dt
from redvox.common.data_window import DataWindow


def redvox_loc(path_redvox: str, stations: List[str], redvox_loc_filename: str):
    """
    Extract the location, temperature, and DC pressure payload from the microphones
    :param path_redvox:
    :param stations: List of strings [' ', ' ']
    :param redvox_loc_filename:
    :return: save to parquet
    """
    # If Need to rerun with raw data: Load RedVox data
    print('Loading rdvx data')
    data_window = DataWindow(input_dir=path_redvox,
                             station_ids=stations,
                             apply_correction=True,
                             structured_layout=True)
    print('Running rdvx data')

    # Extract RedVox location samples, location epoch micros
    locations, phone_loc_times = extract_location(data_window, stations[0])
    phone_loc_times_dt = [dt.datetime_from_epoch_microseconds_utc(time) for time in phone_loc_times]

    # Some of the fields are empty or wrong:
    # TODO: Check on 'horizontal_accuracy', 'location_provider',
    #  'bearing', 'vertical_accuracy', 'speed_accuracy', 'bearing_accuracy'
    skyfall_phone_loc = pd.DataFrame(data={'Unixtime_micros': phone_loc_times,
                                           'Datetime': phone_loc_times_dt,
                                           'Lat_deg': locations[0],
                                           'Lon_deg': locations[1],
                                           'Alt_m': locations[2],
                                           'Speed_mps': locations[3]})
    # Save to parquet
    skyfall_phone_loc.to_parquet(redvox_loc_filename)
    print('Saved Redvox data to ', redvox_loc_filename)


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
    dtime_unix_micros = dtime.astype('int64')/1E3

    skyfall_bounder_loc = df.filter(['Lat_deg', 'Lon_deg', 'Alt_m', 'Pres_kPa', 'Temp_C', 'Batt_V'])
    skyfall_bounder_loc.insert(0, 'Unixtime_micros', dtime_unix_micros)
    skyfall_bounder_loc.insert(1, 'Datetime', dtime)

    # Save to parquet
    skyfall_bounder_loc.to_parquet(bounder_filename)


def extract_location(data_window, phone_id):
    station = data_window.stations.get_station(phone_id)
    print("RedVox location fields:", station.location_sensor().data_fields())
    print("Mean RedVox location sample rate:")
    print(np.mean(np.diff([dt.datetime_from_epoch_microseconds_utc(i)
                           for i in station.location_sensor().data_timestamps()])))
    # TODO: Print episode start and stop
    print("RedVox location start time:")
    print(dt.datetime_from_epoch_microseconds_utc(station.location_sensor().first_data_timestamp()))
    print("RedVox location stop time:")
    print(dt.datetime_from_epoch_microseconds_utc(station.location_sensor().last_data_timestamp()))
    return station.location_sensor().samples(), station.location_sensor().data_timestamps()


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


def compute_t_xyz_uvw(unix_micros, lat_deg, lon_deg, alt_m):
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
    t_s = (unix_micros - unix_micros.iloc[-1]).astype(float)/1E6

    # Speed in mps. Compute diff and add zero at terminus (at rest)
    u_mps = np.append(np.diff(x_m)/np.diff(t_s), 0)
    v_mps = np.append(np.diff(y_m)/np.diff(t_s), 0)
    w_mps = np.append(np.diff(z_m)/np.diff(t_s), 0)

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
