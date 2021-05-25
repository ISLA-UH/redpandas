import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List
import redvox.common.date_time_utils as dt
from redvox.common.data_window import DataWindow
from libquantum.plot_templates import plot_geo_scatter_2d_3d as geo_scatter


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


def main(rerun_redvox: bool, rerun_bounder: bool):
    """
    Skyfall trajectory information
    :param rerun_redvox:
    :param rerun_bounder:
    :return:
    """

    EVENT_ORIGIN_EPOCH_S = 1603806313.132  # 2020-10-27T13:45:13.132 start time of first data packet
    minutes_duration = 30

    # TODO: Reorganize project structure
    stations_rdvx = ["1637610021"]
    path_rdvx = \
        "/Users/mgarces/Documents/DATA/SDK_DATA/api900_Skyfall_20201027/api900"
    path_bounder = \
        "/Users/mgarces/Documents/mag_2021/papers_2021/Garces_Skyfall_20/trajectory/skyfall_bounder.CSV"
    parquet_rdvx = \
        "/Users/mgarces/Documents/mag_2021/papers_2021/Garces_Skyfall_20/trajectory/skyfall_phone_loc.parquet"
    parquet_bounder = \
        "/Users/mgarces/Documents/mag_2021/papers_2021/Garces_Skyfall_20/trajectory/skyfall_bounder_loc.parquet"

    if rerun_redvox:
        print(rerun_redvox)
        redvox_loc(path_rdvx, stations_rdvx, parquet_rdvx)
    if rerun_bounder:
        bounder_data(path_bounder, parquet_bounder)

    # Load Parquet with phone data fields
    phone_loc = pd.read_parquet(parquet_rdvx)

    # Load Parquet with bounder data fields
    bounder_loc = pd.read_parquet(parquet_bounder)

    # Remove repeated values and NaNs
    # TODO: FIND WHY REPEATED VALUES
    phone_loc = phone_loc[~phone_loc[['Unixtime_micros']].duplicated(keep='first')].dropna()
    # print("Phone Panda head:", phone_loc.head)
    bounder_loc = bounder_loc[~bounder_loc['Unixtime_micros'].duplicated(keep='first')].dropna()
    # print("Balloon Panda head:", bounder_loc.head)

    # Clock check
    print('Bounder start:', bounder_loc['Datetime'].iloc[0])
    print('Bounder end:', bounder_loc['Datetime'].iloc[-1])
    print('Phone loc start:', phone_loc['Datetime'].iloc[0])
    print('Phone loc end:', phone_loc['Datetime'].iloc[-1])

    # Use atmospheric pressure to construct an elevation model
    elevation_model = model_height_from_pressure(pressure_kPa=bounder_loc['Pres_kPa'])

    plt.figure()
    plt.semilogx(bounder_loc['Pres_kPa'], bounder_loc['Alt_m']/1E3, label='data')
    plt.semilogx(bounder_loc['Pres_kPa'], elevation_model/1E3, '-.', label='polynomial')
    plt.legend()
    plt.ylabel('Height, km')
    plt.xlabel('Pressure, kPa')
    plt.title('Bounder Pressure vs Height')

    # Compute XYZ projection
    txyzuvw_phone = \
        compute_t_xyz_uvw(phone_loc['Unixtime_micros'],
                          phone_loc['Lat_deg'],
                          phone_loc['Lat_deg'],
                          phone_loc['Alt_m'])

    txyzuvw_bounder = \
        compute_t_xyz_uvw(bounder_loc['Unixtime_micros'],
                          bounder_loc['Lat_deg'],
                          bounder_loc['Lat_deg'],
                          bounder_loc['Alt_m'])

    # # TODO 20210120: VERIFY TIME AND LOCATION TERMINUS for BOUNDER

    # Bounder temperature is coarse, 1C steps
    plt.figure()
    plt.plot(txyzuvw_bounder['T_s']/60, bounder_loc['Temp_C'])
    plt.title("Skyfall Bounder, Temperature vs Elapsed Time")
    plt.xlabel('Elapsed Time, minutes')
    plt.ylabel('Temp, C')

    # # Phone height has high errors, so w has high errors
    # Not worth it - velocity falls along trajectory, not much new.
    # plt.plot(np.diff(txyzuvw_phone['W_mps']), 'o')
    # plt.ylabel('diff(vertical speed) from phone location, m/s')

    scatter_dot_size = 24
    scatter_colormap = 'inferno'

    # Phone plots
    # 3D scatter plot, LAT LON
    geo_scatter.location_3d(x=phone_loc['Lon_deg'],
                            y=phone_loc['Lat_deg'],
                            z=phone_loc['Alt_m'],
                            color_guide=txyzuvw_phone['T_s']/60,
                            fig_title="Skyfall Path, Phone",
                            x_label='Lat', y_label='Lon', z_label='Z, km',
                            color_label='Elapsed time, minutes',
                            dot_size=scatter_dot_size, color_map=scatter_colormap,
                            azimuth_degrees=-115, elevation_degrees=34)

    # 3D speed quiver plot, velocity
    geo_scatter.loc_quiver_3d(x=txyzuvw_phone['X_m']/1E3,
                              y=txyzuvw_phone['Y_m']/1E3,
                              z=txyzuvw_phone['Z_m']/1E3,
                              u=txyzuvw_phone['U_mps'],
                              v=txyzuvw_phone['V_mps'],
                              w=txyzuvw_phone['W_mps'],
                              color_guide=txyzuvw_phone['T_s']/60,
                              fig_title="Skyfall Path, Phone",
                              x_label='X, km', y_label='Y, km', z_label='Z, km',
                              color_label='Elapsed time, minutes',
                              dot_size=scatter_dot_size, color_map=scatter_colormap,
                              azimuth_degrees=-115, elevation_degrees=34,
                              arrow_length=0.05)

    # XYZ-T
    geo_scatter.location_3d(x=txyzuvw_bounder['X_m']/1E3,
                            y=txyzuvw_bounder['Y_m']/1E3,
                            z=txyzuvw_bounder['Z_m']/1E3,
                            color_guide=txyzuvw_bounder['T_s']/60,
                            fig_title="Skyfall, Bounder",
                            x_label='X, km', y_label='Y, km', z_label='Z, km',
                            color_label='Elapsed time, minutes',
                            dot_size=scatter_dot_size, color_map=scatter_colormap,
                            azimuth_degrees=-80, elevation_degrees=25)

    # XYZ-P
    geo_scatter.location_3d(x=txyzuvw_bounder['X_m']/1E3,
                            y=txyzuvw_bounder['Y_m']/1E3,
                            z=txyzuvw_bounder['Z_m']/1E3,
                            color_guide=bounder_loc['Pres_kPa'],
                            fig_title="Skyfall, Bounder",
                            x_label='X, km', y_label='Y, km', z_label='Z, km',
                            color_label='Pressure, kPa',
                            dot_size=scatter_dot_size, color_map=scatter_colormap,
                            azimuth_degrees=-80, elevation_degrees=25)

    # XYZ-Speed
    geo_scatter.location_3d(x=txyzuvw_bounder['X_m']/1E3,
                            y=txyzuvw_bounder['Y_m']/1E3,
                            z=txyzuvw_bounder['Z_m']/1E3,
                            color_guide=txyzuvw_bounder['Speed_mps'],
                            fig_title="Skyfall, Bounder",
                            x_label='X, km', y_label='Y, km', z_label='Z, km',
                            color_label='Speed, m/s',
                            dot_size=scatter_dot_size, color_map=scatter_colormap,
                            azimuth_degrees=-80, elevation_degrees=25)


    # Overlay
    geo_scatter.loc_overlay_3d(x1=bounder_loc['Lon_deg'],
                               y1=bounder_loc['Lat_deg'],
                               z1=bounder_loc['Alt_m']/1E3,
                               dot_size1=9,
                               color1='grey',
                               legend1='Bounder',
                               alpha1=1,
                               x2=phone_loc['Lon_deg'],
                               y2=phone_loc['Lat_deg'],
                               z2=phone_loc['Alt_m']/1E3,
                               dot_size2=6,
                               color2='b',
                               legend2='Phone',
                               alpha2=0.6,
                               fig_title="Skyfall Path, Phone and Bounder",
                               x_label='Lat', y_label='Lon', z_label='Z, km',
                               azimuth_degrees=-115, elevation_degrees=34)

    plt.show()


if __name__ == '__main__':
    """
    Paths from phone and bounder for NNSS Skyfall data set
    If true, rerun and save as parquet
    """
    # is_rerun_phone = True
    # is_rerun_bounder = True
    is_rerun_phone = False
    is_rerun_bounder = False
    main(is_rerun_phone, is_rerun_bounder)
