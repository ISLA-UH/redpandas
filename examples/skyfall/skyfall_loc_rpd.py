import os
import pandas as pd
import matplotlib.pyplot as plt
import csv

import redvox.common.date_time_utils as dt
from libquantum.plot_templates import plot_geo_scatter_2d_3d as geo_scatter
import redpandas.redpd_geospatial as rpd_geo

# Import constants
from redpandas.redpd_scales import METERS_TO_KM, SECONDS_TO_MINUTES

# Configuration file
# from examples.skyfall.skyfall_config import EVENT_NAME, OUTPUT_DIR, PD_PQT_FILE, \
#     OTHER_INPUT_PATH, OTHER_INPUT_FILE, OTHER_PD_PQT_FILE, is_rerun_bounder

from examples.skyfall.skyfall_config_file import skyfall_config, OTHER_INPUT_PATH, OTHER_INPUT_FILE, OTHER_PD_PQT_FILE
skyfall_config.is_rerun_bounder = True

def bounder_specs_to_csv(df, csv_export_file):

    with open(csv_export_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        writer.writerow(["Description", "Value"])
        writer.writerow(['Start Date Time', df['Datetime'].iloc[0]])
        writer.writerow(['Start Epoch s', df['Epoch_s'].iloc[0]])
        writer.writerow(['Start Latitude degrees', df['Lat_deg'].iloc[0]])
        writer.writerow(['Start Longitude degrees', df['Lon_deg'].iloc[0]])
        writer.writerow(['Start Altitude m (WGS-84)', df['Alt_m'].iloc[0]])
        writer.writerow(['Stop Date Time', df['Datetime'].iloc[-1]])
        writer.writerow(['Stop Epoch s', df['Epoch_s'].iloc[-1]])
        writer.writerow(['Stop Latitude degrees', df['Lat_deg'].iloc[-1]])
        writer.writerow(['Stop Longitude degrees', df['Lon_deg'].iloc[-1]])
        writer.writerow(['Stop Altitude m  (WGS-84)', df['Alt_m'].iloc[-1]])


if __name__ == '__main__':
    # TODO MC: probs broken

    """
    Paths from phone and bounder for NNSS Skyfall data set
    If true, rerun and save as parquet
    """
    # is_rerun_bounder = True
    # is_rerun_bounder = False

    """
    Skyfall trajectory information
    :param rerun_bounder:
    :return:
    """
    # Use configuration file to load rdvx parquet for the data window
    rdvx_path_pickle_df = os.path.join(skyfall_config.output_dir, skyfall_config.pd_pqt_file)
    # Concentrate on single station
    phone_id = "1637610021"

    # Load for all stations
    df_loc = rpd_geo.redvox_loc(rdvx_path_pickle_df)
    print(f'Dimensions (# of rows, # of columns): {df_loc.shape}')

    # Pick only the balloon station
    m_list = df_loc.index[df_loc['station_id'] == phone_id]
    m = m_list[0]
    phone_loc = df_loc.iloc[m]

    # Verify
    print(f'Verify that balloon station selected matches # of columns: {phone_loc.shape}')

    # Bounder data is a standard rectangular matrix
    if not os.path.exists(OTHER_INPUT_PATH):
        print("Other input directory does not exist, check path:")
        print(OTHER_INPUT_PATH)
        exit()

    if skyfall_config.is_rerun_bounder:
        rpd_geo.bounder_data(OTHER_INPUT_PATH, OTHER_INPUT_FILE, OTHER_PD_PQT_FILE)
        print('Constructing bounder parquet')

    # Load parquet with bounder data fields
    print('Load Bounder parquet:')
    bounder_loc = pd.read_parquet(os.path.join(OTHER_INPUT_PATH, OTHER_PD_PQT_FILE))
    print(f'Dimensions (# of rows, # of columns): {bounder_loc.shape}')
    print(f'Available columns: {bounder_loc.columns}')

    # Remove bounder repeated values and NaNs
    # DataWindow should be cleared of nans
    # phone_loc = phone_loc[~phone_loc['location_epoch_s'].duplicated(keep='first')].dropna()
    bounder_loc = bounder_loc[~bounder_loc['Epoch_s'].duplicated(keep='first')].dropna()

    # Bounder clock, initial, and final conditions
    print('\nBounder Start Time:', bounder_loc['Datetime'].iloc[0])
    print('Bounder Start Epoch s:', bounder_loc['Epoch_s'].iloc[0])
    print('Bounder Start Lat:', bounder_loc['Lat_deg'].iloc[0])
    print('Bounder Start Lon:', bounder_loc['Lon_deg'].iloc[0])
    print('Bounder Start Alt:', bounder_loc['Alt_m'].iloc[0])
    print('\nBounder End Time:', bounder_loc['Datetime'].iloc[-1])
    print('Bounder Terminus Parameters (Ref):')
    print('Bounder Ref Epoch s:', bounder_loc['Epoch_s'].iloc[-1])
    print('Bounder Ref Lat:', bounder_loc['Lat_deg'].iloc[-1])
    print('Bounder Ref Lon:', bounder_loc['Lon_deg'].iloc[-1])
    print('Bounder Ref Alt:', bounder_loc['Alt_m'].iloc[-1])

    # Export Initial and Final states to CSV
    print(f"Export Bounder initial and final states to CSV. Path: "
          f"{os.path.join(OTHER_INPUT_PATH, skyfall_config.event_name + '_bounder_start_end.csv')}")
    file_bounder_start_end_csv = os.path.join(OTHER_INPUT_PATH, skyfall_config.event_name
                                              + '_bounder_start_end.csv')
    bounder_specs_to_csv(df=bounder_loc, csv_export_file=file_bounder_start_end_csv)

    # TODO: Use the bounder terminus values in the configuration file
    # Propagated to tdr
    ref_latitude_deg = 35.83728
    ref_longitude_deg = -115.57234
    ref_altitude_m = 1028.2
    ref_epoch_s = 1603808160

    # Compare to phone
    phone_datetime_start = dt.datetime_from_epoch_seconds_utc(phone_loc['location_epoch_s'][0])
    phone_datetime_end = dt.datetime_from_epoch_seconds_utc(phone_loc['location_epoch_s'][-1])
    print('Phone loc start:', phone_datetime_start)
    print('Phone loc end:', phone_datetime_end)

    # Use atmospheric pressure to construct an elevation model
    elevation_model = rpd_geo.bounder_model_height_from_pressure(pressure_kPa=bounder_loc['Pres_kPa'])

    plt.figure()
    plt.semilogx(bounder_loc['Pres_kPa'], bounder_loc['Alt_m']*METERS_TO_KM, label='data')
    plt.semilogx(bounder_loc['Pres_kPa'], elevation_model*METERS_TO_KM, '-.', label='polynomial')
    plt.legend()
    plt.ylabel('Height, km')
    plt.xlabel('Pressure, kPa')
    # plt.title('Bounder Pressure vs Height')

    # Compute ENU projections
    txyzuvw_phone = \
        rpd_geo.compute_t_xyz_uvw(unix_s=phone_loc['location_epoch_s'],
                                  lat_deg=phone_loc['location_latitude'],
                                  lon_deg=phone_loc['location_longitude'],
                                  alt_m=phone_loc['location_altitude'],
                                  ref_unix_s=ref_epoch_s,
                                  ref_lat_deg=ref_latitude_deg,
                                  ref_lon_deg=ref_longitude_deg,
                                  ref_alt_m=ref_altitude_m)
    txyzuvw_bounder = \
        rpd_geo.compute_t_xyz_uvw(unix_s=bounder_loc['Epoch_s'],
                                  lat_deg=bounder_loc['Lat_deg'],
                                  lon_deg=bounder_loc['Lon_deg'],
                                  alt_m=bounder_loc['Alt_m'],
                                  ref_unix_s=ref_epoch_s,
                                  ref_lat_deg=ref_latitude_deg,
                                  ref_lon_deg=ref_longitude_deg,
                                  ref_alt_m=ref_altitude_m)

    # Internal Bounder temperature is coarse, 1C steps
    plt.figure()
    plt.plot(txyzuvw_bounder['T_s']*SECONDS_TO_MINUTES, bounder_loc['Temp_C'])
    plt.title("Skyfall Bounder, Temperature vs Elapsed Time")
    plt.xlabel('Elapsed Time, minutes')
    plt.ylabel('Temp, C')

    # Scatter plots are cool
    scatter_dot_size = 24
    scatter_colormap = 'inferno'

    # Phone plots
    # 3D scatter plot, LAT LON
    # title_str = "Skyfall Path, Phone"
    title_str = ""
    geo_scatter.location_3d(x=phone_loc['location_longitude'],
                            y=phone_loc['location_latitude'],
                            z=phone_loc['location_altitude']*METERS_TO_KM,
                            color_guide=txyzuvw_phone['T_s']*SECONDS_TO_MINUTES,
                            fig_title=title_str,
                            x_label='Lat', y_label='Lon', z_label='Z, km',
                            color_label='Elapsed time, minutes',
                            dot_size=scatter_dot_size, color_map=scatter_colormap,
                            azimuth_degrees=-134, elevation_degrees=30)

    # 3D speed quiver plot, velocity
    geo_scatter.loc_quiver_3d(x=txyzuvw_phone['X_m']*METERS_TO_KM,
                              y=txyzuvw_phone['Y_m']*METERS_TO_KM,
                              z=txyzuvw_phone['Z_m']*METERS_TO_KM,
                              u=txyzuvw_phone['U_mps'],
                              v=txyzuvw_phone['V_mps'],
                              w=txyzuvw_phone['W_mps'],
                              color_guide=txyzuvw_phone['T_s']*SECONDS_TO_MINUTES,
                              fig_title=title_str,
                              x_label='X, km', y_label='Y, km', z_label='Z, km',
                              color_label='Elapsed time, minutes',
                              dot_size=scatter_dot_size, color_map=scatter_colormap,
                              azimuth_degrees=-134, elevation_degrees=30,
                              arrow_length=0.05)

    # XYZ-T
    geo_scatter.location_3d(x=txyzuvw_bounder['X_m']*METERS_TO_KM,
                            y=txyzuvw_bounder['Y_m']*METERS_TO_KM,
                            z=txyzuvw_bounder['Z_m']*METERS_TO_KM,
                            color_guide=txyzuvw_bounder['T_s']*SECONDS_TO_MINUTES,
                            fig_title=title_str,
                            x_label='X, km', y_label='Y, km', z_label='Z, km',
                            color_label='Elapsed time, minutes',
                            dot_size=scatter_dot_size, color_map=scatter_colormap,
                            azimuth_degrees=-80, elevation_degrees=25)
    #
    # XYZ-P
    # title_str = "Skyfall, Bounder"
    title_str = ""
    geo_scatter.location_3d(x=txyzuvw_bounder['X_m']*METERS_TO_KM,
                            y=txyzuvw_bounder['Y_m']*METERS_TO_KM,
                            z=txyzuvw_bounder['Z_m']*METERS_TO_KM,
                            color_guide=bounder_loc['Pres_kPa'],
                            fig_title=title_str,
                            x_label='X, km', y_label='Y, km', z_label='Z, km',
                            color_label='Pressure, kPa',
                            dot_size=scatter_dot_size, color_map=scatter_colormap,
                            azimuth_degrees=-80, elevation_degrees=25)

    # XYZ-Speed
    geo_scatter.location_3d(x=txyzuvw_bounder['X_m']*METERS_TO_KM,
                            y=txyzuvw_bounder['Y_m']*METERS_TO_KM,
                            z=txyzuvw_bounder['Z_m']*METERS_TO_KM,
                            color_guide=txyzuvw_bounder['Speed_mps'],
                            fig_title=title_str,
                            x_label='X, km', y_label='Y, km', z_label='Z, km',
                            color_label='Speed, m/s',
                            dot_size=scatter_dot_size, color_map=scatter_colormap,
                            azimuth_degrees=-80, elevation_degrees=25)


    # Overlay
    # title_str = "Skyfall Path, Phone and Bounder"
    title_str = ""
    geo_scatter.loc_overlay_3d(x1=bounder_loc['Lon_deg'],
                               y1=bounder_loc['Lat_deg'],
                               z1=bounder_loc['Alt_m']*METERS_TO_KM,
                               dot_size1=9,
                               color1='grey',
                               legend1='Bounder',
                               alpha1=1,
                               x2=phone_loc['location_longitude'],
                               y2=phone_loc['location_latitude'],
                               z2=phone_loc['location_altitude']*METERS_TO_KM,
                               dot_size2=6,
                               color2='b',
                               legend2='Phone',
                               alpha2=0.6,
                               fig_title=title_str,
                               x_label='Lat', y_label='Lon', z_label='Z, km',
                               azimuth_degrees=-134, elevation_degrees=30)

    plt.show()

