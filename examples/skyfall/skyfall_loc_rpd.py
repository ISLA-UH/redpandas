import numpy as np
import os
import matplotlib.pyplot as plt
import redvox.common.date_time_utils as dt
import pandas as pd
import redpandas.redpd_geospatial as rpd_geo
from libquantum.plot_templates import plot_geo_scatter_2d_3d as geo_scatter

# Configuration file
from examples.skyfall.skyfall_config import OUTPUT_DIR, PD_PQT_FILE, OTHER_INPUT_PATH, OTHER_PD_PQT_FILE

def main(rerun_bounder: bool):
    """
    Skyfall trajectory information
    :param rerun_bounder:
    :return:
    """
    # Use configuration file to load rdvx parquet for the data window
    path_pickle_df = os.path.join(OUTPUT_DIR, PD_PQT_FILE)

    phone_loc = rpd_geo.redvox_loc(path_pickle_df)
    print(phone_loc.columns)

    # Bounder data
    if rerun_bounder:
        rpd_geo.bounder_data(OTHER_INPUT_PATH, OTHER_PD_PQT_FILE)
        print('Constructing bounder parquet')

    # Load parquet with bounder data fields
    bounder_loc = pd.read_parquet(OTHER_PD_PQT_FILE)
    print('Loaded Bounder parquet')

    # # Remove repeated values and NaNs
    bounder_loc = bounder_loc[~bounder_loc['Epoch_s'].duplicated(keep='first')].dropna()

    # Clock check
    print('Bounder start:', bounder_loc['Datetime'].iloc[0])
    print('Bounder end:', bounder_loc['Datetime'].iloc[-1])

    phone_datetime_start = dt.datetime_from_epoch_seconds_utc(phone_loc['location_epoch_s'].iloc[0][0])
    phone_datetime_end = dt.datetime_from_epoch_seconds_utc(phone_loc['location_epoch_s'].iloc[0][-1])
    print('Phone loc start:', phone_datetime_start)
    print('Phone loc end:', phone_datetime_end)
    #
    # Use atmospheric pressure to construct an elevation model
    elevation_model = rpd_geo.model_height_from_pressure(pressure_kPa=bounder_loc['Pres_kPa'])

    plt.figure()
    plt.semilogx(bounder_loc['Pres_kPa'], bounder_loc['Alt_m']/1E3, label='data')
    plt.semilogx(bounder_loc['Pres_kPa'], elevation_model/1E3, '-.', label='polynomial')
    plt.legend()
    plt.ylabel('Height, km')
    plt.xlabel('Pressure, kPa')
    plt.title('Bounder Pressure vs Height')
    #
    # # Compute XYZ projection
    # txyzuvw_phone = \
    #     rpd_geo.compute_t_xyz_uvw(phone_loc['location_epoch_s'],
    #                       phone_loc['location_latitude'],
    #                       phone_loc['location_longitude'],
    #                       phone_loc['location_altitude'])
    #
    txyzuvw_bounder = \
        rpd_geo.compute_t_xyz_uvw(bounder_loc['Epoch_s'],
                          bounder_loc['Lat_deg'],
                          bounder_loc['Lon_deg'],
                          bounder_loc['Alt_m'])
    #
    # # # TODO 20210120: VERIFY TIME AND LOCATION TERMINUS for BOUNDER
    #
    # # Bounder temperature is coarse, 1C steps
    # plt.figure()
    # plt.plot(txyzuvw_bounder['T_s']/60, bounder_loc['Temp_C'])
    # plt.title("Skyfall Bounder, Temperature vs Elapsed Time")
    # plt.xlabel('Elapsed Time, minutes')
    # plt.ylabel('Temp, C')
    #
    # # # Phone height has high errors, so w has high errors
    # # Not worth it - velocity falls along trajectory, not much new.
    # # plt.plot(np.diff(txyzuvw_phone['W_mps']), 'o')
    # # plt.ylabel('diff(vertical speed) from phone location, m/s')
    #
    scatter_dot_size = 24
    scatter_colormap = 'inferno'
    #
    # # Phone plots
    # # 3D scatter plot, LAT LON
    # geo_scatter.location_3d(x=phone_loc['Lon_deg'],
    #                         y=phone_loc['Lat_deg'],
    #                         z=phone_loc['Alt_m'],
    #                         color_guide=txyzuvw_phone['T_s']/60,
    #                         fig_title="Skyfall Path, Phone",
    #                         x_label='Lat', y_label='Lon', z_label='Z, km',
    #                         color_label='Elapsed time, minutes',
    #                         dot_size=scatter_dot_size, color_map=scatter_colormap,
    #                         azimuth_degrees=-115, elevation_degrees=34)
    #
    # # 3D speed quiver plot, velocity
    # geo_scatter.loc_quiver_3d(x=txyzuvw_phone['X_m']/1E3,
    #                           y=txyzuvw_phone['Y_m']/1E3,
    #                           z=txyzuvw_phone['Z_m']/1E3,
    #                           u=txyzuvw_phone['U_mps'],
    #                           v=txyzuvw_phone['V_mps'],
    #                           w=txyzuvw_phone['W_mps'],
    #                           color_guide=txyzuvw_phone['T_s']/60,
    #                           fig_title="Skyfall Path, Phone",
    #                           x_label='X, km', y_label='Y, km', z_label='Z, km',
    #                           color_label='Elapsed time, minutes',
    #                           dot_size=scatter_dot_size, color_map=scatter_colormap,
    #                           azimuth_degrees=-115, elevation_degrees=34,
    #                           arrow_length=0.05)
    #
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
    #
    # # XYZ-P
    # geo_scatter.location_3d(x=txyzuvw_bounder['X_m']/1E3,
    #                         y=txyzuvw_bounder['Y_m']/1E3,
    #                         z=txyzuvw_bounder['Z_m']/1E3,
    #                         color_guide=bounder_loc['Pres_kPa'],
    #                         fig_title="Skyfall, Bounder",
    #                         x_label='X, km', y_label='Y, km', z_label='Z, km',
    #                         color_label='Pressure, kPa',
    #                         dot_size=scatter_dot_size, color_map=scatter_colormap,
    #                         azimuth_degrees=-80, elevation_degrees=25)
    #
    # # XYZ-Speed
    # geo_scatter.location_3d(x=txyzuvw_bounder['X_m']/1E3,
    #                         y=txyzuvw_bounder['Y_m']/1E3,
    #                         z=txyzuvw_bounder['Z_m']/1E3,
    #                         color_guide=txyzuvw_bounder['Speed_mps'],
    #                         fig_title="Skyfall, Bounder",
    #                         x_label='X, km', y_label='Y, km', z_label='Z, km',
    #                         color_label='Speed, m/s',
    #                         dot_size=scatter_dot_size, color_map=scatter_colormap,
    #                         azimuth_degrees=-80, elevation_degrees=25)
    #
    #
    # # Overlay
    # geo_scatter.loc_overlay_3d(x1=bounder_loc['Lon_deg'],
    #                            y1=bounder_loc['Lat_deg'],
    #                            z1=bounder_loc['Alt_m']/1E3,
    #                            dot_size1=9,
    #                            color1='grey',
    #                            legend1='Bounder',
    #                            alpha1=1,
    #                            x2=phone_loc['Lon_deg'],
    #                            y2=phone_loc['Lat_deg'],
    #                            z2=phone_loc['Alt_m']/1E3,
    #                            dot_size2=6,
    #                            color2='b',
    #                            legend2='Phone',
    #                            alpha2=0.6,
    #                            fig_title="Skyfall Path, Phone and Bounder",
    #                            x_label='Lat', y_label='Lon', z_label='Z, km',
    #                            azimuth_degrees=-115, elevation_degrees=34)
    #
    plt.show()


if __name__ == '__main__':
    """
    Paths from phone and bounder for NNSS Skyfall data set
    If true, rerun and save as parquet
    """
    is_rerun_bounder = True
    # is_rerun_bounder = False
    main(is_rerun_bounder)
