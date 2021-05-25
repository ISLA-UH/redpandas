import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import redpandas.redpd_geospatial as rpd_geo
from libquantum.plot_templates import plot_geo_scatter_2d_3d as geo_scatter

# Configuration file
from examples.skyfall.skyfall_config import EVENT_NAME, INPUT_DIR, EPISODE_START_EPOCH_S, \
    EPISODE_END_EPOCH_S, STATIONS, PD_PQT_FILE, OUTPUT_DIR, DW_FILE, build_dw_pickle, build_df_parquet, \
    plot_mic_waveforms, print_datawindow_dq, SENSOR_LABEL

def main(rerun_redvox: bool, rerun_bounder: bool):
    """
    Skyfall trajectory information
    :param rerun_redvox:
    :param rerun_bounder:
    :return:
    """

    # TODO: Reorganize project structure
    # Point to configuration file
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
        rpd_geo.redvox_loc(path_rdvx, stations_rdvx, parquet_rdvx)
    if rerun_bounder:
        rpd_geo.bounder_data(path_bounder, parquet_bounder)

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
    elevation_model = rpd_geo.model_height_from_pressure(pressure_kPa=bounder_loc['Pres_kPa'])

    plt.figure()
    plt.semilogx(bounder_loc['Pres_kPa'], bounder_loc['Alt_m']/1E3, label='data')
    plt.semilogx(bounder_loc['Pres_kPa'], elevation_model/1E3, '-.', label='polynomial')
    plt.legend()
    plt.ylabel('Height, km')
    plt.xlabel('Pressure, kPa')
    plt.title('Bounder Pressure vs Height')

    # Compute XYZ projection
    txyzuvw_phone = \
        rpd_geo.compute_t_xyz_uvw(phone_loc['Unixtime_micros'],
                          phone_loc['Lat_deg'],
                          phone_loc['Lat_deg'],
                          phone_loc['Alt_m'])

    txyzuvw_bounder = \
        rpd_geo.compute_t_xyz_uvw(bounder_loc['Unixtime_micros'],
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
