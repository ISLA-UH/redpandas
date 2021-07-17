# Red Pandas modules
from redpandas.redpd_dw_to_parquet import redpd_df, redpd_df_from_config

# RedPandas config file
from examples.skyfall.skyfall_config_file import skyfall_config


def main():
    """
    Beta workflow for API M pipeline
    Last updated: 18 June 2021
    """
    print('Let the sky fall')
    string, df = redpd_df(input_dir=skyfall_config.input_dir,
                          event_name=skyfall_config.event_name,
                          export_dw_pickle=False,
                          export_df_parquet=False,
                          create_dw=True,
                          print_dq=False,
                          show_raw_waveform_plots=True,
                          output_dir=skyfall_config.output_dir,
                          output_filename_pkl=skyfall_config.dw_file,
                          output_filename_pqt=skyfall_config.pd_pqt_file,
                          station_ids=skyfall_config.station_ids,
                          sensor_labels=skyfall_config.sensor_labels,
                          start_epoch_s=skyfall_config.event_start_epoch_s,
                          end_epoch_s=skyfall_config.event_end_epoch_s,
                          start_buffer_minutes=skyfall_config.start_buffer_minutes,
                          end_buffer_minutes=skyfall_config.end_buffer_minutes,
                          debug=False)

if __name__ == "__main__":
    main()
