"""
Exporting RedPandas example: Skyfall
"""
import examples.skyfall.lib.skyfall_dw as sdw
import redpandas.redpd_df as rpd_df
from examples.skyfall.skyfall_config_file import skyfall_config


def main():
    # load dataframe
    df_skyfall = sdw.dw_main(skyfall_config.tdr_load_method)
    # export dataframe to parquet
    path_export = rpd_df.export_df_to_parquet(df=df_skyfall,
                                              output_dir_pqt=skyfall_config.output_dir,
                                              output_filename_pqt=skyfall_config.pd_pqt_file)


if __name__ == "__main__":
    main()
