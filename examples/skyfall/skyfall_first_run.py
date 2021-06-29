from skyfall_config_file import skyfall_config
import examples.skyfall.skyfall_ensonify as skyfall_ensonify
import examples.skyfall.skyfall_tdr_rpd as skyfall_tdr_rpd
import examples.skyfall.skyfall_tfr_rpd as skyfall_tfr_rpd
from redpandas.redpd_dw_to_parquet import redpd_dw_to_parquet_from_config

import os

# assume they have not looked at the config file
# check redpandas requirements are met

# TODO MC: check every import library is in redpandas requirements
# TODO MC: indicate the skyfall_config_file, surpress column and station index in redpd_dw_to_parquet


if __name__ == "__main__":

    print("Welcome to RedVox Redpandas Skyfall example")
    print("Press enter to continue.")
    input()
    print("Before we start, please check that:"
          "\n 1) The Skyfall configuration file (skyfall_config_file.py) is in the same folder as this file (skyfall_first_run.py)"
          "\n 2) The INPUT_DIR has been changed to path/to/file where the Skyfall data is located"
          "\n 3)")
    print("Press enter to continue.")
    input()

    # Check that Skyfall data is where it should be
    if not os.path.exists(skyfall_config.input_dir):
        print(f"\nInput directory does not exist, check INPUT_DIR in skyfall_config_file.py: {skyfall_config.input_dir}")
        exit()

    print("\nFirst step: load the RedVox Skyfall data (.rdvxz) and convert it into a parquet.")
    print("Press enter to run redpd_dw_to_parquet. Note: this might take a few minutes.")
    input()
    redpd_dw_to_parquet_from_config(config=skyfall_config,
                                    show_raw_waveform_plots=False)

    print(f"\nYou can find the RedPandas parquet at {skyfall_config.output_dir + '/' + skyfall_config.pd_pqt_file}")
    print(f"You can also find a pickle file with the RedVox Datawindow {skyfall_config.output_dir + '/dw/' + skyfall_config.dw_file}"
          f"\nand JSON file at {skyfall_config.output_dir + '/' + skyfall_config.output_filename_pkl_pqt + '.json'}")
    print("Press enter to continue.")
    input()

    print("Now that the RedPandas parquet is constructed, we can showcase some RedPandas products.")
    print("In the Skyfall example: "  
          "\n 1) Ensonify"
          "\n 2) Plot Time-Domain Representation"
          "\n 3) Plot Time-Frequency Representation")

    print("Press enter to continue.")
    input()

    # Ensonify data
    print("\nLet's ensonify (aka make audible) the Skyfall data:")
    print("Press enter to run skyfall_ensonify.py")
    input()
    skyfall_ensonify.main()
    print(f"\nYou can find the .wav files in {skyfall_config.input_dir + '/wav'}. You can listen to this data with the free, "
          f"open-source app Audacity.")
    print("Press enter to continue.")
    input()

    # TDR
    print("\nLet's plot time-domain representation of the Skyfall data")
    print("Press enter to run skyfall_tdr_rpd.py")
    input()
    skyfall_tdr_rpd.main()

    # TFR
    print("\nLet's plot time-frequency representation of the Skyfall data")
    print("Press enter to run skyfall_tfr_rpd.py")
    input()
    skyfall_tfr_rpd.main()

    print("\nThank you for running the RedPandas Skyfall example.")
    print("Please feel free to check the RedPandas Documentation at https://github.com/RedVoxInc/redpandas to learn more about RedPandas.")


