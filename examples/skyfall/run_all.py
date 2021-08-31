import examples.skyfall.lib.skyfall_ensonify as sfe
import examples.skyfall.lib.skyfall_gravity as sfg
import examples.skyfall.lib.skyfall_loc_rpd as sfl
import examples.skyfall.lib.skyfall_spinning as sfs
import examples.skyfall.lib.skyfall_station_specs as sfp
import examples.skyfall.lib.skyfall_tdr_rpd as tdr
import examples.skyfall.lib.skyfall_tfr_rpd as tfr


if __name__ == "__main__":
    print("ensonify")
    sfe.main()
    print("\ngravity")
    sfg.main()
    print("\nloc_rpd")
    sfl.main()
    print("\nspinning")
    sfs.main()
    print("\nspecs")
    sfp.main()
    print("\ntdr")
    tdr.main()
    print("\ntfr")
    tfr.main()
