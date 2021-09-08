"""
Runs all skyfall examples
"""

import examples.skyfall.lib.skyfall_tdr_rpd as tdr
import examples.skyfall.lib.skyfall_tfr_rpd as tfr
import examples.skyfall.lib.skyfall_station_specs as sfp
import examples.skyfall.lib.skyfall_ensonify as sfe
import examples.skyfall.lib.skyfall_loc_rpd as sfl
import examples.skyfall.lib.skyfall_spinning as sfs
import examples.skyfall.lib.skyfall_gravity as sfg


if __name__ == "__main__":
    print("RedPandas Example: Skyfall")
    print("\nTime domain representation: skyfall_tdr_rpd.py")
    tdr.main()
    print("\nTime frequency representation: skyfall_tfr_rpd.py")
    tfr.main()
    print("\nStation details: skyfall_station_specs.py")
    sfp.main()
    print("\nSonification: skyfall_ensonify.py")
    sfe.main()
    print("\nLocation: skyfall_loc_rpd.py")
    sfl.main()
    print("\nAcceleration and gravity: skyfall_gravity.py")
    sfg.main()
    print("\nRotation: skyfall_spinning.py")
    sfs.main()
