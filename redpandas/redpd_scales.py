"""
Scale conversions.
"""

import numpy as np

# copied from scales in libquantum, decide if we want to use libquantum.scales or have a redpd scales

""" Smallest number for 64-bit floats. Deploy to avoid division by zero or log zero singularities"""
EPSILON = np.finfo(np.float64).eps
NANOS_TO_S = 1E-9
MICROS_TO_S = 1E-6
MICROS_TO_MILLIS = 1E-3
SECONDS_TO_MINUTES = 1./60.
DEGREES_TO_KM = 111.
DEGREES_TO_METERS = 111000.
METERS_TO_KM = 1E-3
PRESSURE_SEA_LEVEL_KPA = 101.325
KPA_TO_PA = 1E3
MG_RT = 0.00012  # Molar mass of air x gravity / (gas constant x standard temperature)
PRESSURE_REF_kPa = 101.325


class Slice:
    """
    Constants for slice calculations, supersedes inferno/slice
    """
    # Constant Q Base
    G2 = 2.  # Octaves
    G3 = 10. ** 0.3  # Reconciles base2 and base10
    # Time
    T_PLANCK = 5.4E-44  # 2.**(-144)   Planck time in seconds
    T0S = 1E-42  # Universal Scale
    T1S = 1.    # 1 second
    T100S = 100.  # 1 hectosecond, IMS low band edge
    T1000S = 1000.  # 1 kiloseconds = 1 mHz
    T1M = 60.  # 1 minute in seconds
    T1H = T1M*60.  # 1 hour in seconds
    T1D = T1H*24.   # 1 day in seconds
    TU = 2.**58  # Estimated age of the known universe in seconds
    # Frequency
    F1 = 1.  # 1 Hz
    F1000 = 1000.  # 1 kHz
    F0 = 1.E42  # 1/Universal Scale
    FU = 2.**-58  # Estimated age of the universe in Hz
    # Pressure
    PREF_KPA = 101.325  # sea level pressure, kPa
    PREF_PA = 10132500.  # sea level pressure, kPa
