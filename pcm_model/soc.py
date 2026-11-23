import numpy as np

from .pcm_characterisics import DEFAULT_MATERIAL, PCM


def state_of_charge(enthalpy, material: PCM = DEFAULT_MATERIAL):
    """Return the liquid fraction represented by enthalpy."""
    return np.clip((enthalpy - material.h_s) / material.L, 0.0, 1.0)


def mean_state_of_charge(enthalpy, material: PCM = DEFAULT_MATERIAL):
    """Return the spatially averaged liquid fraction represented by enthalpy."""
    return float(np.mean(state_of_charge(enthalpy, material)))


def state_of_charge_rate(soc_now, soc_then, elapsed_seconds):
    """Return the absolute state-of-charge change as percent per hour."""
    return abs(soc_now - soc_then) / elapsed_seconds * 3600.0 * 100.0


def temperature_from_enthalpy(enthalpy, material: PCM = DEFAULT_MATERIAL):
    """Convert enthalpy to temperature using the PCM's sensible and latent regions."""
    return np.where(
        enthalpy <= material.h_s,
        enthalpy / material.c_s,
        np.where(
            enthalpy >= material.h_l,
            material.T_m + (enthalpy - material.h_l) / material.c_l,
            material.T_m,
        ),
    )


def enthalpy_from_state_of_charge(soc, material: PCM = DEFAULT_MATERIAL):
    """Convert a state of charge in [0, 1] to latent-region enthalpy."""
    return material.h_s + np.clip(np.asarray(soc), 0.0, 1.0) * material.L