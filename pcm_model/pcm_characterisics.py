from dataclasses import dataclass


@dataclass(frozen=True)
class PCM:
    name: str
    rho: float
    c_s: float
    c_l: float
    k_s: float
    k_l: float
    L: float
    T_m: float

    @property
    def h_s(self):
        return self.c_s * self.T_m

    @property
    def h_l(self):
        return self.h_s + self.L

paraffin = PCM("Paraffin", 800, 2000, 2200, 0.24, 0.20, 200e3, 50.0)

MATERIALS = {paraffin.name: paraffin}
DEFAULT_MATERIAL = paraffin


def get_material(name: str = "Paraffin") -> PCM:
    """Return a registered PCM material by name."""
    try:
        return MATERIALS[name]
    except KeyError:
        available = ", ".join(sorted(MATERIALS))
        raise ValueError(f"Unknown PCM material {name!r}. Available materials: {available}") from None