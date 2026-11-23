# pcm
Project regarding PCM

## PCM simulations

The 1D and 2D solvers share material properties from `pcm_model.pcm_characterisics`.
Select a registered material with `get_material()` and pass it to the operation you
want to run:

```python
from pcm_model.charge import charge_1d
from pcm_model.discharge import discharge_2d
from pcm_model.pcm_characterisics import get_material

material = get_material("Paraffin")
t, soc, completion_time = charge_1d(70.0, material=material)
result = discharge_2d(
    0.05, 0.04, 50, 40, wall_mask, 30.0, material=material
)
```

`charge_1d`/`discharge_1d` return time, state-of-charge, and completion time.
The 2D operations return the solver result dictionary, including its SoC history.
Use `pcm_model.soc` for enthalpy, temperature, and state-of-charge conversions.
The simulation code is maintained in `charge.py` and `discharge.py`; the former
`heatsim1Dstefan.py` and `heatsim2Dstefan.py` modules have been removed.
