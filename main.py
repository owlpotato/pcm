import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy 

sys.path.append("grid_data_pull")
sys.path.append("pcm_model")
import grid_data_pull.data_supply
import grid_data_pull.solar_pv_data_pull
import grid_data_pull.solar_thermal_data_pull

## data supply from grid data pull. Shoudl bring in the solar data, electricity prices, and water prices. 
data_supply = grid_data_pull.data_supply.get_data()

## runs the PCM model with the framework 
###(Solar PV or Solar Thermal)
### HP model
### (Geometry)
### (material)
### volume
### output: Strategy (BMS), decision via node like decision tree 
### determination of goal: Eneryg Consumption (Carbonized Cost)

## Data Analysis and Visualization of the results.