#%%
from pathlib import Path
import pandas as pd
import numpy as np

# pyhcal imports
from pyhcal import calibrators
from pyhcal import PestWorkflow


#%%
project_path = 'C:\\Users\\mfratki\\Documents\\Projects\\Tests\\Nemadji'

pest = PestWorkflow(project_path)
pest.load_model(0)

#%% Define Parameters to calibrate
block = 'PERLND'
table = 'PWAT-PARM2'
table_id = 0
parameter = 'LZSN'

pest.add_hspf_parameter(block, table, table_id, parameter,single_template=False)



#%% Add observation data
name = 'outlet'
station_ids = ['E05011002']
reach_ids = [103]
constituent = 'Q'
time_step = 'D'
df = pest.cal.compare_simulated_observed(station_ids, reach_ids, constituent, time_step).dropna(subset=['observed'])
pest.add_observation(name, constituent, observed=df['observed'], simulated=df['simulated'])

#%%
pest.build()
pest.write("my_case.pst")


#%%
