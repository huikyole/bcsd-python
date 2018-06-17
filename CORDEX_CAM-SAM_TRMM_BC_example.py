import ocw.dataset_processor as dsp
import ocw.utils as utils

import os, sys
import time
import yaml     

import xarray as xr
import numpy as np

from bias_correct import BiasCorrectDaily, convert_to_float32

config_file = str(sys.argv[1])
args = yaml.load(open(config_file))
print "Case:", args['case_name']
obs_data = xr.open_dataset(args['data']['directory']+args['data']['fobserved'])

print "loading observations"
obs_data.load()
obs_data = convert_to_float32(obs_data)

print "loading modeled"
modeled_data_present = xr.open_dataset(args['data']['directory']+args['data']['fmodeled_present'])
modeled_data_present.load()
convert_to_float32(modeled_data_present)

modeled_data_future = xr.open_dataset(args['data']['directory']+args['data']['fmodeled_future'])
modeled_data_future.load()
convert_to_float32(modeled_data_future)
   
print "starting bias correction"
t0 = time.time()
bc = BiasCorrectDaily()
corrected = bc.bias_correction(obs_data, modeled_data_present, modeled_data_future, 
                               args['data']['observed_varname'],
                               args['data']['modeled_varname'])
print "writing bias corrected model output"
corrected.to_netcdf(args['data']['directory']+'BC_'+args['data']['fmodeled_future'])
print "total running time for the bias correction:", (time.time() - t0), 'seconds'
   
