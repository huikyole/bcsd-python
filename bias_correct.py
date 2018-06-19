import pickle
import os, sys
import time

import numpy as np
import numpy.ma as ma
import xray
from joblib import Parallel, delayed

from qmap import QMap

np.seterr(invalid='ignore')

def mapper(x, y, z, train_num, step=0.5):
    qmap = QMap(step=step)
    qmap.fit(x[:train_num], y[:train_num], z[:train_num], axis=0)
    return qmap.predict(z)

def nanarray(size):
    arr = np.empty(size)
    arr[:] = np.nan
    return arr

def convert_to_float32(ds):
    for var in ds.data_vars:
        if ds[var].dtype == 'float64':
            ds[var] = ds[var].astype('float32', copy=False)
    return ds

class BiasCorrectDaily():
    """ A class which can perform bias correction on daily data

    The process applied is based on the bias correction process applied by
    the NASA NEX team
    (https://nex.nasa.gov/nex/static/media/other/NEX-GDDP_Tech_Note_v1_08June2015.pdf)
    This process does NOT require temporal disaggregation from monthly to daily time steps.
    Instead pooling is used to capture a greater range of variablity
    """
    def __init__(self, pool=15, max_train_year=np.inf, step=0.5):
        self.pool = pool
        self.max_train_year = max_train_year
        self.step = step

    def bias_correction(self, obs, modeled_present, modeled_future, obs_var, modeled_var, njobs=-2):
        """
        Parameters
        ---------------------------------------------------------------
        obs: :py:class:`~xarray.DataArray`, required
            A baseline gridded low resolution observed dataset. This should include
            high quality gridded observations. lat and lon are expected as dimensions.
        modeled_present: :py:class:`~xarray.DataArray`, required
            A gridded low resolution climate variable to train the bias correction. This may include
            reanalysis or GCM datasets for the present climate. 
            The lat and lon dimensions are same as obs.
        modeled_future: :py:class:`~xarray.DataArray`, required
            A gridded low resolution climate variable to be corrected. This may include
            GCM datasets for the future climate. 
            The lat and lon dimensions are same as obs.
        obs_var: str, required
            The variable name in dataset obs which to model
        modeled_var: str, required
            The variable name in Dataset modeled which to bias correct
        njobs: int, optional
            The number of processes to execute in parallel
        """
        # Select intersecting time perids
        #d1 = obs.time.values
        #d2 = modeled.time.values
        #intersection = np.intersect1d(d1, d2)
        #obs = obs.loc[dict(time=intersection)]
        #modeled = modeled.loc[dict(time=intersection)]

        #dayofyear = obs['time.dayofyear']
        dayofyear = np.tile(np.arange(365)+1, 16)    # This is a special case when using datasets for 16 years without leap years.
        lat_vals = modeled_future.lat.values
        lon_vals = modeled_future.lon.values

        # initialize the output data array
        if lat_vals.ndim == 1:
             mapped_data = np.zeros(shape=(modeled_future.time.values.shape[0], lat_vals.shape[0], 
                                       lon_vals.shape[0]))
        elif lat_vals.ndim == 2:
            mapped_data = np.zeros(shape=(modeled_future.time.values.shape[0], lat_vals.shape[0], 
                                      lat_vals.shape[1]))
        # loop through each day of the year, 1 to 365
        nday = 365
        #for day in np.unique(dayofyear.values):
        for day in np.arange(nday)+1:
            t1 = time.time()
            print "Day = %i/365" % day
            # select days +- pool
            dayrange = (np.arange(day-self.pool, day+self.pool+1) + nday) % nday+ 1
            days = np.in1d(dayofyear, dayrange)
            subobs = obs.loc[dict(time=days)]
            submodeled_present = modeled_present.loc[dict(time=days)]
            submodeled_future = modeled_future.loc[dict(time=days)]

            # which rows correspond to these days
            #sub_curr_day_rows = np.where(day == subobs['time.dayofyear'].values)[0]
            #curr_day_rows = np.where(day == obs['time.dayofyear'].values)[0]
            sub_curr_day_rows = np.where(day == dayofyear[days])[0]
            curr_day_rows = np.where(day == dayofyear)[0]
            train_num = np.where(subobs['time.year'] <= self.max_train_year)[0][-1]
            mapped_times = submodeled_future['time'].values[sub_curr_day_rows]

            jobs = [] # list to collect jobs
            for iy in np.arange(obs.dims['y']):
                X_lat = subobs.sel(y=iy, method='nearest')[obs_var].values
                X_lat = ma.masked_where(X_lat > 1.e+3, X_lat)
                X_lat = ma.filled(X_lat, 9999.)
                Y_lat = submodeled_present.sel(y=iy)[modeled_var].values
                Z_lat = submodeled_future.sel(y=iy)[modeled_var].values
                jobs.append(delayed(mapper)(X_lat, Y_lat, Z_lat, train_num, self.step))
            print "Running parallel jobs (number of latitudes)", len(jobs)
            # select only those days which correspond to the current day of the year
            day_mapped = np.asarray(Parallel(n_jobs=njobs)(jobs))[:, sub_curr_day_rows]
            day_mapped = np.swapaxes(day_mapped, 0, 1)
            mapped_data[curr_day_rows, :, :] = day_mapped
            print 'execution time to correct biases for one day:', time.time()-t1, 'seconds'
        # put data into a data array
        dr = xray.DataArray(mapped_data, coords=[modeled_future['time'].values, obs.coords['y'], obs.coords['x']],
                       dims=['time', 'y', 'x'])
        dr.attrs['gridtype'] = 'latlon'
        ds = xray.Dataset({'bias_corrected': dr}) 
        ds = ds.reindex_like(modeled_future)
        modeled_future = modeled_future.merge(ds) # merging aids in preserving netcdf structure
        # delete modeled variable to save space
        del modeled_future[modeled_var]
        return modeled_future
