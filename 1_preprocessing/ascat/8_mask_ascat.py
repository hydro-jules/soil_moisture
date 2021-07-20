#-------------------------------------------------------------------------------
# mask out sea.
# Use CHESS file to mask out non-land pixels
#-------------------------------------------------------------------------------

# interpolation method: 'linear' or 'nearest'.
# 'cubic' not suitable because it gives values < 0.
method = 'linear'

import numpy as np
import pandas as pd
from netCDF4 import Dataset, date2num
import os
from os import listdir
from os.path import isfile, join
import glob
import datetime
import pickle

import sys

# The control file should look like something like this:
'''
# path to 12.5km CHESS file
/prj/hydrojules/data/soil_moisture/preprocessed/chess/chess_2d/merged/chess_12.5km.nc

# path to folder with ASCAT data
/prj/hydrojules/data/soil_moisture/preprocessed/ascat/h115/GB/gridded_data/

# path to output folder:
/prj/hydrojules/data/soil_moisture/preprocessed/ascat/h115/GB/gridded_data/
'''


input_file = open(sys.argv[1])
# skip first line
input_file.readline()

# path to CHESS 12.5km file
chess_file = input_file.readline()[:-1]

# skip next 2 lines
input_file.readline()
input_file.readline()

# next line is the end year
ascat_folder = input_file.readline()[:-1]

# skip next 2 lines
input_file.readline()
input_file.readline()

# next line is the input path
outFolder = input_file.readline()[:-1]

#chess_file = '/prj/hydrojules/data/soil_moisture/preprocessed/data_for_review/chess_12.5km.nc'
ascat_file = ascat_folder + 'ascat_12.5km_'+method+'.nc'
#outFolder = '/prj/hydrojules/data/soil_moisture/preprocessed/ascat/h115/GB/gridded_data/'

dataset_chess = Dataset(chess_file,'r')
chess = np.squeeze(np.array(dataset_chess.variables['sm']))
#chess should be converted from percent to m3/m3
chess = chess/100
chess = np.ma.masked_where((chess<0)|(chess>1), chess)


dataset_ascat = Dataset(ascat_file,'r')
ascat = np.squeeze(np.array(dataset_ascat.variables['sm']))
#read lat/lon data
lat = np.squeeze(np.array(dataset_ascat.variables['lat']))
lon = np.squeeze(np.array(dataset_ascat.variables['lon']))
tmp_date = np.asarray(dataset_ascat.variables['time'])

ascat_tc = np.ma.masked_where(np.isnan(chess), ascat)

#------------------------------------------------------------
# configurating netcdf file
regular_lat = lat
regular_lon = lon

# create output dataset
dataset = Dataset(outFolder+'ascat_masked_12.5km_'+method+'.nc','w',format='NETCDF4')
dataset.createDimension('lat',len(regular_lat))
dataset.createDimension('lon',len(regular_lon))
dataset.createDimension('time',len(tmp_date))

lats = dataset.createVariable('lat',np.float32,('lat',),zlib=True)
lons = dataset.createVariable('lon',np.float32,('lon',),zlib=True)
times_out = dataset.createVariable('time',np.float32,('time',),zlib=True)
sw_bsa = dataset.createVariable('sm',np.float32,('time','lat','lon',),fill_value=-999.,zlib=True)

# create Attributes
times_out.units = 'hours since 1970-01-01 00:00:00.0'
times_out.calendar = 'gregorian'
times_out.long_name = 'time'
times_out[:] = tmp_date[:]

sw_bsa.units = 'Degree of saturation'
sw_bsa.long_name = 'Soil moisture'

lats.standard_name = 'latitude'
lats.units = 'degrees_north'
lons.units = 'degrees_east'
lons.standard_name = 'longitude'
lats[:] = regular_lat
lons[:] = regular_lon
#-------------------------------------------------------------

sw_bsa[:] = ascat_tc