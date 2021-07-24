#-------------------------------------------------------------------------------
# Mask out NI in SMAP data using CHESS
#-------------------------------------------------------------------------------
version = 'v1'
import datetime
import numpy as np
import pdb
import os, os.path
import sys
import glob
import netCDF4 as nc
from numpy import inf


infolder = '/prj/hydrojules/data/soil_moisture/preprocessed/data_for_review/'

chess_file = infolder + 'chess_12.5km_march2016.nc'
dataset_chess = nc.Dataset(chess_file,'r')
lat = np.squeeze(np.array(dataset_chess.variables['lat']))
lon = np.squeeze(np.array(dataset_chess.variables['lon']))
tmp_date = np.asarray(dataset_chess.variables['time'])
chess = np.squeeze(np.array(dataset_chess.variables['sm']))
chess = chess/100
##chess = np.ma.masked_where((chess<0)|(chess>1), chess)
##chess=np.ma.filled(chess.astype(float), np.nan)

smap_file = infolder + 'smap_12.5km_march2016_with_cdo.nc'
dataset_smap = nc.Dataset(smap_file,'r')
smap = np.squeeze(np.array(dataset_smap.variables['sm']))
smap[np.isposinf(smap)] = -999
smap[np.isneginf(smap)] = -999
smap[np.isnan(smap)] = -999
##smap = np.ma.masked_where((smap<0)|(smap>1), smap)
##smap=np.ma.filled(smap.astype(float), np.nan)




dataset = nc.Dataset(infolder + 'masked_smap_12.5km_march2016_'+version+'.nc','w',format='NETCDF4')

# create dimensions
dataset.createDimension('lat',len(lat[:]))
dataset.createDimension('lon',len(lon[:]))
dataset.createDimension('time',len(tmp_date))
# create variables
lats = dataset.createVariable('lat',np.float32,('lat',))
lons = dataset.createVariable('lon',np.float32,('lon',))
times_out = dataset.createVariable('time',np.float32,('time',))

#sm_cal = dataset.createVariable('sm',np.float32,('time','lat','lon',),fill_value=9.96921e+36,zlib=True)
sm_cal = dataset.createVariable('sm',np.float32,('time','lat','lon',),fill_value=-999,zlib=True)

# create Attributes
times_out.units = 'hours since 1970-01-01 00:00:00.0'
times_out.calendar = 'gregorian'
times_out.long_name = 'time'

sm_cal.units = 'm3/m3'
sm_cal.long_name = 'Soil moisture'

lats.standard_name = 'latitude'
lats.units = 'degrees_north'
lons.units = 'degrees_east'
lons.standard_name = 'longitude'


smap= np.where((chess<0)| (chess>1),-999, smap)
smap= np.where(np.isnan(chess), -999, smap)
smap = np.ma.masked_where((smap<0) | (smap>1),smap)
smap=np.ma.filled(smap.astype(float), np.nan)

sm_cal[:] = smap[:]
lats[:]= lat[:]
lons[:]=lon[:]
times_out[:] = tmp_date[:]
# close file
dataset.close()
os.chmod(infolder + 'masked_smap_12.5km_march2016_'+version+'.nc',0664)
