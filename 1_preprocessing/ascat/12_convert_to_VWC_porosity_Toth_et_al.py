#-------------------------------------------------------------------------------
# Convert ASCAT degree of saturation to VWC (degree Sat x porosity)
#-------------------------------------------------------------------------------

# interpolation method: 'linear' or 'nearest'.
# 'cubic' not suitable because it gives values < 0.
method = 'linear'

# there is the choice between two bulk density dataset 'T_BULK_DEN' or 'T_REF_BULK'
bulk_den = 'T_BULK_DEN'
#bulk_den = 'T_REF_BULK'

# equation with Organic matter content: yes (True) or no (False)
##oc_eq = True
##if oc_eq==True:
##    oc_text = 'with_oc'
##else:
##    oc_text = 'without_oc'
### there is also the option to use GLDAS porosity
##GLDAS = False
##
### Or the option to use Hamburg version of HWSD
##hamburg = False
##
##if (hamburg==True) & (GLDAS==True):
##    print '\nWARNING: Choose your porosity dataset correctly.\n GLDAS will used.\n'

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
/prj/hydrojules/data/soil_moisture/preprocessed/chess/chess_2d/merged/

# path to folder with HWSD data
/prj/hydrojules/data/soil_moisture/preprocessed/ascat/HWSD/

# path to folder with ASCAT data
/prj/hydrojules/data/soil_moisture/preprocessed/ascat/h115/GB/gridded_data/
'''


input_file = open(sys.argv[1])
# skip first line
input_file.readline()

# path to CHESS 12.5km file
chess_folder = input_file.readline()[:-1]

# skip next 2 lines
input_file.readline()
input_file.readline()

# path to HWSD folder
HWSD_folder = input_file.readline()[:-1]

# skip next 2 lines
input_file.readline()
input_file.readline()

# path to HWSD folder
ascat_folder = input_file.readline()[:-1]

ascat_file = ascat_folder+'ascat_masked_12.5km_'+method+'.nc'

#porosity_file = '/prj/hydrojules/data/soil_moisture/preprocessed/data_for_review/HWSD_1247/data/'+'porosity_with_'+bulk_den+'_' + oc_text + '_12.5km.nc'
porosity_file = HWSD_folder+'porosity_Toth_et_al_with_'+bulk_den+'_GAPFILLED_12.5km.nc'
por_var = 'porosity'
factor = 1.0
outname_root = method+'_'+bulk_den
outFolder = ascat_folder
chess_file = chess_folder+'chess_12.5km.nc'

dataset_ascat = Dataset(ascat_file,'r')
ascat = np.squeeze(np.array(dataset_ascat.variables['sm']))

#read lat/lon data
lat = np.squeeze(np.array(dataset_ascat.variables['lat']))
lon = np.squeeze(np.array(dataset_ascat.variables['lon']))
tmp_date = np.asarray(dataset_ascat.variables['time'])

dataset_porosity = Dataset(porosity_file,'r')
porosity_base = np.squeeze(np.array(dataset_porosity.variables[por_var]))
porosity = np.repeat(porosity_base.T[:, :, np.newaxis], len(tmp_date), axis=2).T

##print ascat.shape
##print porosity.shape

# calculate vwc
ascat_vwc = ascat * (porosity / factor) / 100.0
ascat_vwc = np.where((ascat_vwc<0)|(ascat_vwc>1),-999, ascat_vwc)

# mask sea using chess file
dataset_chess = Dataset(chess_file,'r')
chess = np.squeeze(np.array(dataset_chess.variables['sm']))
#chess should be converted from percent to m3/m3
chess = chess/100
chess = np.ma.masked_where((chess<0)|(chess>1), chess)

ascat_vwc_masked = np.ma.masked_where(np.isnan(chess), ascat_vwc)
ascat_vwc_masked = np.ma.masked_where(ascat_vwc_masked<0, ascat_vwc_masked)

#------------------------------------------------------------
# configurating netcdf file
regular_lat = lat
regular_lon = lon

# create output dataset
dataset = Dataset(outFolder+'ascat_VWC_' + outname_root + '_porosity_Toth_et_al_12.5km.nc','w',format='NETCDF4')
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

sw_bsa.units = 'm3/m3'
sw_bsa.long_name = 'Soil moisture - Volumetric water content'

lats.standard_name = 'latitude'
lats.units = 'degrees_north'
lons.units = 'degrees_east'
lons.standard_name = 'longitude'
lats[:] = regular_lat
lons[:] = regular_lon
#-------------------------------------------------------------

sw_bsa[:] = ascat_vwc_masked