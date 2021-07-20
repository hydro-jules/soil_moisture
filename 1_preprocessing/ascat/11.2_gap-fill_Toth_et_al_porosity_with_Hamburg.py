#-------------------------------------------------------------------------------
# check how much diff with hamburg data
#-------------------------------------------------------------------------------

# there is the choice between two bulk density dataset 'T_BULK_DEN' or 'T_REF_BULK'
bulk_den = 'T_BULK_DEN'
#bulk_den = 'T_REF_BULK'
# equation with Organic matter content: yes (True) or no (False)
##oc_eq = True
##if oc_eq==True:
##    oc_text = 'with_oc'
##else:
##    oc_text = 'without_oc'

import numpy as np
import pandas as pd
from netCDF4 import Dataset, date2num
import os
from os import listdir
from os.path import isfile, join
import glob
import datetime


import sys

# The control file should look like something like this:
'''
# path to 12.5km CHESS file
/prj/hydrojules/data/soil_moisture/preprocessed/chess/chess_2d/merged/

# path to folder with HWSD data
/prj/hydrojules/data/soil_moisture/preprocessed/ascat/HWSD/
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

porosity_file = HWSD_folder+'porosity_Toth_et_al_with_'+bulk_den+'_12.5km.nc'
hamburg_file = HWSD_folder+'ASCAT_Hamburg_porosity_12.5km.nc'
outFolder = HWSD_folder
chess_file = chess_folder + 'chess_12.5km.nc'

#chess will be used to determine the land mask
dataset_chess = Dataset(chess_file,'r')
chess = np.squeeze(np.array(dataset_chess.variables['sm']))[0,:,:]
#chess should be converted from percent to m3/m3
chess = chess/100
chess = np.ma.masked_where((chess<0)|(chess>1), chess)

# porosity dataset
porosity_dataset = Dataset(porosity_file,'r')
porosity = np.squeeze(np.array(porosity_dataset.variables['porosity']))

#read lat/lon data
lat = np.squeeze(np.array(porosity_dataset.variables['lat']))
lon = np.squeeze(np.array(porosity_dataset.variables['lon']))

dataset_hamburg =  Dataset(hamburg_file,'r')
hamburg = np.squeeze(np.array(dataset_hamburg.variables['HWSD_porosity']))
hamburg = np.ma.masked_where(hamburg<=0, hamburg)

porosity = np.where(porosity<0, hamburg, porosity)
porosity = np.where(np.isnan(porosity), hamburg, porosity)

porosity = np.ma.masked_where(np.isnan(chess), porosity)

#------------------------------------------------------------
# configurating netcdf file
regular_lat = lat
regular_lon = lon

# create output dataset
dataset = Dataset(outFolder+'porosity_Toth_et_al_with_'+bulk_den+'_GAPFILLED_12.5km.nc','w',format='NETCDF4')
dataset.createDimension('lat',len(regular_lat))
dataset.createDimension('lon',len(regular_lon))


lats = dataset.createVariable('lat',np.float32,('lat',),zlib=True)
lons = dataset.createVariable('lon',np.float32,('lon',),zlib=True)
sw_bsa = dataset.createVariable('porosity',np.float32,('lat','lon',),fill_value=-999.,zlib=True)

sw_bsa.units = 'm3/m3'
sw_bsa.long_name = 'Porosity (gapfilled with Hamburg version of HWSD)'

lats.standard_name = 'latitude'
lats.units = 'degrees_north'
lons.units = 'degrees_east'
lons.standard_name = 'longitude'
lats[:] = regular_lat
lons[:] = regular_lon
#-------------------------------------------------------------

sw_bsa[:] = porosity