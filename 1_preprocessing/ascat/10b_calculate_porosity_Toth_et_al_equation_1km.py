#-------------------------------------------------------------------------------
# calculate soil porosity using Toth et al. (2015) pedo transfer functions.
# The one suitable is eq. 22 from supplementary info. https://doi.org/10.1111/ejss.12192
# Saturated soil moisture content (~= porosity) = 0.63052 - 0.10262 * BD^2 + 0.0002904 * pH^2 + 0.0003335 * Cl
#-------------------------------------------------------------------------------

# there is the choice between two bulk density dataset 'T_BULK_DEN' or 'T_REF_BULK'
bulk_den = 'T_BULK_DEN'
#bulk_den = 'T_REF_BULK'

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
# path to HWSD data
/prj/hydrojules/data/soil_moisture/preprocessed/ascat/HWSD/
'''

input_file = open(sys.argv[1])
# skip first line
input_file.readline()

# path to HWSD 1km file
HWSDfolder = input_file.readline()[:-1]

bulk_density_file = HWSDfolder+bulk_den+'_1km.nc'
ph_file = HWSDfolder+'T_PH_H2O_1km.nc'
clay_file = HWSDfolder+'T_CLAY_1km.nc'
outFolder = HWSDfolder

dataset_bulk_density = Dataset(bulk_density_file,'r')
bulk_density = np.squeeze(np.array(dataset_bulk_density.variables[bulk_den]))

bulk_density = np.ma.masked_where(bulk_density<0, bulk_density)

#read lat/lon data
lat = np.squeeze(np.array(dataset_bulk_density.variables['lat']))
lon = np.squeeze(np.array(dataset_bulk_density.variables['lon']))

dataset_clay = Dataset(clay_file,'r')
#clay = np.divide(np.squeeze(np.array(dataset_clay.variables['T_CLAY'])),100)
clay = np.squeeze(np.array(dataset_clay.variables['T_CLAY']))

dataset_ph = Dataset(ph_file,'r')
ph = np.squeeze(np.array(dataset_ph.variables['T_PH_H2O']))

# Equation for porosity
porosity = 0.63052 - 0.10262 * (bulk_density*bulk_density) + 0.0002904 * (ph*ph) + 0.0003335 * clay

porosity_masked = np.ma.masked_where(np.isnan(bulk_density), porosity)

#------------------------------------------------------------
# configurating netcdf file
regular_lat = lat
regular_lon = lon

# create output dataset
dataset = Dataset(outFolder+'porosity_Toth_et_al_with_'+bulk_den+'_1km.nc','w',format='NETCDF4')
dataset.createDimension('lat',len(regular_lat))
dataset.createDimension('lon',len(regular_lon))


lats = dataset.createVariable('lat',np.float32,('lat',),zlib=True)
lons = dataset.createVariable('lon',np.float32,('lon',),zlib=True)
sw_bsa = dataset.createVariable('porosity',np.float32,('lat','lon',),fill_value=-999.,zlib=True)

sw_bsa.units = 'm3/m3'
sw_bsa.long_name = 'Porosity'

lats.standard_name = 'latitude'
lats.units = 'degrees_north'
lons.units = 'degrees_east'
lons.standard_name = 'longitude'
lats[:] = regular_lat
lons[:] = regular_lon
#-------------------------------------------------------------

sw_bsa[:] = porosity_masked