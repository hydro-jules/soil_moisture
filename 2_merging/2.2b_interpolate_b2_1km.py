#-------------------------------------------------------------------------------
# Interpolate scaling factor b2 (CHESS) to gap-fill.
# Nearest neighbour interpolation.
#
# Author:      Maliko Tanguy (malngu@ceh.ac.uk)
# Created:     14/12/2020
#-------------------------------------------------------------------------------

import datetime
import numpy as np
import pdb
import os, os.path
import sys
import glob
import pytesmo.scaling as scaling
import pytesmo.metrics as metrics
import netCDF4 as nc
from scipy import stats


# The control file should look like something like this:
'''
# path to folder with chess file:
/prj/hydrojules/data/soil_moisture/preprocessed/chess/chess_2d/merged/

# path to folder with the scaling factor b2:
/prj/hydrojules/data/soil_moisture/merged/weights/
'''

input_file = open(sys.argv[1])
# skip first line
input_file.readline()

# next line is the start year
chess_folder = input_file.readline()[:-1]

# skip next 2 lines
input_file.readline()
input_file.readline()

# next line is the output path
merged_folder = input_file.readline()[:-1]

#merged_folder = '/prj/hydrojules/data/soil_moisture/merged/Paper_review_2/'
chess_file = chess_folder + 'chess_1km.nc'
b2_file = merged_folder + 'scaling_factor_b2_1km.nc'

dataset_chess = nc.Dataset(chess_file,'r')
chess = np.squeeze(np.array(dataset_chess.variables['sm']))
chess=chess/100
chess = np.ma.masked_where((chess<0)|(chess>1), chess)

dataset_b2 = nc.Dataset(b2_file,'r')
lat = np.squeeze(np.array(dataset_b2.variables['lat']))
lon = np.squeeze(np.array(dataset_b2.variables['lon']))
a = np.squeeze(np.array(dataset_b2.variables['b2']))

index = np.where(np.logical_and(a>=0, a<=100))
##indexT=[]
##for i in range(len(index[0])):
##    indexT.append([index[1][i],index[0][i]])

print index
values = a[index]
print values

##values = []
##for j in range(len(indexT)):
##    values.append(a[indexT[j][1],indexT[j][0]])
##valuesNP = np.array(values)


##*******************************************************
##DOESN'T WORK
##from scipy.spatial import KDTree
##x,y=np.mgrid[0:a.shape[0],0:a.shape[1]]
##
##xygood = np.array((x[~a.mask],y[~a.mask])).T
##xybad = np.array((x[a.mask],y[a.mask])).T
##
##a[a.mask] = a[~a.mask][KDTree(xygood).query(xybad)[1]]
##*******************************************************

from scipy.interpolate import griddata
grid_y, grid_x = np.mgrid[0:len(lat), 0:len(lon)]
#grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
#grid_z0 = griddata(indexT, valuesNP, (grid_x, grid_y), method='nearest')
grid_z0 = griddata(index, values, (grid_y, grid_x), method='nearest')

dataset = nc.Dataset(merged_folder + 'b2_interpolated_1km.nc','w',format='NETCDF4')
# create dimensions
dataset.createDimension('lat',len(lat[:]))
dataset.createDimension('lon',len(lon[:]))

# create variables
lats = dataset.createVariable('lat',np.float32,('lat',))
lons = dataset.createVariable('lon',np.float32,('lon',))

sm_cal = dataset.createVariable('b2',np.float32,('lat','lon',),fill_value=-999.,zlib=True)

# create Attributes
sm_cal.units = 'unitless'
sm_cal.long_name = 'scaling factor b2'

lats.standard_name = 'latitude'
lats.units = 'degrees_north'
lons.units = 'degrees_east'
lons.standard_name = 'longitude'

grid_z0 = np.ma.masked_where(np.isnan(chess[0,:,:]), grid_z0)

sm_cal[:] = grid_z0[:]
lats[:]= lat[:]
lons[:]=lon[:]
# close file
dataset.close()
os.chmod(merged_folder + 'b2_interpolated_1km.nc',0664)
