
#############################fill the gaps of TC merged product with naive average of chess, smos and smap data############################
import datetime
import numpy as np
import pdb
import os, os.path
import sys
import glob
import netCDF4 as nc

# The control file should look like something like this:
'''
# path to folder with chess file:
/prj/hydrojules/data/soil_moisture/preprocessed/chess/chess_2d/merged/

# path to folder with smap file:
/prj/hydrojules/data/soil_moisture/preprocessed/smap/smap_merged/

# path to folder with smos file:
/prj/hydrojules/data/soil_moisture/preprocessed/smos/smos_merged/

# path to folder with merged file (and where gap-filled file will be created):
/prj/hydrojules/data/soil_moisture/merged/
'''

input_file = open(sys.argv[1])
# skip first line
input_file.readline()

# next line is the start year
chess_folder = input_file.readline()[:-1]

# skip next 2 lines
input_file.readline()
input_file.readline()

# next line is the end year
smap_folder = input_file.readline()[:-1]

# skip next 2 lines
input_file.readline()
input_file.readline()

# next line is the input path
smos_folder = input_file.readline()[:-1]

# skip next 2 lines
input_file.readline()
input_file.readline()

# next line is the output path
merged_folder = input_file.readline()[:-1]


merge_file = merged_folder + 'merge_9km_tc.nc'
dataset_merge = nc.Dataset(merge_file,'r')
merge = np.squeeze(np.array(dataset_merge.variables['sm']))
lat = np.squeeze(np.array(dataset_merge.variables['lat']))
lon = np.squeeze(np.array(dataset_merge.variables['lon']))
tmp_date = np.asarray(dataset_merge.variables['time'])

chess_file = chess_folder + 'chess_9km.nc'
dataset_chess = nc.Dataset(chess_file,'r')
chess = np.squeeze(np.array(dataset_chess.variables['sm']))
chess = chess/100
chess = np.ma.masked_where((chess<0)|(chess>1), chess)

smap_file = smap_folder + 'smap_9km.nc'
dataset_smap = nc.Dataset(smap_file,'r')
smap = np.squeeze(np.array(dataset_smap.variables['sm']))
smap = np.ma.masked_where((smap<0)|(smap>1), smap)

smos_file = smos_folder + 'smos_9km.nc'
dataset_smos = nc.Dataset(smos_file,'r')
smos = np.squeeze(np.array(dataset_smos.variables['sm']))
smos = np.ma.masked_where((smos<0)|(smos>1), smos)

chess=np.ma.filled(chess.astype(float), np.nan)
smap=np.ma.filled(smap.astype(float), np.nan)
smos=np.ma.filled(smos.astype(float), np.nan)


dataset = nc.Dataset(merged_folder + 'merge_9km_tc_no_gap.nc','w',format='NETCDF4')

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

###simply average the soil moisture for pixels that have no values###
merge_mean = np.nanmean([chess,smap,smos],0)
merge_mean = np.ma.masked_where((merge_mean<0)|(merge_mean>1)|(np.isnan(chess)), merge_mean)
merge_mean=np.ma.filled(merge_mean.astype(float), np.nan)

sm_fill= np.ma.masked_where((merge<=0)| (merge>=0.99), merge)
sm_fill=np.ma.filled(sm_fill.astype(float), np.nan)

#####fill the gaps of tc merged SM product####################
sm_fill[np.isnan(sm_fill)] =merge_mean[np.isnan(sm_fill)]

sm_cal[:] = sm_fill[:]
lats[:]= lat[:]
lons[:]=lon[:]
times_out[:] = tmp_date[:]
# close file
dataset.close()
os.chmod(merged_folder + 'merge_9km_tc_no_gap.nc',0664)











