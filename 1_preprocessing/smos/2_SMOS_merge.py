#!/usr/bin/env python

#-------------------------------------------------------------------------------
# Code to merge SMOS dataset into one large dataset
# Author: Maliko Tanguy (malngu@ceh.ac.uk) based on Jian Peng's code
# Date: 15/06/2020
#-------------------------------------------------------------------------------

import sys
import os

# The control file should look like something like this:
'''
# start date (format YYYY-MM-DD):
2015-01-01

# end date (format YYYY-MM-DD):
2017-12-31

# Main input folder with NetCDF files
/prj/hydrojules/data/soil_moisture/preprocessed/smos/smos_nc/

# Output folder
/prj/hydrojules/data/soil_moisture/preprocessed/smos/smos_merged/
'''

input_file = open(sys.argv[1])
# skip first line
input_file.readline()

# next line is the start year
start_date = input_file.readline()[:-1]

# skip next 2 lines
input_file.readline()
input_file.readline()

# next line is the end year
end_date = input_file.readline()[:-1]

# skip next 2 lines
input_file.readline()
input_file.readline()

# next line is the input path
roodirc = input_file.readline()[:-1]

# skip next 2 lines
input_file.readline()
input_file.readline()

# next line is the output path
outFolder = input_file.readline()[:-1]



#########sample dataset to get lat/lon for targeted file

import pandas as pd
import netCDF4 as nc
import numpy as np
import datetime
import glob
import gdal
import osr

#get the lat/long from a sample file
filelists = glob.glob(roodirc+'*.nc')
filelists = sorted(filelists)
firstFile = filelists[0]

sm_dataset =nc.Dataset(firstFile,'r')
lat = np.asarray(sm_dataset.variables['lat'])
lon = np.asarray(sm_dataset.variables['lon'])
sm_dataset.close()

start=datetime.datetime.strptime(start_date,'%Y-%m-%d')
end=datetime.datetime.strptime(end_date,'%Y-%m-%d')
dates  = pd.date_range(start,end,freq='1D')
bla = dates[:]
totDays=len(bla)

#####create target file
dataset = nc.Dataset(outFolder+'smos_all_uk.nc','w',format='NETCDF4')
# create dimensions
dataset.createDimension('lat',lat.shape[0])
dataset.createDimension('lon',lon.shape[0])
dataset.createDimension('time',totDays)                      ##################from 2015.04.01--2017.10.12############
# create variables
lats = dataset.createVariable('lat',np.float32,('lat',),zlib=True)
lons = dataset.createVariable('lon',np.float32,('lon',),zlib=True)
times_out = dataset.createVariable('time',np.float32,('time',),zlib=True)
sw_bsa = dataset.createVariable('sm',np.float32,('time','lat','lon',),fill_value=-999.,zlib=True)


# create Attributes
times_out.units = 'hours since 1970-01-01 00:00:00.0'
times_out.calendar = 'gregorian'
times_out.long_name = 'time'

sw_bsa.units = 'Degree of saturation'
sw_bsa.long_name = 'Soil moisture'

lats.standard_name = 'latitude'
lats.units = 'degrees_north'
lons.units = 'degrees_east'
lons.standard_name = 'longitude'
lats[:] = lat[:]
lons[:] = lon[:]


########set the shape of the files and copy each file to part of the merged file
#date_start = pd.datetime(2015,04,01)
#date_end   = pd.datetime(2017,10,12)
#dates  = pd.date_range(date_start,date_end,freq='1D')
#bla = dates[:]

i =0
for date_in in bla:
    print date_in

    years = date_in.year
    months = date_in.month
    days =date_in.day
    tmp_date = datetime.datetime(int(years), int(months), int(days))

    # create  timestamp
    times_out[i] = nc.date2num(tmp_date,units=times_out.units,calendar=times_out.calendar)
    datestr=date_in.strftime('%Y') + date_in.strftime('%m') + date_in.strftime('%d')
    filelists = glob.glob(roodirc+'BEC_SM____SMOS__EUM_L4__A_'+datestr+'*_001km_1d_REP_v5.0.nc')
    file_chess=filelists[0]

    #file_chess =roodirc + 'chess_reprojected_'+date_in.strftime('%Y') + date_in.strftime('%m') + date_in.strftime('%d')+'.nc'

    if os.path.isfile(file_chess) == True:
        sm_data =nc.Dataset(file_chess,'r')
        sm_chess1 = np.asarray(sm_data.variables['SM'])

        sm_chess2 = np.ma.masked_where(sm_chess1==-999,sm_chess1)
        sm_chess = np.ma.masked_where(sm_chess2<=0,sm_chess2)
        sw_bsa[i,:] = sm_chess

    i = i+1
# close file
dataset.close()
os.chmod(outFolder+'smos_all_uk.nc',0664)
