#-------------------------------------------------------------------------------
# produce interpolated grids from time series
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

from scipy.interpolate import griddata

import sys

# The control file should look like something like this:
'''
# start date:
2015-04-01

# end date:
2017-12-31

# path to folder with ASCAT raw data
/prj/hydrojules/data/soil_moisture/preprocessed/ascat/h115/GB/
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
inParentFolder = input_file.readline()[:-1]


# start date (format YYYY-MM-DD):
#start_date = '2015-04-01'

# end date (format YYYY-MM-DD):
#end_date = '2017-10-12'

locFolder = inParentFolder + 'locFolder/'
outFolder = inParentFolder + 'gridded_data/'

# Create sub-directories if they don't exist
if not os.path.exists(outFolder):
    os.mkdir(outFolder)
else:
    dummy = 0

# load location summary file
dfLoc = pd.read_csv(locFolder + 'summary_location.csv')
lat = dfLoc['Lat'].tolist()
lon = dfLoc['Lon'].tolist()
points = []
points.append(lat)
points.append(lon)
points = np.array(points)
#print[points]

# load SM data
massiveDF = pd.read_csv(locFolder + 'massive_csv.csv')
massiveDF['Time']= pd.to_datetime(massiveDF['Time'])
massiveDF.set_index('Time', inplace=True)

# define start and end date:
start=datetime.datetime.strptime(start_date,'%Y-%m-%d')
end=datetime.datetime.strptime(end_date,'%Y-%m-%d')
dates  = pd.date_range(start,end,freq='1D')
bla = dates[:]
totDays=len(bla)

grid_lat, grid_lon = np.mgrid[49:60:88j, -8:2:80j]

#------------------------------------------------------------
# configurating netcdf file
regular_lat = np.arange(49,60,0.125)
regular_lon = np.arange(-8,2,0.125)

# create output dataset
dataset = Dataset(outFolder+'ascat_h115_'+method+'.nc','w',format='NETCDF4')
dataset.createDimension('lat',len(regular_lat))
dataset.createDimension('lon',len(regular_lon))
dataset.createDimension('time',totDays)

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
lats[:] = regular_lat
lons[:] = regular_lon
#-------------------------------------------------------------

i=0
# Loop through dates
for date_in in bla:
    print date_in
    years = date_in.year
    months = date_in.month
    days =date_in.day
    tmp_date = datetime.datetime(int(years), int(months), int(days))

    # create  timestamp
    times_out[i] = date2num(tmp_date,units=times_out.units,calendar=times_out.calendar)

    # get all SM values for this date
    values = massiveDF.loc[date_in].values
    # interpolation using griddata function
    grid_z0 = griddata(points.T, values, (grid_lat, grid_lon), method=method)
    sw_bsa[i,:] = grid_z0[:]
    i = i+1





