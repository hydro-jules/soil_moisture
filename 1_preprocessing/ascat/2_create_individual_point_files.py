#-------------------------------------------------------------------------------
# convert H115 file into individual csv file per point
#-------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from netCDF4 import Dataset
import os
from os import listdir
from os.path import isfile, join
import sys

# Function to convert from netcdf date to datetime date
##def netcdf2datetime(t,netcdfUnits):
def netcdf2datetime(t):
	import netcdftime
	import datetime
	cdftime = netcdftime.utime(netcdfUnits)
	date = cdftime.num2date(t)
	dateText = date.isoformat()
	return dateText


# The control file should look like something like this:
'''
# path to folder with ASCAT raw data
/prj/hydrojules/data/soil_moisture/preprocessed/ascat/h115/GB/
'''

input_file = open(sys.argv[1])
# skip first line
input_file.readline()

# next line is input folder  (with H115 data)
inFolder = input_file.readline()[:-1]

listCell = ['1252','1253','1288','1289','1324']
#listCell = ['1290']

# Directory with H115 data:
#inFolder = '/prj/hydrojules/data/soil_moisture/preprocessed/ascat/h115/GB/'
locFolder = inFolder + 'locFolder/'
timeSeriesFolder = inFolder + 'timeSeriesFolder/'

# Create sub-directories if they don't exist
if not os.path.exists(locFolder):
    os.mkdir(locFolder)
else:
    dummy = 0

if not os.path.exists(timeSeriesFolder):
    os.mkdir(timeSeriesFolder)
else:
    dummy = 0

for cell in listCell:
    # create file to save locations coordinates
    locFile = open(locFolder + 'location_'+cell+'.csv','w')
    locFile.write('Location,Lat,Lon\n')
    # read in dimensions and variables from Netcdf file
    nc = Dataset(inFolder + 'H115_' + cell + '.nc','r')
    loc = nc.dimensions['locations']
    obs = nc.dimensions['obs']
    lat = nc.variables['lat'][:]
    lon = nc.variables['lon'][:]
    sm = nc.variables['sm'][:]
    time = nc.variables['time']
    netcdfUnits = time.units
    time= time[:]
    row_size = nc.variables['row_size'][:]

    current_obs = 0
    # loop through locations
    for i in range(len(loc)):
        name = cell + '_' + str(i)
        print name
        sm_array = []
        time_array = []
        row_size_loc = row_size[i]
        locFile.write(name + ',' + str(lat[i]) +','+ str(lon[i])+'\n')
        for j in range(row_size_loc):
            sm_array.append(sm[current_obs])
            time_array.append(time[current_obs])
            current_obs += 1
        time_newUnits = map(netcdf2datetime,time_array)
        dic4df = {'Time':time_newUnits,'sm':sm_array}
        df = pd.DataFrame(dic4df)
        df.to_csv(timeSeriesFolder + name + '.csv',index=False)

    locFile.close()








