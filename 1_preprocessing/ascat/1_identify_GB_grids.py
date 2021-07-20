#-------------------------------------------------------------------------------
# Check which files are within the extension of GB
# Use Grid point locator: https://dgg.geo.tuwien.ac.at/ to have a first guess
#-------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import netCDF4 as nc
import os
from os import listdir
from os.path import isfile, join
from shutil import copyfile
import sys

# The control file should look like something like this:
'''
# path to folder with ASCAT raw data
/prj/hydrojules/data/soil_moisture/preprocessed/ascat/h115/GB/
'''

input_file = open(sys.argv[1])
# skip first line
input_file.readline()

# next line is input folder
inFolder = input_file.readline()[:-1]

# Directory with H115 data:
#inFolder = '/prj/hydrojules/data/soil_moisture/preprocessed/ascat/h115/GB/'
#inFolder = '/scratch/malngu/soil_moisture/new_data_and_code/'
onlyfiles = sorted([ f for f in listdir(inFolder) if isfile(join(inFolder,f)) ])

# Extensions for GB
minLat = 49
maxLat = 60
minLon = -8
maxLon = 2

# Loop through files
for myFile in onlyfiles:
    #print myFile
    ncFile = nc.Dataset(inFolder + myFile)
    lat = ncFile.variables['lat'][:]
    lon = ncFile.variables['lon'][:]
    # a is a dummy variable
    if (np.amin(lat)>maxLat) or (np.amax(lat)<minLat):
        #copyfile(inFolder + myFile, inFolder + 'not_GB/' + myFile)
        a=1
    else:
        if (np.amin(lon)>maxLon) or (np.amax(lon)<minLon):
            #copyfile(inFolder + myFile, inFolder + 'not_GB/' + myFile)
            a=2
        else:
            #copyfile(inFolder + myFile, inFolder + 'GB/' + myFile)
            print 'lat and lon within range for '+myFile




