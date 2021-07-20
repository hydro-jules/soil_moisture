#-------------------------------------------------------------------------------
# create summary file with list of grids + lat and lon list
#-------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from netCDF4 import Dataset
import os
from os import listdir
from os.path import isfile, join
import glob
import datetime
import pickle
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
inParentFolder = input_file.readline()[:-1]

inFolder = inParentFolder + 'locFolder/'
#listCell = ['1252','1253','1288','1289','1324','1290']
listCell = ['1252','1253','1288','1289','1324']

# create summary location file
print 'Summary location file'
summaryFile = open(inFolder + 'summary_location.csv','w')
summaryFile.write('Location,Lat,Lon\n')

for cell in listCell:
    inFile = open(inFolder + 'location_'+cell+'.csv','r')
    # skip first line
    inFile.readline()
    text = inFile.read()
    summaryFile.write(text)
    inFile.close()
summaryFile.close()

# create lat and lon summary files
print 'Creating Lat Lon files'
df = pd.read_csv(inFolder + 'summary_location.csv')
lat = sorted(list(set(df['Lat'].tolist())))
lon = sorted(list(set(df['Lon'].tolist())))

with open(inFolder + 'Lat.pkl', 'wb') as f:
    pickle.dump(lat, f)

with open(inFolder + 'Lon.pkl', 'wb') as f:
    pickle.dump(lon, f)

