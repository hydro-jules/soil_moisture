#-------------------------------------------------------------------------------
# make one massive file with all data from all locations
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


locFolder = inParentFolder+'locFolder/'
dataFolder = inParentFolder+'timeSeriesFolder/dailyMean/'

df = pd.read_csv(dataFolder + '1252_0.csv')
df['Time']= pd.to_datetime(df['Time'])
df.set_index('Time', inplace=True)
df.rename(columns={'sm':"1252_0"})


# load location summary file
dfLoc = pd.read_csv(locFolder + 'summary_location.csv')
for ind in dfLoc.index:
    location = dfLoc['Location'][ind]
    if location!='1252_0':
        print location
        dfNew = pd.read_csv(dataFolder + location + '.csv')
        dfNew.set_index('Time', inplace=True)
        df[location] = dfNew['sm'].values
df.to_csv(locFolder + 'massive_csv.csv',index=True)

