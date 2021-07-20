#-------------------------------------------------------------------------------
# Create daily mean time series. Subset for period of interest
# April 2015 to Dec 2017
#-------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from netCDF4 import Dataset
import os
from os import listdir
from os.path import isfile, join
import glob
import datetime
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
start_date_text = input_file.readline()[:-1]

# skip next 2 lines
input_file.readline()
input_file.readline()

# next line is the end year
end_date_text = input_file.readline()[:-1]

# skip next 2 lines
input_file.readline()
input_file.readline()

# next line is the input path
inParentFolder = input_file.readline()[:-1]

# define start and end dates
start_date_array = start_date_text.split('-')
start_year = int(start_date_array[0])
start_month = int(start_date_array[1])
start_day = int(start_date_array[2])

end_date_array = end_date_text.split('-')
end_year = int(end_date_array[0])
end_month = int(end_date_array[1])
end_day = int(end_date_array[2])

start_date = datetime.datetime(start_year,start_month,start_day,0,0,0)
end_date = datetime.datetime(end_year,end_month,end_day,23,59,59)

listCell = ['1252','1253','1288','1289','1324','1290']
#listCell = ['1290']

inFolder = inParentFolder+'timeSeriesFolder/'
outFolder = inFolder + 'dailyMean/'

# Create sub-directories if they don't exist
if not os.path.exists(outFolder):
    os.mkdir(outFolder)
else:
    dummy = 0


for cell in listCell:
    onlyfiles = glob.glob(inFolder + cell + '_*.csv')
    for myFile in onlyfiles:
        fileName = myFile.split('/')[-1]
        print fileName
        df = pd.read_csv(myFile)
        df['Time']= pd.to_datetime(df['Time'])
        mask = (df['Time'] > start_date) & (df['Time'] <= end_date)
        df = df.loc[mask]
        df.set_index('Time', inplace=True)
        try:
            df['sm'] = df['sm'].str.replace('--','NaN')
        except:
            print "could not df['sm'] = df['sm'].str.replace('--','NaN')"
        df['sm'] = df['sm'].astype(float)
        means = df.groupby(pd.Grouper(freq='1D')).mean()

        means.to_csv(outFolder + fileName,index=True)






