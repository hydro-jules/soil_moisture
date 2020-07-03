#-------------------------------------------------------------------------------
# Code to merge all SMAP netcdf files into one large one. It also averages AM
# and PM soil moisture data.
# Author: Maliko Tanguy (malngu@ceh.ac.uk) -  based on Jian Peng's code
# Date: 16/06/2020
#-------------------------------------------------------------------------------

import sys
import os
import netCDF4 as nc
import numpy as np
import datetime
import glob
import gdal
import osr

# The control file should look like something like this:
'''
# start date (format YYYY-MM-DD):
2015-01-01

# end date (format YYYY-MM-DD):
2017-12-31

# Main input folder (there should be subfolders 'smap_AM' and 'smap_PM' in it)
/prj/hydrojules/data/soil_moisture/preprocessed/smap/smap_tif/

# Main output folder (subfolders 'smap_AM' and 'smap_PM' will be created if they
don't already exist
/prj/hydrojules/data/soil_moisture/preprocessed/smap/smap_nc/
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
outdir = input_file.readline()[:-1]



# list of data files for AM and PM (+ first file of AM data to get extension, etc.)
filelistsAM = glob.glob(roodirc+'smap_AM/*.nc')
filelistsAM = sorted(filelistsAM)
firstFile = filelistsAM[0]


import pandas as pd

sm_dataset =nc.Dataset(firstFile,'r')
lat = np.asarray(sm_dataset.variables['lat'])
lon = np.asarray(sm_dataset.variables['lon'])
sm_dataset.close()

start=datetime.datetime.strptime(start_date,'%Y-%m-%d')
end=datetime.datetime.strptime(end_date,'%Y-%m-%d')
dates  = pd.date_range(start,end,freq='1D')
bla = dates[:]
totDays=len(bla)

#####creat target file
dataset = nc.Dataset(outdir+'smap_am_pm_all_uk.nc','w',format='NETCDF4')
# create dimensions
dataset.createDimension('lat',lat.shape[0])
dataset.createDimension('lon',lon.shape[0])
dataset.createDimension('time',totDays)                      ##################from 2015.04.01--2017.10.12############
# create variables
lats = dataset.createVariable('lat',np.float32,('lat',),zlib=True)
lons = dataset.createVariable('lon',np.float32,('lon',),zlib=True)
times_out = dataset.createVariable('time',np.float32,('time',),zlib=True)
sw_bsa = dataset.createVariable('sm',np.float32,('time','lat','lon',),fill_value=-9999.,zlib=True)


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

    years = date_in.year
    months = date_in.month
    days =date_in.day
    tmp_date = datetime.datetime(int(years), int(months), int(days))
    print str(years)+'-'+str(months)+'-'+str(days)

    # create  timestamp
    times_out[i] = nc.date2num(tmp_date,units=times_out.units,calendar=times_out.calendar)


    file_sm_am =roodirc+'smap_AM/SMAP_L3_SM_P_E_'+ date_in.strftime('%Y') + date_in.strftime('%m') + date_in.strftime('%d')+'_smap_am.nc'   ######################file name need to be changed
    #print file_sm_am
    file_sm_pm =roodirc+'smap_PM/SMAP_L3_SM_P_E_'+date_in.strftime('%Y') + date_in.strftime('%m') + date_in.strftime('%d')+'_smap_pm.nc'   ######################file name need to be changed
    #print file_sm_pm

    if (os.path.isfile(file_sm_am) == True) & (os.path.isfile(file_sm_pm) == True):
        am_data =nc.Dataset(file_sm_am,'r')
        sm_am = np.asarray(am_data.variables['sm'])
        #sm_am_ma=np.ma.masked_where(sm_am==-9999,sm_am)
        sm_am[sm_am==-9999]=np.nan
        #print sm_am.shape

        pm_data =nc.Dataset(file_sm_pm,'r')
        sm_pm = np.asarray(pm_data.variables['sm'])
        #sm_pm_ma=np.ma.masked_where(sm_pm==-9999,sm_pm)
        sm_pm[sm_pm==-9999]=np.nan
        #print sm_pm.shape

        sw_bsa[i,:] = np.nanmean([sm_am,sm_pm],0)[:,:]
        #print sw_bsa[i,:]

    elif (os.path.isfile(file_sm_am) == True) & (os.path.isfile(file_sm_pm) != True):
        am_data =nc.Dataset(file_sm_am,'r')
        sm_am = np.asarray(am_data.variables['sm'])
        sw_bsa[i,:] = sm_am[:,:]

    elif (os.path.isfile(file_sm_am) != True) & (os.path.isfile(file_sm_pm) == True):
        pm_data =nc.Dataset(file_sm_pm,'r')
        sm_pm = np.asarray(pm_data.variables['sm'])
        sw_bsa[i,:] = sm_pm[:,:]

    i = i+1
# close file
dataset.close()
os.chmod(outdir+'smap_am_pm_all_uk.nc',0664)