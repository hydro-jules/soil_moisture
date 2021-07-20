#!/usr/bin/env python

#-------------------------------------------------------------------------------
# Code to convert SMAP GeoTIFF data to netcdf
# Author: Maliko Tanguy (malngu@Ceh.ac.uk)
#-------------------------------------------------------------------------------

import sys
import os

# The control file should look like something like this:
'''
# Main input folder (there should be subfolders 'smap_AM' and 'smap_PM' in it)
/prj/hydrojules/data/soil_moisture/preprocessed/smap/smap_tif/

# Main output folder (subfolders 'smap_AM' and 'smap_PM' will be created if they
don't already exist
/prj/hydrojules/data/soil_moisture/preprocessed/smap/smap_nc/
'''


input_file = open(sys.argv[1])
# skip first line
input_file.readline()


# next line is the input path
roodirc = input_file.readline()[:-1]

# skip next 2 lines
input_file.readline()
input_file.readline()

# next line is the output path
outdir = input_file.readline()[:-1]



import netCDF4 as nc
import numpy as np
import datetime
import glob
import gdal
import osr

#roodirc = '/prj/hydrojules/data/soil_moisture/preprocessed/smap/smap_tif/' ###define the directory where you store your downloaded SMAP 9k geotiff files
#outdir = '/prj/hydrojules/data/soil_moisture/preprocessed/smap/smap_nc/'

# create output subdirectories if they don't exist
if not os.path.exists(outdir+'smap_AM/'):
    os.makedirs(outdir+'smap_AM/')
if not os.path.exists(outdir+'smap_PM/'):
    os.makedirs(outdir+'smap_PM/')

# list of data files for AM and PM (+ first file of AM data to get extension, etc.)
filelistsAM = glob.glob(roodirc+'smap_AM/*.tif')
filelistsAM = sorted(filelistsAM)
firstFile = filelistsAM[0]

filelistsPM = glob.glob(roodirc+'smap_PM/*.tif')
filelistsPM = sorted(filelistsPM)
#print firstFile



#get the lat/long from a sample file, any downloaded file is fine
#dssub = gdal.Open(roodirc+'SMAP_L3_SM_P_E_20150401_R16510_001_Soil_Moisture_Retrieval_Data_PM_soil_moisture_pm_120ff055.tif')
dssub = gdal.Open(firstFile)
#'SMAP_L3_SM_P_E_20170901_R16510_001_Soil_Moisture_Retrieval_Data_PM_soil_moisture_pm_12289b1e.tif'
#'SMAP_L3_SM_P_E_20200222_R16515_001_Soil_Moisture_Retrieval_Data_PM_soil_moisture_pm_7c89ce11.tif'
data = dssub.ReadAsArray()
gt = dssub.GetGeoTransform()
# create output array with grid information
x = np.empty(np.shape(data)[1])
y= np.empty(np.shape(data)[0])
for k in range(np.shape(data)[1]):
    x[k] = gt[0]+k*gt[1]
for k in range(np.shape(data)[0]):
    y[k] = gt[3]+k*gt[5]
# get current grid information
old_cs= osr.SpatialReference()
old_cs.ImportFromWkt(dssub.GetProjectionRef())
# create the new coordinate system
wgs84_wkt = """
GEOGCS["WGS 84",
    DATUM["WGS_1984",
        SPHEROID["WGS 84",6378137,298.257223563,
            AUTHORITY["EPSG","7030"]],
        AUTHORITY["EPSG","6326"]],

    PRIMEM["Greenwich",0,
        AUTHORITY["EPSG","8901"]],
    UNIT["degree",0.01745329251994328,
        AUTHORITY["EPSG","9122"]],
    AUTHORITY["EPSG","4326"]]"""
new_cs = osr.SpatialReference()
new_cs .ImportFromWkt(wgs84_wkt)
# create a transform object to convert between coordinate systems
transform = osr.CoordinateTransformation(old_cs,new_cs)
xmesh,ymesh =np.meshgrid(x,y)
#get the coordinates in lat long
lat = np.empty(np.shape(xmesh))
lon = np.empty(np.shape(xmesh))
for k in range(np.shape(xmesh)[0]):
    for p in range(np.shape(xmesh)[1]):
       lon[k,p],lat[k,p],tempdel = transform.TransformPoint(xmesh[k,p],ymesh[k,p])


##read each file and save it, you mush go to the directory
#filelists = glob.glob(roodirc+'*.tif')

#print filelists

# LOOP for AM
print '\nLOOP FOR AM SOIL MOISTURE\n'
for s in filelistsAM:

    ds = gdal.Open(s)
    band = ds.GetRasterBand(1)
    sm_tif = band.ReadAsArray()
    #print sm_tif[30:35,30:35]
    name=s.split('/')[-1:] [0]
    datestring=name.split('_')[5]
    #print datestring
    years = datestring[0:4]
    #print years
    months = datestring[4:6]
    #print months
    days =datestring[6:8]
    #print days
    print years+'-'+months+'-'+days

    tmp_date = datetime.datetime(int(years), int(months), int(days))

    # dataset = nc.Dataset('test25.nc','w',format='NETCDF4')
    dataset = nc.Dataset(outdir+'smap_AM/SMAP_L3_SM_P_E_'+datestring+'_smap_am.nc','w',format='NETCDF4')
    # create dimensions
    dataset.createDimension('lat',lat.shape[0])
    dataset.createDimension('lon',lon.shape[1])
    dataset.createDimension('time',None)
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
    sw_bsa.long_name = 'Sentinel-1 soil moisture'

    lats.standard_name = 'latitude'
    lats.units = 'degrees_north'
    lons.units = 'degrees_east'
    lons.standard_name = 'longitude'
    lats[:] = lat[:,0]
    lons[:] = lon[0,:]
    sw_bsa[0,:] = sm_tif[:,:]

    # create  timestamp
    times_out[:] = nc.date2num(tmp_date,units=times_out.units,calendar=times_out.calendar)
    # close file
    dataset.close()
    os.chmod(outdir+'smap_AM/SMAP_L3_SM_P_E_'+datestring+'_smap_am.nc',0664)


# LOOP for PM
print '\nLOOP FOR PM SOIL MOISTURE\n'
for s in filelistsPM:

    ds = gdal.Open(s)
    band = ds.GetRasterBand(1)
    sm_tif = band.ReadAsArray()

    name=s.split('/')[-1:] [0]
    datestring=name.split('_')[5]
    #print datestring
    years = datestring[0:4]
    #print years
    months = datestring[4:6]
    #print months
    days =datestring[6:8]
    #print days
    print years+'-'+months+'-'+days

    tmp_date = datetime.datetime(int(years), int(months), int(days))

    # dataset = nc.Dataset('test25.nc','w',format='NETCDF4')
    dataset = nc.Dataset(outdir+'smap_PM/SMAP_L3_SM_P_E_'+datestring+'_smap_pm.nc','w',format='NETCDF4')
    # create dimensions
    dataset.createDimension('lat',lat.shape[0])
    dataset.createDimension('lon',lon.shape[1])
    dataset.createDimension('time',None)
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
    sw_bsa.long_name = 'Sentinel-1 soil moisture'

    lats.standard_name = 'latitude'
    lats.units = 'degrees_north'
    lons.units = 'degrees_east'
    lons.standard_name = 'longitude'
    lats[:] = lat[:,0]
    lons[:] = lon[0,:]
    sw_bsa[0,:] = sm_tif[:,:]

    # create  timestamp
    times_out[:] = nc.date2num(tmp_date,units=times_out.units,calendar=times_out.calendar)
    # close file
    dataset.close()
    os.chmod(outdir+'smap_PM/SMAP_L3_SM_P_E_'+datestring+'_smap_pm.nc',0664)
