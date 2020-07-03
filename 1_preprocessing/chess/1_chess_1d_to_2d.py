#!/usr/bin/env python

import sys
import os

# The control file should look like something like this:
'''
# start year:
2015

# end year:
2017

# path to folder with input 1D CHESS data
/prj/hydrojules/data/soil_moisture/preprocessed/chess/chess_1d/

# path to folder with output 2D CHESS data
/prj/hydrojules/data/soil_moisture/preprocessed/chess/chess_2d/
'''


input_file = open(sys.argv[1])
# skip first line
input_file.readline()

# next line is the start year
start_year = int(input_file.readline())

# skip next 2 lines
input_file.readline()
input_file.readline()

# next line is the end year
end_year = int(input_file.readline())

# skip next 2 lines
input_file.readline()
input_file.readline()

# next line is the input path
inFolder = input_file.readline()[:-1]

# skip next 2 lines
input_file.readline()
input_file.readline()

# next line is the output path
outFolder = input_file.readline()[:-1]


#----------------------------------------------------------
# USER DEFINED VARIABLES:
#----------------------------------------------------------

#start_year=2015
#end_year=2017

#inFolder='/prj/hydrojules/data/soil_moisture/preprocessed/chess/chess_1d/'
#outFolder = '/prj/hydrojules/data/soil_moisture/preprocessed/chess/chess_2d/'

#----------------------------------------------------------
# END OF USER DEFINED VARIABLES
# DO NOT EDIT BEYOND THIS POINT
#----------------------------------------------------------

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy._crs import Globe


from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from matplotlib.ticker import MultipleLocator

OSGB_orig = ccrs.OSGB().proj4_params
OSGB = ccrs.TransverseMercator(central_longitude=OSGB_orig['lon_0'],
                               central_latitude=OSGB_orig['lat_0'],
                               scale_factor=OSGB_orig['k'],
                               false_easting=OSGB_orig['x_0'],
                               false_northing=OSGB_orig['y_0'],
                               globe=Globe(datum='OSGB36', ellipse='airy'))


################################################################################
################################################################################
#
#
#
# ELR 26-02-2019
#
################################################################################
################################################################################

grid_north_pole_latitude = 39.25
grid_north_pole_longitude = 198.0


################################################################################
# Parse the input
################################################################################


################################################################################
# Read the pdef file
################################################################################
def read_pdef(pdefname, nx, ny, bswap=False):

    # read the index (4byte ints)
    indx = np.fromfile(pdefname, dtype='int32', count=nx*ny)
    indx.byteswap(bswap)
    indx = indx.astype('float32')
    indx = indx.reshape([ny, nx])

    # read the weights (floats)
    wt = np.fromfile(pdefname, dtype='float32', count=-1)
    wt = wt[nx*ny:(2*nx*ny)]
    wt.byteswap(bswap)
    wt = wt.reshape([ny, nx])

    return indx,wt

################################################################################
# Reshape the vector to the full grid
################################################################################
def land_to_grid(data, indx, wt, missingval):

    outdata = np.ones(indx.shape)*missingval

    iy, ix = np.where(indx != missingval)

    for i in range(len(ix)):
        outdata[iy[i], ix[i]] = data[int(indx[iy[i], ix[i]]) -1]

    outdata = np.ma.masked_equal(outdata, missingval)
    outdata = outdata[::-1, :]

    return outdata

################################################################################
# Get data and convert from 1D to 2D
################################################################################
for year in range(start_year,end_year+1):
    print "Transforming 1D CHESS data into 2D CHESS data for year " + str(year) + "\n"
    #datafile = '/prj/hydrojules/data/soil_moisture/preprocessed/chess/chess_2016.nc'
    datafile = inFolder + "chess_v1.1.soil_daily."+str(year)+".nc"   #'/group_workspaces/jasmin2/hydro_jules_OLD/data/uk/jules_outputs/chess/chess_v1.1_update.soil_daily.2016.nc'

    datavar= 'smcl'
    pdef = str(os.getcwd()) + '/CHESS_pdef_jasmin.gra'   ###/home/users/jpeng006/CHESS_pdef_jasmin.gra
    nx = 656
    ny=1057

    # Open netCDF file
    f = nc.Dataset(datafile, 'r')
    # extract data
    plotvector = f.variables[datavar][:]
    time_n =f.variables['time'][:]


    # Convert from vector to grid
    indx, wt = read_pdef(pdef, nx, ny, bswap=False)

    # Extract the grid coordinates
    eastings = np.arange(nx)*1000.0 + 500.0
    northings = np.arange(ny)*1000.0 + 500.0
    #lon = f.variables['lon'][:]
    #lat = f.variables['lat'][:]
    gx, gy = np.meshgrid(eastings, northings)


    transform = ccrs.PlateCarree().transform_points(OSGB, gx, gy)
    rlon = transform[:,:,0]
    rlat = transform[:,:,1]

    ##creat new 2D data

    dataset = nc.Dataset(outFolder+'chess_v1.1.soil_daily.'+str(year)+'_2d.nc','w',format='NETCDF4')

    # create dimensions
    dataset.createDimension('lat',rlat.shape[0]*rlon.shape[1])
    dataset.createDimension('lon',rlon.shape[1]*rlat.shape[0])
    dataset.createDimension('y',rlat.shape[0])
    dataset.createDimension('x',rlon.shape[1])
    dataset.createDimension('time',len(time_n))

    # create variables

    east = dataset.createVariable("x",np.double, ["x"])
    east.units = 'm'
    east.long_name = 'easting - OSGB36 grid reference'
    east.standard_name = 'projection_x_coordinate'
    east.point_spacing = "even"

    east[:] = eastings
    #lats = np.arange(xini*5000+2500,xlast*5000+2500,5000)

    north = dataset.createVariable("y",np.double, ["y"])
    north.units = 'm'
    north.long_name = 'northing - OSGB36 grid reference'
    north.standard_name = 'projection_y_coordinate'
    north.point_spacing = "even"
    #lats = np.arange(xini*5000+2500,xlast*5000+2500,5000)

    north[:] = northings

                ##################from 2015.04.01--2017.10.12############

    lats = dataset.createVariable('lat',np.double,["y","x"],zlib=True)
    lons = dataset.createVariable('lon',np.double,["y","x"],zlib=True)
    times_out = dataset.createVariable('time',np.double,('time',),zlib=True)
    sw_bsa = dataset.createVariable('sm',np.double,('time','y','x',),fill_value=-999.,zlib=True)



    if year<=2015:
        times_out.units = 'seconds since 1961-01-01 00:00:00'
    else:
        times_out.units = 'seconds since '+str(year)+'-01-01 00:00:00'
    times_out.calendar = 'standard'
    times_out.long_name = 'time'

    sw_bsa.units = 'm3/m3'
    sw_bsa.long_name = 'soil moisture'

    lats.standard_name = 'latitude'
    lats.units = 'degrees_north'
    lons.units = 'degrees_east'
    lons.standard_name = 'longitude'


    #lats[:] = rlat[:,0]
    #lons[:] = rlon[0,:]
    realLat = np.load('Lat_1km.npy')
    realLon = np.load('Lon_1km.npy')
    lats[:] = realLat
    lons[:] = realLon
    times_out[:] = time_n[:]

    coord = dataset.createVariable("crs",np.int16)
    coord.long_name = 'coordinate_reference_system'
    coord.grid_mapping_name = "transverse_mercator"
    coord.semi_major_axis = 6377563.396
    coord.semi_minor_axis = 6356256.910
    coord.inverse_flattening = 299.3249646
    coord.latitude_of_projection_origin = 49.0
    coord.longitude_of_projection_origin = -2.0
    coord.false_easting = 400000.0
    coord.false_northing = -100000.0
    coord.scale_factor_at_projection_origin = 0.9996012717
    coord.EPSG_code = "EPSG:27700"

    for i in range(len(time_n)):
        plotve = plotvector[i,0,0,:]
        plotdata = land_to_grid(plotve, indx, wt, -999)[::-1,:]
        sw_bsa[i,:]=plotdata[:]

    dataset.grid_mapping = 'crs'
    dataset.geospatial_lat_min = np.nanmin(realLat)
    dataset.geospatial_lat_max = np.nanmax(realLat)
    dataset.geospatial_lon_min = np.nanmin(realLon)
    dataset.geospatial_lon_max = np.nanmax(realLon)
    dataset.spatial_resolution_distance = float(1000)
    dataset.spatial_resolution_unit = 'urn:ogc:def:uom:EPSG::9001'


    # Close the file
    f.close()
    dataset.close()
    # change the permissions
    os.chmod(outFolder+'chess_v1.1.soil_daily.'+str(year)+'_2d.nc',0664)



















