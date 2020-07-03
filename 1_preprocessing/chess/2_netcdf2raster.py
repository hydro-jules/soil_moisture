# Code to convert CHESS netcdf files to geotiff format (to perform the reprojection)
# Maliko Tanguy (malngu@ceh.ac.uk) - 24/06/2020

import sys
import os

# The control file should look like something like this:
'''
# start year:
2015

# end year:
2017

# path to folder with input 2D NetCDF CHESS files
/prj/hydrojules/data/soil_moisture/preprocessed/chess/chess_2d/

# path to folder with output GeoTiff CHESS files
/prj/hydrojules/data/soil_moisture/preprocessed/chess/chess_2d/geotiff/
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


def netCDF2Raster(netFile,doy,x_size,y_size,x_min,y_max,outName,NoDataValue=-999,spatRef='None'):

	import numpy
	import osr
	from netCDF4 import Dataset
	from osgeo import gdal as osgdal
	#import grid

	"""
	Converts netCDF data to raster so that]
	GDAL library can be applied
	netFile -- path and name of the netCDF file
	day -- day of year
	x_size -- grid size in x axis
	y_size -- grid size in y axis
	"""

	readNet = Dataset(netFile, "r")

	varRainfall = readNet.variables["sm"]
	days,ytot,xtot = numpy.shape(varRainfall)

	crs = readNet.variables["crs"]

	driver = osgdal.GetDriverByName('GTiff')

	dataset = driver.Create(outFolder + outName +'.tif',xtot,ytot,1,osgdal.GDT_Float32)

	#dataset.SetGeoTransform((x_min,x_size,0,y_max,0,-y_size))
	dataset.SetGeoTransform((x_min,1000,0,y_max,0,-1000))

	out_srs = osr.SpatialReference()

	if spatRef=='None':
		out_srs.ImportFromEPSG(int(crs.EPSG_code.split(':')[1]))
	else:
		out_srs.ImportFromEPSG(int(spatRef))
	#print out_srs.ExportToWkt()

	dataset.SetProjection(out_srs.ExportToWkt())

	band = dataset.GetRasterBand(1)
	band.WriteArray(varRainfall[doy,::-1,:])
	band.SetNoDataValue(NoDataValue)

	dataset = None
	os.chmod(outFolder + outName +'.tif',0664)
	#print 'ok1'
	#stats = grid.loop_zonal_stats('/prj/lwis/maliko/python/test_shape.shp', '/prj/lwis/maliko/python/test.tif')
	#print 'ok2'
	#return stats



# 17/11/2010 = doy 321
if __name__ == "__main__":

    import datetime
    start_date = datetime.datetime(start_year,01,01)
    end_date = datetime.datetime(end_year,12,31)
    current_day = start_date
    diff =  end_date - start_date
    diffDays = diff.days
    for day in range(diffDays+1):
        textDate = current_day.isoformat().split("T")[0]
        isoYear = textDate.split("-")[0]
        isoMonth = textDate.split("-")[1]
        isoDay = textDate.split("-")[2]
        textDate2 = isoYear +'-'+ isoMonth +'-'+ isoDay
        year = isoYear
        # The following three lines don't work for dates pre-1900
        #textDate = current_day.strftime("%Y-%m-%d")
        #textDate2 = current_day.strftime("%Y%m%d")
        #year=current_day.strftime("%Y")
        DOY1=current_day-datetime.datetime(int(year),01,01)
        DOY=DOY1.days
        print 'Creating raster for '+textDate+'...'
        netFile = inFolder + "chess_v1.1.soil_daily."+year+"_2d.nc"
        #month = DOY
        #print year
        x_size = 656
        y_size = 1057
        x_min = 0
        y_max = 1057000
        outName = 'chess-'+textDate2
        netCDF2Raster(netFile,DOY,x_size,y_size,x_min,y_max,outName,spatRef='27700')
        current_day = current_day + datetime.timedelta(days=1)





