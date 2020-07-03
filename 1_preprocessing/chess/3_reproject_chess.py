#-------------------------------------------------------------------------------
# Reproject chess data using arcpy
#-------------------------------------------------------------------------------

start_year = 2015
end_year = 2017

##====================================
##Project Raster
##Usage: ProjectRaster_management in_raster out_raster out_coor_system {NEAREST | BILINEAR
##                                | CUBIC | MAJORITY} {cell_size} {geographic_transform;
##                                geographic_transform...} {Registration_Point} {in_coor_system}

import arcpy
import os

#workFolder="W:\\MALNGU\\Hydrojules\\data\\chess\\"
#workFolder='\\\\nercwlsmb01\\prj\\hydrojules\\data\\soil_moisture\\preprocessed\\chess\\chess_2d\\geotiff\\NEW_TEST\\'
workFolder="C:\\chess_tif\\"
arcpy.env.workspace = workFolder
arcpy.env.overwriteOutput = True
if not os.path.exists(workFolder+"temp\\"):
    os.makedirs(workFolder+"temp\\")

# You can find list of geographic transformations here:
# http://help.arcgis.com/en/arcgisdesktop/10.0/help/003r/pdf/geographic_transformations.pdf
# or here:
# https://desktop.arcgis.com/en/arcmap/latest/map/projections/pdf/geographic_transformations.pdf
geographic_transform='OSGB_1936_To_WGS_1984_1'
#in_coor_system = arcpy.SpatialReference("W:\MALNGU\Hydrojules\data\chess\prj\27700.prj")
in_coor_system = arcpy.SpatialReference(27700)
#out_coor_system = arcpy.SpatialReference("W:\MALNGU\Hydrojules\data\chess\prj\4326.prj")
out_coor_system = arcpy.SpatialReference(4326)

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
    outName = 'chess-'+textDate2
    year = isoYear
    # The following three lines don't work for dates pre-1900
    #textDate = current_day.strftime("%Y-%m-%d")
    #textDate2 = current_day.strftime("%Y%m%d")
    #year=current_day.strftime("%Y")
    DOY1=current_day-datetime.datetime(int(year),01,01)
    DOY=DOY1.days
    print 'Creating reprojected raster for '+textDate+'...'

    ##Reproject a TIFF image with Datumn transfer
    arcpy.ProjectRaster_management(workFolder+outName +'.tif', workFolder+"temp\\"+outName+"_reproject.tif",out_coor_system ,\
                                   "BILINEAR",'#', geographic_transform,'#',in_coor_system )
    current_day = current_day + datetime.timedelta(days=1)


