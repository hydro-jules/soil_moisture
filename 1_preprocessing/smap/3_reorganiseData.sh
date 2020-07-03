#!/bin/sh

targetFolder="/prj/hydrojules/data/soil_moisture/preprocessed/smap/smap_tif/"
newFolder1=$targetFolder"smap_error/"
newFolder2=$targetFolder"smap_AM/"
newFolder3=$targetFolder"smap_PM/"
mkdir -p $newFolder1
mkdir -p $newFolder2
mkdir -p $newFolder3

cd $targetFolder
echo -e "Moving error files"
echo -e "\n"
mv *_error_*.tif ./smap_error/

echo -e "Moving Soil Moisture AM files"
echo -e "\n"
mv *_AM_soil_moisture_*.tif ./smap_AM/

echo -e "Moving Soil Moisture PM files"
echo -e "\n"
mv *_PM_soil_moisture_pm_*.tif ./smap_PM/   

files1=$newFolder1"*.tif"
files2=$newFolder2"*.tif"
files3=$newFolder3"*.tif"

chmod 664 $files1
chmod 664 $files2
chmod 664 $files3
