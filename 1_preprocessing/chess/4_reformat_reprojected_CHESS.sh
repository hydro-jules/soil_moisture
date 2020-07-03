#!/bin/sh

# Script to convert reprojected chess tif files into netcdf

# start and end date from control file (2nd and 5th line)
input_start=$(sed '2q;d' $1)
input_end=$(sed '5q;d' $1)

# input and output file from control file (8th and 11th line)
targetFolder1=$(sed '8q;d' $1)
targetFolder2=$(sed '11q;d' $1)

# start and end date
#input_start=2015-1-1
#input_end=2015-1-3

# After this, startdate and enddate will be valid ISO 8601 dates,
# or the script will have aborted when it encountered unparseable data
# such as input_end=abcd
startdate=$(date -I -d "$input_start") || exit -1
enddate=$(date -I -d "$input_end")     || exit -1
enddate=$(date -I -d "$enddate + 1 day")

#targetFolder1="/prj/hydrojules/data/soil_moisture/preprocessed/chess/chess_2d/geotiff/temp/"
#targetFolder2="/prj/hydrojules/data/soil_moisture/preprocessed/chess/chess_2d/reprojected/"

d="$startdate"
while [ "$d" != "$enddate" ]; do 
  echo -e '\n'
  echo $d
  DAY=$(date -d "$d" '+%d')
  MONTH=$(date -d "$d" '+%m')
  YEAR=$(date -d "$d" '+%Y')
  inFile=${targetFolder1}"chess-${YEAR}-${MONTH}-${DAY}_reproject.tif"
  outFile=${targetFolder2}"chess-${YEAR}-${MONTH}-${DAY}_reproject.nc"
    
  echo "Input file is: ${inFile}"
  echo "Output file is: ${outFile}"
  gdal_translate -of netCDF -co "FORMAT=NC4" $inFile $outFile
  chmod 664 $outFile
  d=$(date -I -d "$d + 1 day")
done





