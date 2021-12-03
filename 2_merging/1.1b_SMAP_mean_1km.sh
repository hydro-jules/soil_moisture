#!/bin/sh

# Script to calculate temporal mean of SMAP data (1km version)
 
# Seconde line of the control file is the path to the target folder
targetFolder=$(sed '2q;d' $1)

# Calculate temporal SMAP mean
cdo -b F64 -timmean -setmissval,nan ${targetFolder}smap_1km.nc ${targetFolder}smap_1km_mean.nc

chmod 664 ${targetFolder}smap_1km_mean.nc
