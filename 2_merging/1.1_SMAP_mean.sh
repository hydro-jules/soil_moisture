#!/bin/sh

# Script to calculate temporal mean of SMAP data (12.5km version)
 
# Seconde line of the control file is the path to the target folder
targetFolder=$(sed '2q;d' $1)

# Calculate temporal SMAP mean
cdo -b F64 -timmean -setmissval,nan ${targetFolder}smap_12.5km.nc ${targetFolder}smap_12.5km_mean.nc

chmod 664 ${targetFolder}smap_12.5km_mean.nc
