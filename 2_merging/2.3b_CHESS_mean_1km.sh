#!/bin/sh

# Script to calculate temporal mean of CHESS data (1km version)
 
# Seconde line of the control file is the path to the target folder
targetFolder=$(sed '2q;d' $1)

# Calculate temporal CHESS mean
cdo -b F64 -timmean -setmissval,nan ${targetFolder}chess_1km.nc ${targetFolder}chess_1km_mean.nc

chmod 664 ${targetFolder}chess_1km_mean.nc
