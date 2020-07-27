#!/bin/sh

# Script to regrid CHESS to 1km grids
# USAGE: 6b_CHESS_regridding_1km.sh <control_file_name>
# The control file must have the path to the merged CHESS netCDF 
# file on the second line

# Second line of control file is the path to the target folder
targetFolder=$(sed '2q;d' $1)

cat > mygrid_1km << EOF
gridtype = latlon
xsize    = 910
ysize    = 945
xfirst   = -7.54
xinc     = 0.01
yfirst   = 59.21
yinc     = -0.01
EOF

cdo sellonlatbox,-7.55,1.55,49.77,59.21
cdo remapnn,mygrid_1km ${targetFolder}chess_all_uk.nc ${targetFolder}chess_1km.nc
chmod 664 ${targetFolder}chess_1km.nc
chmod 664 mygrid_1km

