#!/bin/sh

# Script to regrid Hamburg porosity to 1km grids
# USAGE: Hamburg_regridding.sh <control_file_name>
# The control file must have the path to the mHamburg netCDF 
# file on the second line

# Second line of control file is the path to the target folder
targetFolder=$(sed '2q;d' $1)

cat > mygrid_1km << EOF
gridtype = latlon
xsize    = 945
ysize    = 945
xfirst   = -7.54
xinc     = 0.01
yfirst   = 59.21
yinc     = -0.01
EOF

cdo sellonlatbox,-7.55,1.55,49.77,59.21
cdo remapnn,mygrid_1km ${targetFolder}ASCAT_Hamburg_porosity.nc ${targetFolder}ASCAT_Hamburg_porosity_1km.nc
chmod 664 ${targetFolder}ASCAT_Hamburg_porosity_1km.nc
chmod 664 mygrid_1km

