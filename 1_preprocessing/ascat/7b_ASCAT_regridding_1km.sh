#!/bin/sh

# Script to regrid ASCAT to 12.5km grids
# USAGE: SMAP_regridding.sh <control_file_name>
# The control file must have the path to the merged ASCAT netCDF 
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
cdo remapnn,mygrid_1km ${targetFolder}ascat_h115_linear.nc ${targetFolder}ascat_1km_linear.nc
#cdo remapnn,mygrid_1km ${targetFolder}ascat_h115_nearest.nc ${targetFolder}ascat_1km_nearest.nc
chmod 664 ${targetFolder}ascat_1km_linear.nc
#chmod 664 ${targetFolder}ascat_1km_nearest.nc
chmod 664 mygrid_1km

