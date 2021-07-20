#!/bin/sh

# Script to regrid HWSD to 1km grids
# USAGE: HWSD_regridding_1km.sh <control_file_name>
# The control file must have the path to the merged HWSD netCDF 
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
cdo remapnn,mygrid_1km ${targetFolder}T_BULK_DEN.nc4 ${targetFolder}T_BULK_DEN_1km.nc
cdo remapnn,mygrid_1km ${targetFolder}T_REF_BULK.nc4 ${targetFolder}T_REF_BULK_1km.nc
cdo remapnn,mygrid_1km ${targetFolder}T_OC.nc4 ${targetFolder}T_OC_1km.nc
cdo remapnn,mygrid_1km ${targetFolder}T_PH_H2O.nc4 ${targetFolder}T_PH_H2O_1km.nc
cdo remapnn,mygrid_1km ${targetFolder}T_CLAY.nc4 ${targetFolder}T_CLAY_1km.nc
chmod 664 ${targetFolder}T_BULK_DEN_1km.nc
chmod 664 ${targetFolder}T_REF_BULK_1km.nc
chmod 664 ${targetFolder}T_OC_1km.nc
chmod 664 ${targetFolder}T_PH_H2O_1km.nc
chmod 664 ${targetFolder}T_CLAY_1km.nc
chmod 664 mygrid_1km

