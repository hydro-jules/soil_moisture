#!/bin/sh

# Script to regrid HWSD to 12.5km grids
# USAGE: HWSD_regridding.sh <control_file_name>
# The control file must have the path to the merged HWSD netCDF 
# file on the second line

# Second line of control file is the path to the target folder
targetFolder=$(sed '2q;d' $1)

cat > mygrid_12.5km << EOF
gridtype = latlon
xsize    = 76
ysize    = 76
xfirst   = -7.46
xinc     = 0.125
yfirst   = 59.19
yinc     = -0.125
EOF

cdo sellonlatbox,-7.55,1.55,49.77,59.21
cdo remapnn,mygrid_12.5km ${targetFolder}T_BULK_DEN.nc4 ${targetFolder}T_BULK_DEN_12.5km.nc
cdo remapnn,mygrid_12.5km ${targetFolder}T_REF_BULK.nc4 ${targetFolder}T_REF_BULK_12.5km.nc
cdo remapnn,mygrid_12.5km ${targetFolder}T_OC.nc4 ${targetFolder}T_OC_12.5km.nc
cdo remapnn,mygrid_12.5km ${targetFolder}T_PH_H2O.nc4 ${targetFolder}T_PH_H2O_12.5km.nc
cdo remapnn,mygrid_12.5km ${targetFolder}T_CLAY.nc4 ${targetFolder}T_CLAY_12.5km.nc
chmod 664 ${targetFolder}T_BULK_DEN_12.5km.nc
chmod 664 ${targetFolder}T_REF_BULK_12.5km.nc
chmod 664 ${targetFolder}T_OC_12.5km.nc
chmod 664 ${targetFolder}T_PH_H2O_12.5km.nc
chmod 664 ${targetFolder}T_CLAY_12.5km.nc
chmod 664 mygrid_12.5km

