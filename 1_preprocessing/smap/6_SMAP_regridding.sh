#!/bin/sh

# Script to regrid SMAP to 9km grids
# USAGE: SMAP_regridding.sh <control_file_name>
# The control file must have the path to the merged SMAP netCDF 
# file on the second line

# Second line of control file is the path to the target folder
targetFolder=$(sed '2q;d' $1)

cat > mygrid_9km << EOF
gridtype = latlon
xsize    = 101
ysize    = 105
xfirst   = -7.46
xinc     = 0.09
yfirst   = 59.19
yinc     = -0.09
EOF

#targetFolder=/prj/hydrojules/data/soil_moisture/preprocessed/smap/smap_merged/

cdo sellonlatbox,-7.55,1.55,49.77,59.21
cdo remapnn,mygrid_9km ${targetFolder}smap_am_pm_all_uk.nc ${targetFolder}smap_9km.nc

chmod 664 ${targetFolder}smap_9km.nc
chmod 664 mygrid_9km


