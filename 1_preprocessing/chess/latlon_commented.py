#-------------------------------------------------------------------------------
# Create Lat Lon numpy arrays
#
# Author:      Maliko Tanguy
#              malngu@ceh.ac.uk
#              18/12/2015 (updated: 22/06/2020)
#-------------------------------------------------------------------------------


import numpy as np
import datetime
import mpl_toolkits.basemap.pyproj as pyproj

# define x and y dimensions
#xdim = 180
#ydim = 290
xdim = 656
ydim=1057

# define x and y range and step (1000 = 1km grid)
# Note: the coordinates are at the CENTRE of the grid cell
#coordX = np.arange(-197500,697500+5000,5000)
#coordY = np.arange(-197500,1247500+5000,5000)

# Extract the grid coordinates
coordX = np.arange(xdim)*1000.0 + 500.0
coordY = np.arange(ydim)*1000.0 + 500.0

# define coordinate systems
wgs84=pyproj.Proj("+init=EPSG:4326") # LatLon with WGS84 datum
osgb36=pyproj.Proj("+init=EPSG:27700") # UK Ordnance Survey, 1936 datum

realLat = np.zeros((ydim,xdim))
realLon = np.zeros((ydim,xdim))
for y in range(ydim):
    if y%100 == 0:
        print y
    for x in range(xdim):
        Lon, Lat = pyproj.transform(osgb36, wgs84, coordX[x], coordY[y])
        realLat[y,x] = Lat
        realLon[y,x] = Lon

np.save('Lat_1km.npy',realLat)
np.save('Lon_1km.npy',realLon)
os.chmod('Lat_1km.npy',0644)
os.chmod('Lon_1km.npy',0644)
# you can then load these files as numpy arrays using:
# latArray = np.load('Lat_1km.npy')
