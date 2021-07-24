# NOTE: 5a needs to be run first.

version = 'v1'

import scipy
import datetime
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import netCDF4 as nc
from matplotlib import rc
rc('mathtext',default = 'regular')
import pylab as py
#from pycmbs.region import RegionBboxLatLon
#from pycmbs.plots import GlobalMeanPlot
from pycmbs.data import Data
#from pycmbs.mapping import map_plot
#from pycmbs.diagnostic import PatternCorrelation
#from pycmbs.plots import ScatterPlot
from scipy.stats import linregress

inFolder = '/prj/hydrojules/data/soil_moisture/preprocessed/data_for_review/'
mergedFolder = '/prj/hydrojules/data/soil_moisture/merged/Paper_review_2/'
plotFolder = '/prj/hydrojules/users/malngu/plots/plots_for_review/number_5/'

############################################################################
#plot 9km soil moisture product
############################################################################

# To calculate monthly mean ncdf: cdo monmean in.nc out.nc
# cdo monmean smap_12.5km.nc smap_12.5km_monmean.nc
# cdo monmean chess_12.5km.nc chess_12.5km_monmean.nc
# cdo monmean ascat_VWC_nearest_T_BULK_DEN_porosity_Toth_et_al.nc ascat_12.5km_monmean.nc
# cdo monmean merge_12.5km_tc_ref_smap.nc merge_12.5km_monmean.nc

# To select March 2016 (timestep number 12): ncea -F -d time,first,last in.out out.nc
# ncea -F -d time,12,12 ascat_12.5km_monmean.nc ascat_12.5km_march2016.nc
# ncea -F -d time,12,12 smap_12.5km_monmean.nc smap_12.5km_march2016.nc
# ncea -F -d time,12,12 chess_12.5km_monmean.nc chess_12.5km_march2016.nc
# ncea -F -d time,12,12 merge_12.5km_monmean.nc merge_12.5km_march2016.nc


# Didn't work for SMAP, so this is what I did:
# First extract daily timesteps for March 2016
# ncea -F -d time,336,366 smap_12.5km.nc smap_12.5km_march2016_daily.nc
# Then calculate monthly mean (have to set missing values to nan for it to work properly).
# cdo -b F64 -timmean -setmissval,nan smap_12.5km_march2016_daily.nc smap_12.5km_march2016_with_cdo.nc
# Then mask out northern Ireland!!!
# Using python code 5a.mask_out_NI_SMAP.py

filename_chess =inFolder + 'chess_12.5km.nc'
chess = Data(filename_chess, 'sm',start_time=datetime.datetime(2016, 3, 1),stop_time=datetime.datetime(2016, 3, 31), read=True)
chess.data = chess.data/100
##chess.data[(chess.data>1)|(chess.data<0)]=np.nan
##chess.data  = np.ma.masked_where(np.isnan(chess.data), chess.data)

filename_smap9km =inFolder + 'smap_12.5km.nc'
smap_9 = Data(filename_smap9km,'sm',start_time=datetime.datetime(2016, 3, 1),stop_time=datetime.datetime(2016, 3, 31), read=True)
##smap_9.data[(smap_9.data>1)|(smap_9.data<0)]=np.nan
##smap_9.data  = np.ma.masked_where(np.isnan(smap_9.data)|np.isnan(chess.data), smap_9.data)

filename_smos_9 =inFolder + 'ascat_VWC_linear_T_BULK_DEN_porosity_Toth_et_al.nc' #smos1km_uk_all_cut_9km
smos_9 = Data(filename_smos_9, 'sm',start_time=datetime.datetime(2016, 3, 1),stop_time=datetime.datetime(2016, 3, 31), read=True)
##smos_9.data[(smos_9.data>1)|(smos_9.data<0)]=np.nan
##smos_9.data  = np.ma.masked_where(np.isnan(smos_9.data)|np.isnan(chess.data), smos_9.data)

filename_merge =mergedFolder + 'merge_12.5km_tc_ref_smap.nc'
merge = Data(filename_merge, 'sm',start_time=datetime.datetime(2016, 3,1),stop_time=datetime.datetime(2016, 3, 31), read=True)
##merge.data[(merge.data>1)|(merge.data<0)]=np.nan
##merge.data  = np.ma.masked_where(np.isnan(merge.data)|np.isnan(chess.data), merge.data)



###############################################################################
###################plot TC weight patterns########################################
filename_tc =[mergedFolder+'weight1.nc' ,mergedFolder+'weight2.nc',mergedFolder+'weight3.nc'] #spot_stdanom_australia spot_monmean_australia
label = ['w1','w2','w3']
name_cap = ['SMAP_L3E','JULES-CHESS','ASCAT']
for i in [0,1,2]:

    tc_err = Data(filename_tc[i], label[i],read=True)
    tc_err.data[(tc_err.data>1.1)|(tc_err.data<=0)]=np.nan
    tc_err.data  = np.ma.masked_where(np.isnan(tc_err.data), tc_err.data)

    fig = plt.figure(figsize=(5,4))  #figsize=(12,10)
    ax = fig.add_subplot(1,1,1)
    m = Basemap(projection='cyl',llcrnrlat=49.5,urcrnrlat=59,\
    llcrnrlon=-7.5,urcrnrlon=2.1,resolution='c',ax = ax)

    # m.drawmapboundary()
    m.drawparallels(np.arange(-100.,100.,4.),labels=[1,0,0,0],fontsize=11,color='grey',linewidth=0.4)
    m.drawmeridians(np.arange(-180,180,6),labels=[0,1,0,1],fontsize=11,color='grey',linewidth=0.4)
    x,y = m(tc_err.lon,tc_err.lat)
    im = m.pcolormesh(x,y,tc_err.data[:,:],cmap=plt.cm.RdYlBu,zorder=0)
    im.set_clim(0,1)
    cbar = m.colorbar(im,format='%1.2g',size="2%", pad=0.05)#,location='bottom',pad="10%")
    cbar.set_label('Weight',fontsize = 11)
    cbar.ax.tick_params(labelsize=10)
    # m.drawcoastlines(linewidth=0.7)

    plt.figtext(0.61,0.13, name_cap[i],fontsize = 11)
    fig.savefig(plotFolder + 'TC_weight_all'+name_cap[i]+'_'+version+'.png',dpi=400,bbox_inches='tight')
    plt.close('all')


###############################################################################

###############################################################################
################### plot SM ########################################

filename_tc =[inFolder+'masked_smap_12.5km_march2016_'+version+'.nc' ,inFolder+'chess_12.5km_march2016.nc',\
              inFolder+'ascat_12.5km_march2016.nc', mergedFolder + 'merge_12.5km_march2016.nc'] #spot_stdanom_australia spot_monmean_australia
label = ['sm','sm','sm']
name_cap = ['SMAP_L3E','JULES-CHESS','ASCAT','Merged']
for i in [0,1,2,3]:
    print i
    dataset = nc.Dataset(filename_tc[i],'r')
    sm = np.squeeze(np.array(dataset.variables['sm']))
    lat = np.squeeze(np.array(dataset.variables['lat']))
    lon = np.squeeze(np.array(dataset.variables['lon']))

    if i==1:
        sm=sm/100.0

    sm[(sm>1.1)|(sm<=0)]=np.nan
    sm  = np.ma.masked_where(np.isnan(sm), sm)

    fig = plt.figure(figsize=(5,4))  #figsize=(12,10)
    ax = fig.add_subplot(1,1,1)
    m = Basemap(projection='cyl',llcrnrlat=49.5,urcrnrlat=59,\
    llcrnrlon=-7.5,urcrnrlon=2.1,resolution='c',ax = ax)

    # m.drawmapboundary()
    m.drawparallels(np.arange(-100.,100.,4.),labels=[1,0,0,0],fontsize=11,color='grey',linewidth=0.4)
    m.drawmeridians(np.arange(-180,180,6),labels=[0,1,0,1],fontsize=11,color='grey',linewidth=0.4)
    x,y = m(lon,lat)
    im = m.pcolormesh(x,y,sm,cmap=plt.cm.RdYlBu,zorder=0)
    im.set_clim(0,0.6)
    cbar = m.colorbar(im,format='%1.2g',size="2%", pad=0.05)#,location='bottom',pad="10%")
    cbar.set_label('Soil moisture (m3/m3)',fontsize = 11)
    cbar.ax.tick_params(labelsize=10)
    # m.drawcoastlines(linewidth=0.7)

    plt.figtext(0.61,0.13, name_cap[i],fontsize = 11)
    fig.savefig(plotFolder + 'Soil_Moisture_all'+name_cap[i]+'_'+version+'.png',dpi=400,bbox_inches='tight')
    plt.close('all')


###############################################################################
