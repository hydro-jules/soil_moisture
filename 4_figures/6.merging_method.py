version = 'v1'

import scipy
import matplotlib as mpl
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



###############################################################################
###################plot TC weight patterns########################################
filename_tc =[mergedFolder+'merging_method_summary.nc' ] #spot_stdanom_australia spot_monmean_australia
label = ['merging_method']
name_cap = ['Merging method']
for i in [0]:

    tc_err = Data(filename_tc[i], label[i],read=True)
    tc_err.data[(tc_err.data>4)|(tc_err.data<0)]=np.nan
    tc_err.data  = np.ma.masked_where(np.isnan(tc_err.data), tc_err.data)

    fig = plt.figure(figsize=(5,4))  #figsize=(12,10)
    ax = fig.add_subplot(1,1,1)
    m = Basemap(projection='cyl',llcrnrlat=49.5,urcrnrlat=59,\
    llcrnrlon=-7.5,urcrnrlon=2.1,resolution='c',ax = ax)

    cmap = py.cm.get_cmap('jet_r', 7)

    # m.drawmapboundary()
    m.drawparallels(np.arange(-100.,100.,4.),labels=[1,0,0,0],fontsize=11,color='grey',linewidth=0.4)
    m.drawmeridians(np.arange(-180,180,6),labels=[0,1,0,1],fontsize=11,color='grey',linewidth=0.4)
    x,y = m(tc_err.lon,tc_err.lat)
    #im = m.pcolormesh(x,y,tc_err.data[:,:],cmap=plt.cm.RdYlBu,zorder=0)
    im = m.pcolormesh(x,y,tc_err.data[:,:],cmap=cmap,zorder=0)
    im.set_clim(-0.5,6.5)
    cbar = m.colorbar(im,format='%1.2g',size="5%", pad=0.2)#,location='bottom',pad="10%")
    cbar.set_ticks([0,1,2,3,4,5,6])
    cbar.set_ticklabels(["0", "1", "2", "3","4","5","6"])

    cbar.set_label('Merging Method',fontsize = 11)
   # cbar.ax.tick_params(labelsize=10)
    # m.drawcoastlines(linewidth=0.7)

    #plt.figtext(0.61,0.13, name_cap[i],fontsize = 11)
    fig.savefig(plotFolder + 'Map_'+name_cap[i]+'_'+version+'.png',dpi=400,bbox_inches='tight')
    plt.close('all')

