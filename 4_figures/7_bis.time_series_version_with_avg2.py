version = 'v1'

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import netCDF4 as nc
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
from scipy import stats
import glob
from mpl_toolkits.basemap import Basemap
import pytesmo.time_series.anomaly as ts
import pytesmo.time_series.filtering as mv


# statistics and plotting functions
####################################################
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def rmse_c(predictions, targets):
    return np.sqrt((((predictions-np.nanmean(predictions)) - (targets-np.nanmean(targets))) ** 2).mean())

# CREATE BAR PLOT OF BIAS AND SO ON
def autolabel(rects,labels):
    counter = 0
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, str(int(labels[counter])) ,
                ha='center', va='bottom', rotation=90)
        counter = counter + 1

# CREATE BAR PLOT OF BIAS AND SO ON
def autolabel_num(rects,labels):
    counter = 0
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        axarr[1].text(rect.get_x()+rect.get_width()/2., 0.16, str(int(labels[counter])) ,
                ha='center', va='bottom', rotation=0,fontsize=15,fontweight='bold')
        counter = counter + 1
####################################################

# Extract point value based on given lat/lon
def extract_point(file_link,data_length,site_count,lat,lon,variable,scale):

    dataset = nc.Dataset(file_link,'r')
    s_lat = np.asarray(dataset.variables['lat'])
    s_lon = np.asarray(dataset.variables['lon'])
    lons, lats = np.meshgrid(s_lon, s_lat)

    s_sm = np.empty([len(data_length),site_count])
    s_sm[:] = float('NaN')

    count = 0
    for i in lat:
      cc=np.sqrt((lats-lat[count])**2+(lons-lon[count])**2)
      #calculate the minimum distance of the gridcell to station
      index= np.where(cc==np.min(cc))
      lat1=index[0][0]
      lon1=index[1][0]
      s_sm[:,count] = np.squeeze(np.asarray(dataset.variables[variable][:,lat1,lon1])).flatten()
      count =count +1
    s_sm = s_sm/scale
    s_sm[s_sm<=0.0]=float('NaN')
    # Create Pandas dataframe
    s_data = pd.DataFrame(s_sm[0:,0:])
    s_data['timestamp'] =  pd.to_datetime(data_length)
    s_point = s_data.set_index('timestamp')
    dataset.close()
    return s_point


# 1. Define time horizone that should be compared
date_start = pd.datetime(2015,04,01)
date_end   = pd.datetime(2017,10,12)
date_range_cf  = pd.date_range(date_start,date_end,freq='1D')




####load cosmos measurements
#Define directory of COSMOS measurements
#station_dir = '/cosmos/'

station_dir = '/prj/hydrojules/data/soil_moisture/COSMOS/data_for_review/'
SM_data_dir = '/prj/hydrojules/data/soil_moisture/preprocessed/data_for_review/'
plot_dir_1 = '/prj/hydrojules/users/malngu/plots/plots_for_review/number_7/merged_and_mean/'
plot_dir_2 = '/prj/hydrojules/users/malngu/plots/plots_for_review/number_7/smos_smap_ascat/'
plot_dir_3 = '/prj/hydrojules/users/malngu/plots/plots_for_review/number_7/jules_and_merged/'
merged_dir = '/prj/hydrojules/data/soil_moisture/merged/Paper_review_2/'

#########################################################################################
#######two files are used, because new cosmos_2017 has more sites since 2017, but has no lat/lon info for all sites
cosmos = pd.read_csv(station_dir+'COSMOS-UK_VolumetricWaterContent_Daily_2013-2017.csv',skiprows=1,header=None,delim_whitespace=False,
                           names=['site','date','value1','value2'],parse_dates=['date'],dayfirst=True,
                           index_col=[1])

cosmos_2016 = pd.read_csv(station_dir+'COSMOS-UK_VolumetricWaterContent_Daily_2013-2016.csv',skiprows=1,header=None,delim_whitespace=False,
                           names=['site','date','value1','value2'],parse_dates=['date'],dayfirst=True,
                           index_col=[1])

print 'Extracting COSMOS data...\n'

####creat empty data frame with time from 2015 to 2016
date_start = pd.datetime(2015,04,1,00,00)
date_end   = pd.datetime(2017,10,12,00,00)
date_range = pd.date_range(date_start,date_end,freq='1D')
station_data =  pd.DataFrame.from_items([('timestamp',date_range)])
station_data = station_data.set_index('timestamp')

for site in cosmos_2016['site'].unique():
    # print site
    # site = 'WYTH1'
    temp = cosmos.loc[cosmos['site'] == site]
    ###subset the years from 2015 to 2016
    temp_sub =temp['20150401':'20171012']
    ###add data to the dataframe
    station_data = station_data.join(temp_sub['value1'],how='outer')
    ####change column name value to site name
    station_data = station_data.rename(columns={'value1':site})
    del temp_sub
    del temp

station_data=station_data/100
station_data=station_data.mask((station_data<0)|(station_data>1))
station_names =  np.asarray(station_data.columns.values)

dataset = nc.Dataset(station_dir+'cosmos_daily_2013-2016.nc','r')
lat = np.asarray(dataset.variables['latitude'])
lon = np.asarray(dataset.variables['longitude'])
dataset.close()


###################################################################################################################################################
#2. Load dataset from gridded datasets
###################################################################################################################################################

file_smap9km = SM_data_dir + '/smap_9km_short.nc'
file_smos1km = SM_data_dir + '/smos1km_uk_all.nc'
file_chess1km = SM_data_dir + '/chess_1km_short.nc'
file_smap_s1 = SM_data_dir + '/smap_s1_all_v2.nc'
file_s1 = SM_data_dir + '/s1_uk_all_hwsd.nc'
file_smapl4 = SM_data_dir + '/smap_L4_9km_short.nc'
file_tc_merge = merged_dir + 'merge_12.5km_tc_ref_smap.nc'
#file_ascat = SM_data_dir + '/ascat_12.5km.nc'
file_ascat = SM_data_dir + '/ascat_VWC_linear_T_BULK_DEN_porosity_Toth_et_al.nc'
#file_mean = '/prj/hydrojules/data/soil_moisture/merged/Paper_review/merge_9km_chess_smap_mean_short.nc'
file_mean = '/prj/hydrojules/data/soil_moisture/merged/Paper_review_2/merge_9km_chess_smap_ascat_mean_short.nc'

print 'Extracting satellite data...\n'
smap9km = extract_point(file_smap9km,date_range_cf,41,lat,lon,'sm',1)
for k in range(41):
   smap9km = smap9km.rename(columns={k:'smap9km' + station_names[k] })

smos1km = extract_point(file_smos1km,date_range_cf,41,lat,lon,'sm',1)
for k in range(41):
   smos1km = smos1km.rename(columns={k:'smos1km' + station_names[k] })

chess1km = extract_point(file_chess1km,date_range_cf,41,lat,lon,'sm',100)
for k in range(41):
   chess1km = chess1km.rename(columns={k:'chess1km' + station_names[k] })

tc_merge = extract_point(file_tc_merge,date_range_cf,41,lat,lon,'sm',1)
for k in range(41):
   tc_merge = tc_merge.rename(columns={k:'tc_merge' + station_names[k] })

s1 = extract_point(file_mean,date_range_cf,41,lat,lon,'sm',1)
for k in range(41):
   s1 = s1.rename(columns={k:'s1' + station_names[k] })

smap_l4 = extract_point(file_smapl4,date_range_cf,41,lat,lon,'sm',1)
for k in range(41):
   smap_l4 = smap_l4.rename(columns={k:'smap_l4' + station_names[k] })

ascat12_5km = extract_point(file_ascat,date_range_cf,41,lat,lon,'sm',1)
for k in range(41):
   ascat12_5km = ascat12_5km.rename(columns={k:'ascat12_5km' + station_names[k] })


##################################time series for each site between Chess and tc_merged product##################################
################################################################################################################################################
#################################################################################################################################################################

new = station_data
new = new.join(smap9km)
new = new.join(smos1km)
new = new.join(chess1km)
new = new.join(tc_merge)
new = new.join(s1)
new = new.join(smap_l4)
new = new.join(ascat12_5km)

##station_names = ['ALIC1','BALRD','BICKL','BUNNY','CARDT','CHIMN','CHOBH','COCHN','COCLP','CRICH','CGARW',\
##                'EASTB','ELMST','EUSTN','FINCH','GISBN','GLENS','GLENW','HADLW','HARTW','HARWD','HENFS',\
##                'HYBRY','HILLB','HOLLN','LODTN','LULLN','MOORH','MORLY','NWYKE','PLYNL','PORTN','REDHL',\
##                'RDMER','RISEH','ROTHD','SHEEP','SOURH','SPENF','STIPS','STGHT','TADHM','LIZRD','WADDN',\
##                'WRTTL','WYTH1']


station_names = ['ALIC1','BALRD','BICKL','BUNNY','CARDT','CHIMN','CHOBH','COCLP','CRICH',\
                'EASTB','ELMST','EUSTN','GISBN','GLENS','GLENW','HADLW','HARTW','HARWD','HENFS',\
                'HILLB','HOLLN','LODTN','LULLN','MOORH','MORLY','NWYKE','PLYNL','PORTN','REDHL',\
                'RDMER','RISEH','ROTHD','SHEEP','SOURH','SPENF','STIPS','STGHT','TADHM','LIZRD',\
                'WADDN','WYTH1']

#station_names = ['HOLLN']

#=========================================================================
# First series of plots (merged and mean)
print '\n#==============================================================#'
print '                             PLOT 1'
print '#==============================================================#\n'

for site in station_names:
    print site

    mask = ~np.isnan(new[site])& ~np.isnan(new['tc_merge'+site])& ~np.isnan(new['s1'+site])
  # site = 'M5'
    fig = plt.figure(figsize=(7,3))
    ax1 = plt.subplot()
    ax1.plot(new[site][mask].index,new[site][mask],'b.',label='Station_'+site)

    ax1.plot(new['tc_merge'+site][mask].index,new['tc_merge'+site][mask],'rx',label='Merge')
    ax1.plot(new['s1'+site][mask].index,new['s1'+site][mask],'m4',label='Average')

    ax1.set_ylim(0, 0.8)
    ax1.set_ylabel('SM \n' + r'$\,\mathrm{(m^3/m^3)}$',fontsize=11)
    ax1.legend(loc='upper left', ncol=3,frameon=False,fontsize=11)
    ax1.grid(True, which='major')
    # plt.show()
    plt.tick_params(axis='x', labelsize=11)
    plt.tick_params(axis='y', labelsize=11)
    fig.autofmt_xdate()
    fig.text(0.145, 0.7, 'Site: '+site,fontsize=11)
    plt.savefig(plot_dir_1 + site +'_time_series_mean_merged_with_avg2_'+version+'.png',dpi=300,bbox_inches='tight')
    plt.close()



"""
#=========================================================================
# Second series of plots (SMAPs ASCAT)
print '\n#==============================================================#'
print '                             PLOT 2'
print '#==============================================================#\n'
for site in station_names:
    print site

    mask = ~np.isnan(new['ascat12_5km'+site])& ~np.isnan(new['smap9km'+site])& ~np.isnan(new['smap_l4'+site])
  # site = 'M5'
    fig = plt.figure(figsize=(7,3))
    ax1 = plt.subplot()

    ax1.plot(new['ascat12_5km'+site][mask].index,new['ascat12_5km'+site][mask],'y+',label='ASCAT')
    ax1.plot(new['smap9km'+site][mask].index,new['smap9km'+site][mask],'c|',label='SMAP L3E')
    ax1.plot(new['smap_l4'+site][mask].index,new['smap_l4'+site][mask],'m4',label='SMAP L4')

    ax1.set_ylim(0, 0.8)
    ax1.set_ylabel('SM \n' + r'$\,\mathrm{(m^3/m^3)}$',fontsize=11)
    ax1.legend(loc='upper left', ncol=3,frameon=False,fontsize=11)
    ax1.grid(True, which='major')
    # plt.show()
    plt.tick_params(axis='x', labelsize=11)
    plt.tick_params(axis='y', labelsize=11)
    fig.autofmt_xdate()
    fig.text(0.145, 0.7, 'Site: '+site,fontsize=11)
    plt.savefig(plot_dir_2 + site +'_time_series_ascat_smap_'+version+'.png',dpi=300,bbox_inches='tight')
    plt.close()




#=========================================================================
# Third series of plots (merged and mean)
print '\n#==============================================================#'
print '                             PLOT 3'
print '#==============================================================#\n'
for site in station_names:
    print site

    mask = ~np.isnan(new[site])& ~np.isnan(new['tc_merge'+site])& ~np.isnan(new['chess1km'+site])
  # site = 'M5'
    fig = plt.figure(figsize=(7,3))
    ax1 = plt.subplot()
    ax1.plot(new[site][mask].index,new[site][mask],'b.',label='Station_'+site)

    ax1.plot(new['chess1km'+site][mask].index,new['chess1km'+site][mask],'g1',label='JULES-CHESS')
    ax1.plot(new['tc_merge'+site][mask].index,new['tc_merge'+site][mask],'rx',label='Merge')


    ax1.set_ylim(0, 0.8)
    ax1.set_ylabel('SM \n' + r'$\,\mathrm{(m^3/m^3)}$',fontsize=11)
    ax1.legend(loc='upper left', ncol=3,frameon=False,fontsize=11)
    ax1.grid(True, which='major')
    # plt.show()
    plt.tick_params(axis='x', labelsize=11)
    plt.tick_params(axis='y', labelsize=11)
    fig.autofmt_xdate()
    fig.text(0.145, 0.7, 'Site: '+site,fontsize=11)
    plt.savefig(plot_dir_3 + site +'_time_series_jules_merged_'+version+'.png',dpi=300,bbox_inches='tight')
    plt.close()



#################################################################################################################################################################
"""