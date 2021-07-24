# Script for plotting all products (direct sample,not collocated), inlucding box plot, bar plot
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


# 'cubic' not suitable because it gives values < 0.
method = 'linear'

# there is the choice between two bulk density dataset 'T_BULK_DEN' or 'T_REF_BULK'
bulk_den = 'T_BULK_DEN'
#bulk_den = 'T_REF_BULK'


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
############################################################
# Function to show values on heatmap
def show_values(pc,ax, fmt="%.2f", **kw):
    from itertools import izip
    pc.update_scalarmappable()
    ax = pc.get_axes()
    for p, color, value in izip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.3):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", fontsize=4,color=color, **kw)

# 1. Define time horizone that should be compared
date_start = pd.datetime(2015,04,01)
date_end   = pd.datetime(2017,10,12)
date_range_cf  = pd.date_range(date_start,date_end,freq='1D')




####load cosmos measurements
#Define directory of COSMOS measurements
##station_dir = '/cosmos/'
station_dir = '/prj/hydrojules/data/soil_moisture/COSMOS/data_for_review/'
SM_data_dir = '/prj/hydrojules/data/soil_moisture/preprocessed/data_for_review/'
plot_dir = '/prj/hydrojules/users/malngu/plots/plots_for_review/number_2/'+method + '_' + bulk_den + '/'

#########################################################################################
#######two files are used, because new cosmos_2017 has more sites since 2017, but has no lat/lon info for all sites
cosmos = pd.read_csv(station_dir+'COSMOS-UK_VolumetricWaterContent_Daily_2013-2017.csv',skiprows=1,header=None,delim_whitespace=False,
                           names=['site','date','value1','value2'],parse_dates=['date'],dayfirst=True,
                           index_col=[1])

cosmos_2016 = pd.read_csv(station_dir+'COSMOS-UK_VolumetricWaterContent_Daily_2013-2016.csv',skiprows=1,header=None,delim_whitespace=False,
                           names=['site','date','value1','value2'],parse_dates=['date'],dayfirst=True,
                           index_col=[1])

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

#site WYTH1 has no collocated data with all products
# station_data=station_data.drop(['WYTH1'],1)
station_data=station_data/100
station_data=station_data.mask((station_data<0)|(station_data>1))
station_names_temp =  np.asarray(station_data.columns.values)

dataset = nc.Dataset(station_dir+'cosmos_daily_2013-2016.nc','r')
lat_temp = np.asarray(dataset.variables['latitude'])
lon_temp = np.asarray(dataset.variables['longitude'])
dataset.close()

################ delete statiobns that in northern ireland, longitute less than -6###################
lat = list()
lon = list()
station_names = list()

for i in range(len(lat_temp)):
    if station_names_temp[i] not in ['WYTH1','HILLB','GLENW']:
        lat.append(lat_temp[i])
        lon.append(lon_temp[i])
        station_names.append(station_names_temp[i])

lon = np.asarray(lon)
lat = np.asarray(lat)
station_names = np.asarray(station_names)



###################################################################################################################################################
#2. Load dataset from gridded datasets
###################################################################################################################################################

file_smap9km = SM_data_dir + '/smap_9km_short.nc'
file_smos1km = SM_data_dir + '/smos1km_uk_all.nc'
file_chess1km = SM_data_dir + '/chess_1km_short.nc'
file_smap_s1 = SM_data_dir + '/smap_s1_all_v2.nc'
file_s1 = SM_data_dir + '/s1_uk_all_hwsd.nc'
file_smapl4 = SM_data_dir + '/smap_L4_9km_short.nc'
#file_ascat = SM_data_dir + '/ascat_12.5km.nc'
file_ascat = SM_data_dir + '/ascat_VWC_'+method+'_'+bulk_den+'_porosity_Toth_et_al.nc'

smap9km = extract_point(file_smap9km,date_range_cf,38,lat,lon,'sm',1)
for k in range(38):
   smap9km = smap9km.rename(columns={k:'smap9km' + station_names[k] })

smos1km = extract_point(file_smos1km,date_range_cf,38,lat,lon,'sm',1)
for k in range(38):
   smos1km = smos1km.rename(columns={k:'smos1km' + station_names[k] })

chess1km = extract_point(file_chess1km,date_range_cf,38,lat,lon,'sm',100)
for k in range(38):
   chess1km = chess1km.rename(columns={k:'chess1km' + station_names[k] })

smap_s1 = extract_point(file_smap_s1,date_range_cf,38,lat,lon,'sm',1)
for k in range(38):
   smap_s1 = smap_s1.rename(columns={k:'smap_s1' + station_names[k] })

s1 = extract_point(file_s1,date_range_cf,38,lat,lon,'sm',1)
for k in range(38):
   s1 = s1.rename(columns={k:'s1' + station_names[k] })

smap_l4 = extract_point(file_smapl4,date_range_cf,38,lat,lon,'sm',1)
for k in range(38):
   smap_l4 = smap_l4.rename(columns={k:'smap_l4' + station_names[k] })

ascat12_5km = extract_point(file_ascat,date_range_cf,38,lat,lon,'sm',1)
for k in range(38):
   ascat12_5km = ascat12_5km.rename(columns={k:'ascat12_5km' + station_names[k] })

#################################calcuate error statistics for each station###########
######################################################################################

new = station_data
new = new.join(smap9km)
new = new.join(smos1km)
new = new.join(chess1km)
new = new.join(smap_s1)
new = new.join(s1)
new = new.join(smap_l4)
new = new.join(ascat12_5km)

############### Remove seasonality using moving window, set threshold to 6 days#####################
for s in new.columns:

    anomaly = new[s] - mv.moving_average(new[s], window_size=35,min_obs=6)
    new[s] = anomaly

################################################################################################################
#Calculate Bias,rmse,crmse and R-value between input data and in-situ data for later use in following plots #
################################################################################################################
# create ouput arrays

bias_smap = np.empty([0])
bias_smos = np.empty([0])
bias_chess = np.empty([0])
bias_s1 = np.empty([0])
bias_smap_s1 = np.empty([0])
bias_smap_l4 = np.empty([0])
bias_ascat = np.empty([0])

crmse_smap = np.empty([0])
crmse_smos = np.empty([0])
crmse_chess = np.empty([0])
crmse_s1 = np.empty([0])
crmse_smap_s1 = np.empty([0])
crmse_smap_l4 = np.empty([0])
crmse_ascat = np.empty([0])

rmse_smap = np.empty([0])
rmse_smos = np.empty([0])
rmse_chess = np.empty([0])
rmse_s1 = np.empty([0])
rmse_smap_s1 = np.empty([0])
rmse_smap_l4 = np.empty([0])
rmse_ascat = np.empty([0])

rvalue_smap = np.empty([0])
rvalue_smos = np.empty([0])
rvalue_chess = np.empty([0])
rvalue_s1 = np.empty([0])
rvalue_smap_s1 = np.empty([0])
rvalue_smap_l4 = np.empty([0])
rvalue_ascat = np.empty([0])

pvalue_smap = np.empty([0])
pvalue_smos = np.empty([0])
pvalue_chess = np.empty([0])
pvalue_s1 = np.empty([0])
pvalue_smap_s1 = np.empty([0])
pvalue_smap_l4 = np.empty([0])
pvalue_ascat = np.empty([0])

notnan_sm = np.empty([0])


for site in station_names:

     ### ensure each product has the same sample size with COSMOS station

     mask_smap = ~np.isnan(new[site])& ~np.isnan(new['smap9km'+site])
     mask_s1 = ~np.isnan(new[site])& ~np.isnan(new['s1'+site])
     mask_chess = ~np.isnan(new[site])& ~np.isnan(new['chess1km'+site])
     mask_smos = ~np.isnan(new[site])& ~np.isnan(new['smos1km'+site])
     mask_smap_s1 = ~np.isnan(new[site])& ~np.isnan(new['smap_s1'+site])
     mask_smap_l4 = ~np.isnan(new[site])& ~np.isnan(new['smap_l4'+site])
     mask_ascat = ~np.isnan(new[site])& ~np.isnan(new['ascat12_5km'+site])


     ## calculate bias of each product
     temp_smap = new['smap9km'+site][mask_smap].mean()- new[site][mask_smap].mean()
     temp_smos = new['smos1km'+site][mask_smos].mean()- new[site][mask_smos].mean()
     temp_chess = new['chess1km'+site][mask_chess].mean()- new[site][mask_chess].mean()
     temp_smap_s1 = new['smap_s1'+site][mask_smap_s1].mean()- new[site][mask_smap_s1].mean()
     temp_s1 = new['s1'+site][mask_s1].mean()- new[site][mask_s1].mean()
     temp_smap_l4 = new['smap_l4'+site][mask_smap_l4].mean()- new[site][mask_smap_l4].mean()
     temp_ascat = new['ascat12_5km'+site][mask_ascat].mean()- new[site][mask_ascat].mean()
     # count the non-nan sample number

     bias_smap = np.append(bias_smap,temp_smap)
     if np.sum(mask_smap) == 0:
        slope_smap, intercept_smap, r_value_smap, p_value_smap, std_err_smap = [float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN')]
     else:
        slope_smap, intercept_smap, r_value_smap, p_value_smap, std_err_smap = stats.linregress(new[site][mask_smap],new['smap9km'+site][mask_smap])
     crmse_smap = np.append(crmse_smap,rmse_c(new['smap9km'+site][mask_smap],new[site][mask_smap]))
     rmse_smap = np.append(rmse_smap,rmse(new['smap9km'+site][mask_smap],new[site][mask_smap]))
     rvalue_smap = np.append(rvalue_smap,r_value_smap)
     pvalue_smap = np.append(pvalue_smap,p_value_smap)
     #####################


     bias_smos = np.append(bias_smos,temp_smos)
     if np.sum(mask_smos) == 0:
        slope_smos, intercept_smos, r_value_smos, p_value_smos, std_err_smos = [float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN')]
     else:
        slope_smos, intercept_smos, r_value_smos, p_value_smos, std_err_smos = stats.linregress(new[site][mask_smos],new['smos1km'+site][mask_smos])
     crmse_smos = np.append(crmse_smos,rmse_c(new['smos1km'+site][mask_smos],new[site][mask_smos]))
     rmse_smos = np.append(rmse_smos,rmse(new['smos1km'+site][mask_smos],new[site][mask_smos]))
     rvalue_smos = np.append(rvalue_smos,r_value_smos)
     pvalue_smos = np.append(pvalue_smos,p_value_smos)


     bias_chess = np.append(bias_chess,temp_chess)
     if np.sum(mask_chess) == 0:
        slope_chess, intercept_chess, r_value_chess, p_value_chess, std_err_chess = [float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN')]
     else:
        slope_chess, intercept_chess, r_value_chess, p_value_chess, std_err_chess = stats.linregress(new[site][mask_chess],new['chess1km'+site][mask_chess])
     crmse_chess = np.append(crmse_chess,rmse_c(new['chess1km'+site][mask_chess],new[site][mask_chess]))
     rmse_chess = np.append(rmse_chess,rmse(new['chess1km'+site][mask_chess],new[site][mask_chess]))
     rvalue_chess = np.append(rvalue_chess,r_value_chess)
     pvalue_chess = np.append(pvalue_chess,p_value_chess)


     bias_smap_s1 = np.append(bias_smap_s1,temp_smap_s1)
     if np.sum(mask_smap_s1) == 0:
        slope_smap_s1, intercept_smap_s1, r_value_smap_s1, p_value_smap_s1, std_err_smap_s1 = [float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN')]
     else:
        slope_smap_s1, intercept_smap_s1, r_value_smap_s1, p_value_smap_s1, std_err_smap_s1 = stats.linregress(new[site][mask_smap_s1],new['smap_s1'+site][mask_smap_s1])
     crmse_smap_s1 = np.append(crmse_smap_s1,rmse_c(new['smap_s1'+site][mask_smap_s1],new[site][mask_smap_s1]))
     rmse_smap_s1 = np.append(rmse_smap_s1,rmse(new['smap_s1'+site][mask_smap_s1],new[site][mask_smap_s1]))
     rvalue_smap_s1 = np.append(rvalue_smap_s1,r_value_smap_s1)
     pvalue_smap_s1 = np.append(pvalue_smap_s1,p_value_smap_s1)


     bias_s1 = np.append(bias_s1,temp_s1)
     if np.sum(mask_s1) == 0:
        slope_s1, intercept_s1, r_value_s1, p_value_s1, std_err_s1 = [float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN')]
     else:
        slope_s1, intercept_s1, r_value_s1, p_value_s1, std_err_s1 = stats.linregress(new[site][mask_s1],new['s1'+site][mask_s1])
     crmse_s1 = np.append(crmse_s1,rmse_c(new['s1'+site][mask_s1],new[site][mask_s1]))
     rmse_s1 = np.append(rmse_s1,rmse(new['s1'+site][mask_s1],new[site][mask_s1]))
     rvalue_s1 = np.append(rvalue_s1,r_value_s1)
     pvalue_s1 = np.append(pvalue_s1,p_value_s1)

     bias_smap_l4 = np.append(bias_smap_l4,temp_smap_l4)
     if np.sum(mask_smap_l4) == 0:
        slope_smap_l4, intercept_smap_l4, r_value_smap_l4, p_value_smap_l4, std_err_smap_l4 = [float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN')]
     else:
        slope_smap_l4, intercept_smap_l4, r_value_smap_l4, p_value_smap_l4, std_err_smap_l4 = stats.linregress(new[site][mask_smap_l4],new['smap_l4'+site][mask_smap_l4])
     crmse_smap_l4 = np.append(crmse_smap_l4,rmse_c(new['smap_l4'+site][mask_smap_l4],new[site][mask_smap_l4]))
     rmse_smap_l4 = np.append(rmse_smap_l4,rmse(new['smap_l4'+site][mask_smap_l4],new[site][mask_smap_l4]))
     rvalue_smap_l4 = np.append(rvalue_smap_l4,r_value_smap_l4)
     pvalue_smap_l4 = np.append(pvalue_smap_l4,p_value_smap_l4)

     bias_ascat = np.append(bias_ascat,temp_ascat)
     if np.sum(mask_ascat) == 0:
        slope_ascat, intercept_ascat, r_value_ascat, p_value_ascat, std_err_ascat = [float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN')]
     else:
        slope_ascat, intercept_ascat, r_value_ascat, p_value_ascat, std_err_ascat = stats.linregress(new[site][mask_ascat],new['ascat12_5km'+site][mask_ascat])
     crmse_ascat = np.append(crmse_ascat,rmse_c(new['ascat12_5km'+site][mask_ascat],new[site][mask_ascat]))
     rmse_ascat = np.append(rmse_ascat,rmse(new['ascat12_5km'+site][mask_ascat],new[site][mask_ascat]))
     rvalue_ascat = np.append(rvalue_ascat,r_value_ascat)
     pvalue_ascat = np.append(pvalue_ascat,p_value_ascat)




rvalue_chess = np.ma.masked_where(np.isnan(pvalue_chess)|(pvalue_chess>=0.1), rvalue_chess)
rvalue_smap = np.ma.masked_where(np.isnan(pvalue_smap)|(pvalue_smap>=0.1), rvalue_smap)
rvalue_smos = np.ma.masked_where(np.isnan(pvalue_smos)|(pvalue_smos>=0.1), rvalue_smos)
rvalue_smap_s1= np.ma.masked_where(np.isnan(pvalue_smap_s1)|(pvalue_smap_s1>=0.1), rvalue_smap_s1)
rvalue_s1= np.ma.masked_where(np.isnan(pvalue_s1)|(pvalue_s1>=0.1), rvalue_s1)
rvalue_smap_l4= np.ma.masked_where(np.isnan(pvalue_smap_l4)|(pvalue_smap_l4>=0.1), rvalue_smap_l4)
rvalue_ascat= np.ma.masked_where(np.isnan(pvalue_ascat)|(pvalue_ascat>=0.1), rvalue_ascat)

########################################################################################################

rvalue_chess=np.ma.filled(rvalue_chess.astype(float), np.nan)
rvalue_smap=np.ma.filled(rvalue_smap.astype(float), np.nan)
rvalue_smos=np.ma.filled(rvalue_smos.astype(float), np.nan)
rvalue_smap_s1=np.ma.filled(rvalue_smap_s1.astype(float), np.nan)
rvalue_s1=np.ma.filled(rvalue_s1.astype(float), np.nan)
rvalue_smap_l4=np.ma.filled(rvalue_smap_l4.astype(float), np.nan)
rvalue_ascat=np.ma.filled(rvalue_ascat.astype(float), np.nan)

crmse_smos = np.ma.masked_where(np.isnan(rvalue_smos), crmse_smos)
crmse_smap = np.ma.masked_where(np.isnan(rvalue_smap), crmse_smap)
crmse_smap_s1 = np.ma.masked_where(np.isnan(rvalue_smap_s1), crmse_smap_s1)
crmse_s1 = np.ma.masked_where(np.isnan(rvalue_s1), crmse_s1)
crmse_chess = np.ma.masked_where(np.isnan(rvalue_chess), crmse_chess)
crmse_smap_l4 = np.ma.masked_where(np.isnan(rvalue_smap_l4), crmse_smap_l4)
crmse_ascat = np.ma.masked_where(np.isnan(rvalue_ascat), crmse_ascat)

rmse_smos = np.ma.masked_where(np.isnan(rvalue_smos), rmse_smos)
rmse_smap = np.ma.masked_where(np.isnan(rvalue_smap), rmse_smap)
rmse_smap_s1 = np.ma.masked_where(np.isnan(rvalue_smap_s1), rmse_smap_s1)
rmse_s1 = np.ma.masked_where(np.isnan(rvalue_s1), rmse_s1)
rmse_chess = np.ma.masked_where(np.isnan(rvalue_chess), rmse_chess)
rmse_smap_l4 = np.ma.masked_where(np.isnan(rvalue_smap_l4), rmse_smap_l4)
rmse_ascat = np.ma.masked_where(np.isnan(rvalue_ascat), rmse_ascat)

bias_smos = np.ma.masked_where(np.isnan(rvalue_smos), bias_smos)
bias_smap = np.ma.masked_where(np.isnan(rvalue_smap), bias_smap)
bias_smap_s1 = np.ma.masked_where(np.isnan(rvalue_smap_s1), bias_smap_s1)
bias_s1 = np.ma.masked_where(np.isnan(rvalue_s1), bias_s1)
bias_chess = np.ma.masked_where(np.isnan(rvalue_chess), bias_chess)
bias_smap_l4 = np.ma.masked_where(np.isnan(rvalue_smap_l4), bias_smap_l4)
bias_ascat = np.ma.masked_where(np.isnan(rvalue_ascat), bias_ascat)

# #######box plots for ubRMSE and R for comparsion between merged SM and original SM###################################################
mask_nan_smap = ~np.isnan(rvalue_smap)
mask_nan_smos=~np.isnan(rvalue_smos)
mask_nan_smap_s1=~np.isnan(rvalue_smap_s1)
mask_nan_s1=~np.isnan(rvalue_s1)
mask_nan_chess=~np.isnan(rvalue_chess)
mask_nan_smap_l4=~np.isnan(rvalue_smap_l4)
mask_nan_ascat=~np.isnan(rvalue_ascat)




######################################################box plots for comparison against comsmos###########
######plot box plot for correlation#################
R_box = [rvalue_smos[mask_nan_smos], rvalue_smap[mask_nan_smap],rvalue_smap_s1[mask_nan_smap_s1],rvalue_s1[mask_nan_s1],rvalue_smap_l4[mask_nan_smap_l4],rvalue_chess[mask_nan_chess],rvalue_ascat[mask_nan_ascat]]
fig = plt.figure(figsize=(4,3))
ax1 = fig.add_subplot(111)
#ax1.boxplot(R_box,showfliers=True)
ax1.boxplot(R_box)
ax1.set_xticks([1,2,3,4,5,6,7])
# ax1.set_xticklabels(['SMOS','SMAP','JULES-CHESS','Merge_smap_s1','Merge_s1'],fontsize = 11)
ax1.set_xticklabels(['SMOS L4','SMAP L3E','SMAP/Sentinel 1','Sentinel 1','SMAP L4','JULES-CHESS','ASCAT'],fontsize = 11)
ax1.set_ylim([0,1.1])
ax1.yaxis.set_tick_params(labelsize=11)
ax1.set_ylabel(r'$R$',fontsize = 11)
ax1.grid(True)
fig.autofmt_xdate()
plt.savefig(plot_dir + 'box_corrleation_no_collocation__no_seasonality_with_ASCAT_'+version+'.png',dpi=400,bbox_inches='tight')
plt.close()


print '======================================================='
print 'Median R SMOS, SMAP, SMAP_s1, S1, SMAp_L4, CHESS, ASCAT'
median = [np.median(e) for e in R_box]

print median

######plot box plot for ubRMSE#################
R_box = [crmse_smos[mask_nan_smos], crmse_smap[mask_nan_smap],crmse_smap_s1[mask_nan_smap_s1],crmse_s1[mask_nan_s1],crmse_smap_l4[mask_nan_smap_l4],crmse_chess[mask_nan_chess],crmse_ascat[mask_nan_ascat]]

fig = plt.figure(figsize=(4,3))
ax1 = fig.add_subplot(111)
#ax1.boxplot(R_box,showfliers=True)
ax1.boxplot(R_box)
ax1.set_xticks([1,2,3,4,5,6,7])
ax1.set_xticklabels(['SMOS L4','SMAP L3E','SMAP/Sentinel 1','Sentinel 1','SMAP L4','JULES-CHESS','ASCAT'],fontsize = 11)
ax1.set_ylim([-0.01,0.2])
ax1.yaxis.set_tick_params(labelsize=11)
ax1.set_ylabel('ubRMSE'+ '\n'+ r'$\mathrm{(m^3/m^3)}$',fontsize = 11)
ax1.grid(True)
fig.autofmt_xdate()
plt.savefig(plot_dir + 'box_crmse_no_collocation__no_seasonality_with_ASCAT_'+version+'.png',dpi=400,bbox_inches='tight')
plt.close()


print '\n======================================================='
print 'Median ubRMSE SMOS, SMAP, SMAP_s1, S1, SMAp_L4, CHESS, ASCAT'
median = [np.median(e) for e in R_box]

print median

R_box = [rmse_smos[mask_nan_smos], rmse_smap[mask_nan_smap],rmse_smap_s1[mask_nan_smap_s1],rmse_s1[mask_nan_s1],rmse_smap_l4[mask_nan_smap_l4],rmse_chess[mask_nan_chess],rmse_ascat[mask_nan_ascat]]

fig = plt.figure(figsize=(4,3))
ax1 = fig.add_subplot(111)
#ax1.boxplot(R_box,showfliers=True)
ax1.boxplot(R_box)
ax1.set_xticks([1,2,3,4,5,6,7])
ax1.set_xticklabels(['SMOS L4','SMAP L3E','SMAP/Sentinel 1','Sentinel 1','SMAP L4','JULES-CHESS','ASCAT'],fontsize = 11)
ax1.set_ylim([-0.01,0.2])  #0.7, 0.2
ax1.yaxis.set_tick_params(labelsize=11)
ax1.set_ylabel('RMSE'+ '\n'+ r'$\mathrm{(m^3/m^3)}$',fontsize = 11)
ax1.grid(True)
fig.autofmt_xdate()
plt.savefig(plot_dir + 'box_rmse_no_collocation__no_seasonality_with_ASCAT_'+version+'.png',dpi=400,bbox_inches='tight')
plt.close()



R_box = [bias_smos[mask_nan_smos], bias_smap[mask_nan_smap],bias_smap_s1[mask_nan_smap_s1],bias_s1[mask_nan_s1],bias_smap_l4[mask_nan_smap_l4],bias_chess[mask_nan_chess],bias_ascat[mask_nan_ascat]]

fig = plt.figure(figsize=(4,3))
ax1 = fig.add_subplot(111)
#ax1.boxplot(R_box,showfliers=True)
ax1.boxplot(R_box)
ax1.set_xticks([1,2,3,4,5,6,7])
ax1.set_xticklabels(['SMOS L4','SMAP L3E','SMAP/Sentinel 1','Sentinel 1','SMAP L4','JULES-CHESS','ASCAT'],fontsize = 11)
ax1.set_ylim([-0.55,0.55])
ax1.yaxis.set_tick_params(labelsize=11)
ax1.set_ylabel('Bias'+ '\n'+ r'$\mathrm{(m^3/m^3)}$',fontsize = 11)
ax1.grid(True)
fig.autofmt_xdate()
plt.savefig(plot_dir + 'box_bias_no_collocation__no_seasonality_with_ASCAT_'+version+'.png',dpi=400,bbox_inches='tight')
plt.close()



#################calcuate median value in box plots##########################################

# avg_r_chess = np.nanmedian(rvalue_chess[mask_nan_chess])   #####np.nanmean
# avg_r_smap = np.nanmedian(rvalue_smap[mask_nan_smap])
# avg_r_smos = np.nanmedian(rvalue_smos[mask_nan_smos])
# avg_r_s1 = np.nanmedian(rvalue_s1[mask_nan_s1])
# avg_r_smap_s1 = np.nanmedian(rvalue_smap_s1[mask_nan_smap_s1])
# avg_r_smap_l4 = np.nanmedian(rvalue_smap_l4[mask_nan_smap_l4])

# print avg_r_smos, avg_r_smap, avg_r_smap_s1, avg_r_s1, avg_r_smap_l4, avg_r_chess

# std_chess = np.nanstd(rvalue_chess[mask_nan])
# std_smap = np.nanstd(rvalue_smap[mask_nan])
# std_smos= np.nanstd(rvalue_smos[mask_nan])
# std_s1 = np.nanstd(rvalue_s1[mask_nan])
# std_smap_s1 = np.nanstd(rvalue_smap_s1[mask_nan])

# # #calc avg and standard deviation of CRMSE value over all stations
# avg_crmse_chess = np.nanmedian(crmse_chess[mask_nan_chess])         #np.nanmean
# avg_crmse_smap = np.nanmedian(crmse_smap[mask_nan_smap])
# avg_crmse_smos= np.nanmedian(crmse_smos[mask_nan_smos])
# avg_crmse_s1 = np.nanmedian(crmse_s1[mask_nan_s1])
# avg_crmse_smap_s1 = np.nanmedian(crmse_smap_s1[mask_nan_smap_s1])
# avg_crmse_smap_l4 = np.nanmedian(crmse_smap_l4[mask_nan_smap_l4])

# print avg_crmse_smos, avg_crmse_smap, avg_crmse_smap_s1, avg_crmse_s1,avg_crmse_smap_l4, avg_crmse_chess


# std_crmse_chess = np.nanstd(crmse_chess[mask_nan])
# std_crmse_smap = np.nanstd(crmse_smap[mask_nan])
# std_crmse_smos = np.nanstd(crmse_smos[mask_nan])
# std_crmse_s1 = np.nanstd(crmse_s1[mask_nan])
# std_crmse_smap_s1 = np.nanstd(crmse_smap_s1[mask_nan])

# calc avg and standard deviation of RMSE value over all stations
# avg_rmse_chess = np.nanmedian(rmse_chess[mask_nan_chess])         #np.nanmean
# avg_rmse_smap = np.nanmedian(rmse_smap[mask_nan_smap])
# avg_rmse_smos= np.nanmedian(rmse_smos[mask_nan_smos])
# avg_rmse_s1 = np.nanmedian(rmse_s1[mask_nan_s1])
# avg_rmse_smap_s1 = np.nanmedian(rmse_smap_s1[mask_nan_smap_s1])
# avg_rmse_smap_l4 = np.nanmedian(rmse_smap_l4[mask_nan_smap_l4])

# print avg_rmse_smos, avg_rmse_smap, avg_rmse_smap_s1, avg_rmse_s1,avg_rmse_smap_l4, avg_rmse_chess

# calc avg and standard deviation of BIAS value over all stations
# avg_bias_chess = np.nanmedian(bias_chess[mask_nan_chess])         #np.nanmean
# avg_bias_smap = np.nanmedian(bias_smap[mask_nan_smap])
# avg_bias_smos= np.nanmedian(bias_smos[mask_nan_smos])
# avg_bias_s1 = np.nanmedian(bias_s1[mask_nan_s1])
# avg_bias_smap_s1 = np.nanmedian(bias_smap_s1[mask_nan_smap_s1])
# avg_bias_smap_l4 = np.nanmedian(bias_smap_l4[mask_nan_smap_l4])

# print avg_bias_smos, avg_bias_smap, avg_bias_smap_s1, avg_bias_s1,avg_bias_smap_l4, avg_bias_chess




#####################################################################################
# Bar Plot without seasonality$
#Plot Bar statistics of soil moisture comparision at each station #
#####################################################################################


error_config = {'ecolor': 'k',
                'lw': 0.5,
                'capsize' : 0.5,
                'capthick' :0.5}
font = {'weight' : 'normal',
    'size'   : 6}
mpl.rc('font',**font)

fig = plt.figure(figsize=(7,3.5))
ax1 = plt.subplot2grid((6,1),(0,0))
ax2 = plt.subplot2grid((6,1),(1,0), sharex=ax1)

#bar_width = 0.28
bar_width = 0.2
index = np.asarray(range(0,len(rvalue_smap)*2,2))

#######the number 60 comes from index value above
#cRMSE
rects1 = ax1.bar(index-bar_width*2,crmse_smos,bar_width,label = 'SMOS L4',color = 'green', edgecolor = 'none')
rects2 = ax1.bar(index-bar_width,crmse_smap,bar_width,label = 'SMAP L3E',color = 'red', edgecolor = 'none')
rects4 = ax1.bar(index,crmse_smap_s1,bar_width,label = 'SMAP/Sentinel 1',color = 'blue', edgecolor = 'none')
rects5 = ax1.bar(index+bar_width,crmse_s1,bar_width,label = 'Sentinel 1',color = 'grey', edgecolor = 'none')
rects6 = ax1.bar(index+bar_width*2,crmse_smap_l4,bar_width,label = 'SMAP L4',color = 'orange', edgecolor = 'none')
rects7 = ax1.bar(index+bar_width*3,crmse_chess,bar_width,label = 'JULES-CHESS',color = 'purple', edgecolor = 'none')
rects8 = ax2.bar(index+bar_width*4,crmse_ascat,bar_width,label = 'ASCAT',color = 'black', edgecolor = 'none')

ax1.set_ylabel('ubRMSE'+ '\n'+ r'$\,\mathrm{(m^3/m^3)}$',fontsize=6)
ax1.set_ylim([0,0.25])
ax1.grid(False)
ax1.yaxis.set_ticks(np.arange(0,0.3,0.1))


#R VALUE
rects1 = ax2.bar(index-bar_width*2,rvalue_smos,bar_width,label = 'SMOS L4',color = 'green', edgecolor = 'none')
rects2 = ax2.bar(index-bar_width,rvalue_smap,bar_width,label = 'SMAP L3E',color = 'red', edgecolor = 'none')
rects4 = ax2.bar(index,rvalue_smap_s1,bar_width,label = 'SMAP/Sentinel 1',color = 'blue', edgecolor = 'none')
rects5 = ax2.bar(index+bar_width,rvalue_s1,bar_width,label = 'Sentinel 1',color = 'grey', edgecolor = 'none')
rects6 = ax2.bar(index+bar_width*2,rvalue_smap_l4,bar_width,label = 'SMAP L4',color = 'orange', edgecolor = 'none')
rects7 = ax2.bar(index+bar_width*3,rvalue_chess,bar_width,label = 'JULES-CHESS',color = 'purple', edgecolor = 'none')
rects8 = ax2.bar(index+bar_width*4,rvalue_ascat,bar_width,label = 'ASCAT',color = 'black', edgecolor = 'none')

lgd = plt.legend(bbox_to_anchor=(0.5, -0.69),loc=9,ncol=6,fontsize=6)
ax2.set_ylabel(r'R',fontsize=6)
ax2.grid(False)
ax2.yaxis.set_ticks(np.arange(0,1.1,0.4))
ax2.set_ylim([0,1])

index = np.asarray(range(0,len(rvalue_smos)*2,2))
#####del site WYTH1, due to no collocated data for all satellite products
station_names=np.delete(station_names,[40])
plt.xticks(index+bar_width,station_names, rotation=40,fontsize=5)  #
plt.xlim([-2,(index[-1])+2.5])

for ax in [ax1]:
    plt.setp(ax.get_xticklabels(), visible=False)

for ax in [ax1,ax2]:
    ax.set_yticks(ax.get_yticks()[1:])
plt.show()
plt.savefig(plot_dir+'no_collocation_error_bar_no_seasonality_with ASCAT_'+version+'.png',dpi=600,bbox_inches='tight')

plt.close()

#====================================

error_config = {'ecolor': 'k',
                'lw': 0.5,
                'capsize' : 0.5,
                'capthick' :0.5}
font = {'weight' : 'normal',
    'size'   : 6}
mpl.rc('font',**font)

fig = plt.figure(figsize=(7,3.5))

####Including RMSE
# ax1 = plt.subplot2grid((6,1),(0,0))
# ax2 = plt.subplot2grid((6,1),(1,0), sharex=ax1)
# ax3 = plt.subplot2grid((6,1),(2,0), sharex=ax1)
# ax4 = plt.subplot2grid((6,1),(3,0), sharex=ax1)

#Without RMSE
ax2 = plt.subplot2grid((6,1),(0,0))
#ax3 = plt.subplot2grid((6,1),(1,0), sharex=ax2)
ax4 = plt.subplot2grid((6,1),(2,0), sharex=ax2)


#bar_width = 0.28
bar_width = 0.2
index = np.asarray(range(0,len(rvalue_smap)*2,2))

#######the number 60 comes from index value above

#RMSE
# rects1 = ax1.bar(index-bar_width*2,rmse_smos,bar_width,label = 'SMOS L4',color = 'green', edgecolor = 'none')
# rects2 = ax1.bar(index-bar_width,rmse_smap,bar_width,label = 'SMAP L3E',color = 'red', edgecolor = 'none')
# rects4 = ax1.bar(index,rmse_smap_s1,bar_width,label = 'SMAP/Sentinel 1',color = 'blue', edgecolor = 'none')
# rects5 = ax1.bar(index+bar_width,rmse_s1,bar_width,label = 'Sentinel 1',color = 'grey', edgecolor = 'none')
# rects6 = ax1.bar(index+bar_width*2,rmse_smap_l4,bar_width,label = 'SMAP L4',color = 'orange', edgecolor = 'none')
# rects7 = ax1.bar(index+bar_width*3,rmse_chess,bar_width,label = 'JULES-CHESS',color = 'purple', edgecolor = 'none')

# ax1.set_ylabel('RMSE'+ '\n'+ r'$\,\mathrm{(m^3/m^3)}$',fontsize=6)
# ax1.set_ylim([0,0.7])
# ax1.grid(False)
# ax1.yaxis.set_ticks(np.arange(0,0.7,0.2))



#cRMSE
rects1 = ax2.bar(index-bar_width*2,crmse_smos,bar_width,label = 'SMOS L4',color = 'green', edgecolor = 'none')
rects2 = ax2.bar(index-bar_width,crmse_smap,bar_width,label = 'SMAP L3E',color = 'red', edgecolor = 'none')
rects4 = ax2.bar(index,crmse_smap_s1,bar_width,label = 'SMAP/Sentinel 1',color = 'blue', edgecolor = 'none')
rects5 = ax2.bar(index+bar_width,crmse_s1,bar_width,label = 'Sentinel 1',color = 'grey', edgecolor = 'none')
rects6 = ax2.bar(index+bar_width*2,crmse_smap_l4,bar_width,label = 'SMAP L4',color = 'orange', edgecolor = 'none')
rects7 = ax2.bar(index+bar_width*3,crmse_chess,bar_width,label = 'JULES-CHESS',color = 'purple', edgecolor = 'none')
rects8 = ax2.bar(index+bar_width*4,crmse_ascat,bar_width,label = 'ASCAT',color = 'black', edgecolor = 'none')

ax2.set_ylabel('ubRMSE'+ '\n'+ r'$\,\mathrm{(m^3/m^3)}$',fontsize=6)
ax2.set_ylim([0,0.25])
ax2.grid(False)
ax2.yaxis.set_ticks(np.arange(0,0.3,0.1))

#Bias
##rects1 = ax3.bar(index-bar_width*2,bias_smos,bar_width,label = 'SMOS L4',color = 'green', edgecolor = 'none')
##rects2 = ax3.bar(index-bar_width,bias_smap,bar_width,label = 'SMAP L3E',color = 'red', edgecolor = 'none')
##rects4 = ax3.bar(index,bias_smap_s1,bar_width,label = 'SMAP/Sentinel 1',color = 'blue', edgecolor = 'none')
##rects5 = ax3.bar(index+bar_width,bias_s1,bar_width,label = 'Sentinel 1',color = 'grey', edgecolor = 'none')
##rects6 = ax3.bar(index+bar_width*2,bias_smap_l4,bar_width,label = 'SMAP L4',color = 'orange', edgecolor = 'none')
##rects7 = ax3.bar(index+bar_width*3,bias_chess,bar_width,label = 'JULES-CHESS',color = 'purple', edgecolor = 'none')
##rects8 = ax3.bar(index+bar_width*4,bias_ascat,bar_width,label = 'ASCAT',color = 'black', edgecolor = 'none')
##
##
##ax3.set_ylabel('Bias'+ '\n'+ r'$\,\mathrm{(m^3/m^3)}$',fontsize=6)
##ax3.grid(False)
##ax3.yaxis.set_ticks(np.arange(-0.6,0.6,0.3))
##ax3.set_ylim([-0.6,0.6])


#R VALUE
rects1 = ax4.bar(index-bar_width*2,rvalue_smos,bar_width,label = 'SMOS L4',color = 'green', edgecolor = 'none')
rects2 = ax4.bar(index-bar_width,rvalue_smap,bar_width,label = 'SMAP L3E',color = 'red', edgecolor = 'none')
rects4 = ax4.bar(index,rvalue_smap_s1,bar_width,label = 'SMAP/Sentinel 1',color = 'blue', edgecolor = 'none')
rects5 = ax4.bar(index+bar_width,rvalue_s1,bar_width,label = 'Sentinel 1',color = 'grey', edgecolor = 'none')
rects6 = ax4.bar(index+bar_width*2,rvalue_smap_l4,bar_width,label = 'SMAP L4',color = 'orange', edgecolor = 'none')
rects7 = ax4.bar(index+bar_width*3,rvalue_chess,bar_width,label = 'JULES-CHESS',color = 'purple', edgecolor = 'none')
rects8 = ax4.bar(index+bar_width*4,rvalue_ascat,bar_width,label = 'ASCAT',color = 'black', edgecolor = 'none')



lgd = plt.legend(bbox_to_anchor=(0.5, -0.69),loc=9,ncol=7,fontsize=6)
ax4.set_ylabel(r'R',fontsize=6)
ax4.grid(False)
ax4.yaxis.set_ticks(np.arange(0,1.1,0.4))
ax4.set_ylim([0,1])

index = np.asarray(range(0,len(rvalue_smos)*2,2))
#####del site WYTH1, due to no collocated data for all satellite products
station_names=np.delete(station_names,[40])
plt.xticks(index+bar_width,station_names, rotation=40,fontsize=5)  #
plt.xlim([-2,(index[-1])+2.5])

for ax in [ax2]:
    plt.setp(ax.get_xticklabels(), visible=False)

for ax in [ax2,ax4]:
    ax.set_yticks(ax.get_yticks()[1:])
plt.show()
plt.savefig(plot_dir+'no_collocation_error_bar_no_seasonality_with_ASCAT_2_'+version+'.png',dpi=600,bbox_inches='tight')
plt.close()
#========================================================



#####################################################################################
# Heatmap with seasonality$
# Heatmap statistics of soil moisture comparision at each station #
#####################################################################################


error_config = {'ecolor': 'k',
                'lw': 0.5,
                'capsize' : 0.5,
                'capthick' :0.5}
font = {'weight' : 'normal',
    'size'   : 6}
mpl.rc('font',**font)

fig = plt.figure(figsize=(10,10))

####Including RMSE
# ax1 = plt.subplot2grid((6,1),(0,0))
# ax2 = plt.subplot2grid((6,1),(1,0), sharex=ax1)
# ax3 = plt.subplot2grid((6,1),(2,0), sharex=ax1)
# ax4 = plt.subplot2grid((6,1),(3,0), sharex=ax1)

#Without RMSE
ax2 = plt.subplot2grid((6,1),(0,0))
ax4 = plt.subplot2grid((6,1),(1,0), sharex=ax2)
#ax4 = plt.subplot2grid((6,1),(2,0), sharex=ax2)


#bar_width = 0.28
bar_width = 0.2
index = np.asarray(range(0,len(rvalue_smap)*2,2))

#######the number 60 comes from index value above


#cRMSE
heatmap = np.empty((7,38))
heatmap[:] = np.nan

for i in range(len(crmse_smos)):
    heatmap[0,i] = crmse_smos[i]
    heatmap[1,i] = crmse_smap[i]
    heatmap[2,i] = crmse_smap_s1[i]
    heatmap[3,i] = crmse_s1[i]
    heatmap[4,i] = crmse_smap_l4[i]
    heatmap[5,i] = crmse_chess[i]
    heatmap[6,i] = crmse_ascat[i]
myheatmap = np.ma.masked_where(np.isnan(heatmap),heatmap)

#cmap = plt.get_cmap('gist_earth_r')
cmap = plt.cm.RdYlBu_r
c2=ax2.pcolor(myheatmap, edgecolors='k', linewidths=0, cmap=cmap, vmin=0.0, vmax=0.22)
show_values(c2,ax2)
#fig.colorbar(c2, ax=ax2)
cb2=plt.colorbar(c2, ax=ax2,aspect=10)
from matplotlib import ticker
tick_locator = ticker.MaxNLocator(nbins=5)
cb2.locator = tick_locator
cb2.update_ticks()

# Hide major tick labels
ax2.yaxis.set_major_formatter(ticker.NullFormatter())

# Customize minor tick labels
ax2.yaxis.set_minor_locator(ticker.FixedLocator([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5]))
ax2.tick_params(which='minor', length=0)
ax2.yaxis.set_minor_formatter(ticker.FixedFormatter(['SMOS L4', 'SMAP L3E', 'SMAP/Sentinel 1', 'Sentinel 1', \
                                                    'SMAP L4', 'JULES-CHESS','ASCAT']))
ax2.title.set_text('ubRMSE'+ r'$\,\mathrm{(m^3/m^3)}$')

###Bias
##heatmap = np.empty((7,38))
##heatmap[:] = np.nan
##
##for i in range(len(bias_smos)):
##    heatmap[0,i] = bias_smos[i]
##    heatmap[1,i] = bias_smap[i]
##    heatmap[2,i] = bias_smap_s1[i]
##    heatmap[3,i] = bias_s1[i]
##    heatmap[4,i] = bias_smap_l4[i]
##    heatmap[5,i] = bias_chess[i]
##    heatmap[6,i] = bias_ascat[i]
##myheatmap = np.ma.masked_where(np.isnan(heatmap),heatmap)
##
#####im = ax2.imshow(heatmap, interpolation='nearest')
##c3=ax3.pcolor(myheatmap, edgecolors='k', linewidths=0, cmap='gist_earth', vmin=-0.57, vmax=0.57)
##show_values(c3,ax3)
###fig.colorbar(c3, ax=ax3)
##plt.colorbar(c3, ax=ax3,aspect=10)
##
### Hide major tick labels
##ax3.yaxis.set_major_formatter(ticker.NullFormatter())
##
### Customize minor tick labels
##ax3.yaxis.set_minor_locator(ticker.FixedLocator([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5]))
##ax3.tick_params(which='minor', length=0)
##ax3.yaxis.set_minor_formatter(ticker.FixedFormatter(['SMOS L4', 'SMAP L3E', 'SMAP/Sentinel 1', 'Sentinel 1', \
##                                                    'SMAP L4', 'JULES-CHESS','ASCAT']))
##ax3.title.set_text('Bias'+ r'$\,\mathrm{(m^3/m^3)}$')

#R VALUE
heatmap = np.empty((7,38))
heatmap[:] = np.nan

for i in range(len(rvalue_smos)):
    heatmap[0,i] = rvalue_smos[i]
    heatmap[1,i] = rvalue_smap[i]
    heatmap[2,i] = rvalue_smap_s1[i]
    heatmap[3,i] = rvalue_s1[i]
    heatmap[4,i] = rvalue_smap_l4[i]
    heatmap[5,i] = rvalue_chess[i]
    heatmap[6,i] = rvalue_ascat[i]
myheatmap = np.ma.masked_where(np.isnan(heatmap),heatmap)

#c4=ax4.pcolor(myheatmap, edgecolors='k', linewidths=0, cmap='gist_earth', vmin=0.0, vmax=1)
c4=ax4.pcolor(myheatmap, edgecolors='k', linewidths=0, cmap=plt.cm.RdYlBu, vmin=0.0, vmax=1)
show_values(c4,ax4)
#fig.colorbar(c4, ax=ax4)
plt.colorbar(c4, ax=ax4,aspect=10)

# Hide major tick labels
ax4.yaxis.set_major_formatter(ticker.NullFormatter())

# Customize minor tick labels
ax4.yaxis.set_minor_locator(ticker.FixedLocator([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5]))
ax4.tick_params(which='minor', length=0)
ax4.yaxis.set_minor_formatter(ticker.FixedFormatter(['SMOS L4', 'SMAP L3E', 'SMAP/Sentinel 1', 'Sentinel 1', \
                                                    'SMAP L4', 'JULES-CHESS','ASCAT']))
ax4.title.set_text(r'R')

index = np.asarray(range(0,len(rvalue_smos)+1,1))
#####del site WYTH1, due to no collocated data for all satellite products
station_names=np.delete(station_names,[40])
plt.xticks(index,station_names, rotation=40,fontsize=5)  #
plt.xlim([0,(index[-1])])

for ax in [ax2,ax4]:
    plt.setp(ax.get_xticklabels(), visible=False)

for ax in [ax2,ax4]:
    ax.set_yticks(ax.get_yticks()[1:])
plt.show()
plt.savefig(plot_dir+'no_collocation_error_HEATMAP_no_seasonality_with_ASCAT_NEW_'+version+'.png',dpi=600,bbox_inches='tight')
plt.close()








