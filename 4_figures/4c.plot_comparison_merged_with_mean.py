# Script for plotting all collocated products, inlucding box plot, bar plot, and TC standadized erro bar and box plot

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


# VWC calculated with GLDAS:
GLDAS = False

# Or the option to use Hamburg version of HWSD
hamburg = False

if (hamburg==True) & (GLDAS==True):
    print '\nWARNING: Choose your porosity dataset correctly.\n GLDAS will used.\n'

# equation with Organic matter content: yes (True) or no (False)
oc_eq = True
if oc_eq==True:
    oc_text = 'with_oc'
else:
    oc_text = 'without_oc'

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
merged_dir = '/prj/hydrojules/data/soil_moisture/merged/Paper_review_2/'
if GLDAS == False:
    if hamburg==True:
        plot_dir = '/prj/hydrojules/users/malngu/plots/plots_for_review/number_4/Hamburg/'
    else:
        plot_dir = '/prj/hydrojules/users/malngu/plots/plots_for_review/number_4_merged_and_mean/'+method + '_' + bulk_den + '/'
else:
    plot_dir = '/prj/hydrojules/users/malngu/plots/plots_for_review/number_4/GLDAS/'


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
file_smos1km = SM_data_dir + '/ascat_VWC_linear_T_BULK_DEN_porosity_Toth_et_al.nc'
file_chess1km = SM_data_dir + '/chess_1km_short.nc'
file_smap_s1 = SM_data_dir + '/smap_s1_all_v2.nc'
file_mean = '/prj/hydrojules/data/soil_moisture/merged/Paper_review/merge_9km_chess_smap_mean_short.nc'
file_smapl4 = SM_data_dir + '/smap_L4_9km_short.nc'
#file_ascat = SM_data_dir + '/ascat_12.5km.nc'
if GLDAS == False:
    if hamburg==True:
        file_ascat = SM_data_dir + '/ascat_VWC_nearest_Hamburg_HWSD.nc'
    else:
        file_ascat = SM_data_dir + '/ascat_VWC_'+method+'_'+bulk_den+'_porosity_Toth_et_al.nc'
else:
    file_ascat = SM_data_dir + '/ascat_VWC_nearest_GLDAS_.nc'
file_tc_merge = merged_dir + 'merge_12.5km_tc_ref_smap.nc'
file_mean2 = merged_dir + 'merge_9km_chess_smap_ascat_mean_short.nc'

smap9km = extract_point(file_smap9km,date_range_cf,38,lat,lon,'sm',1)
for k in range(38):
   smap9km = smap9km.rename(columns={k:'smap9km' + station_names[k] })

smos1km = extract_point(file_smos1km,date_range_cf,38,lat,lon,'sm',1)
for k in range(38):
   smos1km = smos1km.rename(columns={k:'smos1km' + station_names[k] })

chess1km = extract_point(file_chess1km,date_range_cf,38,lat,lon,'sm',100)
for k in range(38):
   chess1km = chess1km.rename(columns={k:'chess1km' + station_names[k] })

tc_merge = extract_point(file_tc_merge,date_range_cf,38,lat,lon,'sm',1)
for k in range(38):
   tc_merge = tc_merge.rename(columns={k:'tc_merge' + station_names[k] })

s1 = extract_point(file_mean,date_range_cf,38,lat,lon,'sm',1)
for k in range(38):
   s1 = s1.rename(columns={k:'s1' + station_names[k] })

mean = extract_point(file_mean2,date_range_cf,38,lat,lon,'sm',1)
for k in range(38):
   mean = mean.rename(columns={k:'mean' + station_names[k] })

smap_l4 = extract_point(file_smapl4,date_range_cf,38,lat,lon,'sm',1)
for k in range(38):
   smap_l4 = smap_l4.rename(columns={k:'smap_l4' + station_names[k] })

##ascat12_5km = extract_point(file_ascat,date_range_cf,38,lat,lon,'sm',1)
##for k in range(38):
##   ascat12_5km = ascat12_5km.rename(columns={k:'ascat12_5km' + station_names[k] })


#################################calcuate error statistics for each station###########
######################################################################################

new = station_data
new = new.join(smap9km)
new = new.join(smos1km)
new = new.join(chess1km)
new = new.join(tc_merge)
new = new.join(s1)
new = new.join(smap_l4)
new = new.join(mean)

############### Remove seasonality using moving window of 35 days, set threshold to 6 days#####################
###for non-seasonaltiy, set p-value to 0.5 in line 293############################
# for s in new.columns:

#     # anomaly = ts.calc_anomaly(new[s], window_size=30)
#     anomaly = new[s] - mv.moving_average(new[s], window_size=35,min_obs=6)
#     new[s] = anomaly

################################################################################################################
#Calculate Bias,rmse,crmse and R-value between input data and in-situ data for later use in following plots #
################################################################################################################
# create ouput arrays

bias_smap = np.empty([0])
bias_smos = np.empty([0])
bias_chess = np.empty([0])
bias_s1 = np.empty([0])
bias_tc_merge = np.empty([0])
bias_smap_l4 = np.empty([0])
bias_mean = np.empty([0])

crmse_smap = np.empty([0])
crmse_smos = np.empty([0])
crmse_chess = np.empty([0])
crmse_s1 = np.empty([0])
crmse_tc_merge = np.empty([0])
crmse_smap_l4 = np.empty([0])
crmse_mean = np.empty([0])

rmse_smap = np.empty([0])
rmse_smos = np.empty([0])
rmse_chess = np.empty([0])
rmse_s1 = np.empty([0])
rmse_tc_merge = np.empty([0])
rmse_smap_l4 = np.empty([0])
rmse_mean = np.empty([0])

rvalue_smap = np.empty([0])
rvalue_smos = np.empty([0])
rvalue_chess = np.empty([0])
rvalue_s1 = np.empty([0])
rvalue_tc_merge = np.empty([0])
rvalue_smap_l4 = np.empty([0])
rvalue_mean = np.empty([0])

pvalue_smap = np.empty([0])
pvalue_smos = np.empty([0])
pvalue_chess = np.empty([0])
pvalue_s1 = np.empty([0])
pvalue_tc_merge = np.empty([0])
pvalue_smap_l4 = np.empty([0])
pvalue_mean = np.empty([0])

notnan_sm = np.empty([0])

for site in station_names:

     ### collocat all products, to ensure all have same sample size
     mask = ~np.isnan(new[site])& ~np.isnan(new['smap9km'+site])& ~np.isnan(new['chess1km'+site]) & ~np.isnan(new['smos1km'+site])& ~np.isnan(new['s1'+site])& ~np.isnan(new['tc_merge'+site])& ~np.isnan(new['smap_l4'+site])& ~np.isnan(new['mean'+site])
     ## calculate bias of each product
     temp_smap = new['smap9km'+site][mask].mean()- new[site][mask].mean()
     temp_smos = new['smos1km'+site][mask].mean()- new[site][mask].mean()
     temp_chess = new['chess1km'+site][mask].mean()- new[site][mask].mean()
     temp_tc_merge = new['tc_merge'+site][mask].mean()- new[site][mask].mean()
     temp_s1 = new['s1'+site][mask].mean()- new[site][mask].mean()
     temp_smap_l4 = new['smap_l4'+site][mask].mean()- new[site][mask].mean()
     temp_mean = new['mean'+site][mask].mean()- new[site][mask].mean()
     # count the non-nan sample number
     notnan_sm = np.append(notnan_sm,sum(mask))

     bias_smap = np.append(bias_smap,temp_smap)
     if np.sum(mask) == 0:
        slope_smap, intercept_smap, r_value_smap, p_value_smap, std_err_smap = [float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN')]
     else:
        slope_smap, intercept_smap, r_value_smap, p_value_smap, std_err_smap = stats.linregress(new[site][mask],new['smap9km'+site][mask])
     crmse_smap = np.append(crmse_smap,rmse_c(new['smap9km'+site][mask],new[site][mask]))
     rmse_smap = np.append(rmse_smap,rmse(new['smap9km'+site][mask],new[site][mask]))
     rvalue_smap = np.append(rvalue_smap,r_value_smap)
     pvalue_smap = np.append(pvalue_smap,p_value_smap)
     #####################


     bias_smos = np.append(bias_smos,temp_smos)
     if np.sum(mask) == 0:
        slope_smos, intercept_smos, r_value_smos, p_value_smos, std_err_smos = [float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN')]
     else:
        slope_smos, intercept_smos, r_value_smos, p_value_smos, std_err_smos = stats.linregress(new[site][mask],new['smos1km'+site][mask])
     crmse_smos = np.append(crmse_smos,rmse_c(new['smos1km'+site][mask],new[site][mask]))
     rmse_smos = np.append(rmse_smos,rmse(new['smos1km'+site][mask],new[site][mask]))
     rvalue_smos = np.append(rvalue_smos,r_value_smos)
     pvalue_smos = np.append(pvalue_smos,p_value_smos)


     bias_chess = np.append(bias_chess,temp_chess)
     if np.sum(mask) == 0:
        slope_chess, intercept_chess, r_value_chess, p_value_chess, std_err_chess = [float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN')]
     else:
        slope_chess, intercept_chess, r_value_chess, p_value_chess, std_err_chess = stats.linregress(new[site][mask],new['chess1km'+site][mask])
     crmse_chess = np.append(crmse_chess,rmse_c(new['chess1km'+site][mask],new[site][mask]))
     rmse_chess = np.append(rmse_chess,rmse(new['chess1km'+site][mask],new[site][mask]))
     rvalue_chess = np.append(rvalue_chess,r_value_chess)
     pvalue_chess = np.append(pvalue_chess,p_value_chess)


     bias_tc_merge = np.append(bias_tc_merge,temp_tc_merge)
     if np.sum(mask) == 0:
        slope_tc_merge, intercept_tc_merge, r_value_tc_merge, p_value_tc_merge, std_err_tc_merge = [float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN')]
     else:
        slope_tc_merge, intercept_tc_merge, r_value_tc_merge, p_value_tc_merge, std_err_tc_merge = stats.linregress(new[site][mask],new['tc_merge'+site][mask])
     crmse_tc_merge = np.append(crmse_tc_merge,rmse_c(new['tc_merge'+site][mask],new[site][mask]))
     rmse_tc_merge = np.append(rmse_tc_merge,rmse(new['tc_merge'+site][mask],new[site][mask]))
     rvalue_tc_merge = np.append(rvalue_tc_merge,r_value_tc_merge)
     pvalue_tc_merge = np.append(pvalue_tc_merge,p_value_tc_merge)


     bias_s1 = np.append(bias_s1,temp_s1)
     if np.sum(mask) == 0:
        slope_s1, intercept_s1, r_value_s1, p_value_s1, std_err_s1 = [float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN')]
     else:
        slope_s1, intercept_s1, r_value_s1, p_value_s1, std_err_s1 = stats.linregress(new[site][mask],new['s1'+site][mask])
     crmse_s1 = np.append(crmse_s1,rmse_c(new['s1'+site][mask],new[site][mask]))
     rmse_s1 = np.append(rmse_s1,rmse(new['s1'+site][mask],new[site][mask]))
     rvalue_s1 = np.append(rvalue_s1,r_value_s1)
     pvalue_s1 = np.append(pvalue_s1,p_value_s1)

     bias_smap_l4 = np.append(bias_smap_l4,temp_smap_l4)
     if np.sum(mask) == 0:
        slope_smap_l4, intercept_smap_l4, r_value_smap_l4, p_value_smap_l4, std_err_smap_l4 = [float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN')]
     else:
        slope_smap_l4, intercept_smap_l4, r_value_smap_l4, p_value_smap_l4, std_err_smap_l4 = stats.linregress(new[site][mask],new['smap_l4'+site][mask])
     crmse_smap_l4 = np.append(crmse_smap_l4,rmse_c(new['smap_l4'+site][mask],new[site][mask]))
     rmse_smap_l4 = np.append(rmse_smap_l4,rmse(new['smap_l4'+site][mask],new[site][mask]))
     rvalue_smap_l4 = np.append(rvalue_smap_l4,r_value_smap_l4)
     pvalue_smap_l4 = np.append(pvalue_smap_l4,p_value_smap_l4)

     bias_mean = np.append(bias_mean,temp_mean)
     if np.sum(mask) == 0:
        slope_mean, intercept_mean, r_value_mean, p_value_mean, std_err_mean = [float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN')]
     else:
        slope_mean, intercept_mean, r_value_mean, p_value_mean, std_err_mean = stats.linregress(new[site][mask],new['mean'+site][mask])
     crmse_mean = np.append(crmse_mean,rmse_c(new['mean'+site][mask],new[site][mask]))
     rmse_mean = np.append(rmse_mean,rmse(new['mean'+site][mask],new[site][mask]))
     rvalue_mean = np.append(rvalue_mean,r_value_mean)
     pvalue_mean = np.append(pvalue_mean,p_value_mean)



###set the sample size threshold for each station, not used for the  plots below#######
notnan_sm = np.ma.masked_where(notnan_sm<0, notnan_sm)


rvalue_chess = np.ma.masked_where(np.isnan(pvalue_chess)|(pvalue_chess>=0.5), rvalue_chess)
rvalue_smap = np.ma.masked_where(np.isnan(pvalue_smap)|(pvalue_smap>=0.5), rvalue_smap)
rvalue_smos = np.ma.masked_where(np.isnan(pvalue_smos)|(pvalue_smos>=0.5), rvalue_smos)
rvalue_tc_merge= np.ma.masked_where(np.isnan(pvalue_tc_merge)|(pvalue_tc_merge>=0.5), rvalue_tc_merge)
rvalue_s1= np.ma.masked_where(np.isnan(pvalue_s1)|(pvalue_s1>=0.5), rvalue_s1)
rvalue_smap_l4= np.ma.masked_where(np.isnan(pvalue_smap_l4)|(pvalue_smap_l4>=0.5), rvalue_smap_l4)
rvalue_mean= np.ma.masked_where(np.isnan(pvalue_mean)|(pvalue_mean>=0.5), rvalue_mean)
########################################################################################################

rvalue_chess=np.ma.filled(rvalue_chess.astype(float), np.nan)
rvalue_smap=np.ma.filled(rvalue_smap.astype(float), np.nan)
rvalue_smos=np.ma.filled(rvalue_smos.astype(float), np.nan)
rvalue_tc_merge=np.ma.filled(rvalue_tc_merge.astype(float), np.nan)
rvalue_s1=np.ma.filled(rvalue_s1.astype(float), np.nan)
rvalue_smap_l4=np.ma.filled(rvalue_smap_l4.astype(float), np.nan)
rvalue_mean=np.ma.filled(rvalue_mean.astype(float), np.nan)

#######box plots for ubRMSE and R for comparsion between merged SM and original SM###################################################
mask_nan = ~np.isnan(rvalue_smap)&~np.isnan(rvalue_smos)&~np.isnan(rvalue_tc_merge)&~np.isnan(rvalue_s1)&~np.isnan(rvalue_chess)&~np.isnan(rvalue_smap_l4) &~np.isnan(rvalue_mean)






######plot box plot for correlation#################
######plot box plot for correlation#################
######plot box plot for correlation#################
######plot box plot for correlation#################
R_box = [rvalue_tc_merge[mask_nan],rvalue_s1[mask_nan],rvalue_mean[mask_nan]]
fig = plt.figure(figsize=(4,3))
ax1 = fig.add_subplot(111)
#ax1.boxplot(R_box,showfliers=True)
ax1.boxplot(R_box)
ax1.set_xticks([1,2,3])
# ax1.set_xticklabels(['SMOS','SMAP','JULES-CHESS','Merge_tc_merge','Merge_s1'],fontsize = 11)
ax1.set_xticklabels(['Merge','Average1','Average2'],fontsize = 11)
ax1.set_ylim([0,1.1])
ax1.yaxis.set_tick_params(labelsize=11)
ax1.set_ylabel(r'$R$',fontsize = 11)
ax1.grid(True)
fig.autofmt_xdate()
plt.savefig(plot_dir + 'box_corrleation_merge_and_mean_with_seasonality_'+version+'.png',dpi=400,bbox_inches='tight')
# plt.savefig('box_corrleation_merge_no_seasonality.png',dpi=400,bbox_inches='tight')
plt.close()


######plot box plot for ubRMSE#################
R_box = [crmse_tc_merge[mask_nan],crmse_s1[mask_nan],crmse_mean[mask_nan]]

fig = plt.figure(figsize=(4,3))
ax1 = fig.add_subplot(111)
ax1.boxplot(R_box)
#ax1.boxplot(R_box,showfliers=True)
ax1.set_xticks([1,2,3])
ax1.set_xticklabels(['Merge','Average1','Average2'],fontsize = 11)
ax1.set_ylim([-0.01,0.14])
ax1.yaxis.set_tick_params(labelsize=11)
ax1.set_ylabel('ubRMSE'+ '\n'+ r'$\mathrm{(m^3/m^3)}$',fontsize = 11)
ax1.grid(True)
fig.autofmt_xdate()
plt.savefig(plot_dir + 'box_crmse_merge_and_mean_with_seasonality_'+version+'.png',dpi=400,bbox_inches='tight')
# plt.savefig('box_crmse_merge_no_seasonality.png',dpi=400,bbox_inches='tight')
plt.close()

#######box plots for RMSE for comparsion between merged SM and original SM###################################################

R_box = [rmse_tc_merge[mask_nan],rmse_s1[mask_nan],rmse_mean[mask_nan]]

fig = plt.figure(figsize=(4,3))
ax1 = fig.add_subplot(111)
ax1.boxplot(R_box)
ax1.set_xticks([1,2,3])
ax1.set_xticklabels(['Merge','Average1','Average2'],fontsize = 11)
ax1.set_ylim([-0.01,0.7])  #0.7, 0.2
ax1.yaxis.set_tick_params(labelsize=11)
ax1.set_ylabel('RMSE'+ '\n'+ r'$\mathrm{(m^3/m^3)}$',fontsize = 11)
ax1.grid(True)
fig.autofmt_xdate()
plt.savefig(plot_dir +'box_rmse_merge_and_mean_with_seasonality_'+version+'.png',dpi=400,bbox_inches='tight')
# plt.savefig('box_rmse_merge_no_seasonality.png',dpi=400,bbox_inches='tight')
plt.close()


R_box = [bias_tc_merge[mask_nan],bias_s1[mask_nan],bias_mean[mask_nan]]

fig = plt.figure(figsize=(4,3))
ax1 = fig.add_subplot(111)
ax1.boxplot(R_box)
ax1.set_xticks([1,2,3])
ax1.set_xticklabels(['Merge','Average1','Average2'],fontsize = 11)
ax1.set_ylim([-0.55,0.55])
ax1.yaxis.set_tick_params(labelsize=11)
ax1.set_ylabel('Bias'+ '\n'+ r'$\mathrm{(m^3/m^3)}$',fontsize = 11)
ax1.grid(True)
fig.autofmt_xdate()
plt.savefig(plot_dir +'box_bias_merge_and_mean_with_seasonality_'+version+'.png',dpi=400,bbox_inches='tight')
# plt.savefig('box_bias_merge_no_seasonality.png',dpi=400,bbox_inches='tight')
plt.close()


