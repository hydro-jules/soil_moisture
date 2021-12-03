## merge data based on Triple collocation error statistics

import datetime
import numpy as np
import pdb
import os, os.path
import sys
import glob
import pytesmo.scaling as scaling
import pytesmo.metrics as metrics
import netCDF4 as nc
from scipy import stats

# The control file should look like something like this:
'''
# path to folder with chess file:
/prj/hydrojules/data/soil_moisture/preprocessed/chess/chess_2d/merged/

# path to folder with smap file:
/prj/hydrojules/data/soil_moisture/preprocessed/smap/smap_merged/

# path to folder with smos file:
/prj/hydrojules/data/soil_moisture/preprocessed/smos/smos_merged/

# path to folder where merged file should be created:
/prj/hydrojules/data/soil_moisture/merged/
'''

input_file = open(sys.argv[1])
# skip first line
input_file.readline()

# next line is the start year
chess_folder = input_file.readline()[:-1]

# skip next 2 lines
input_file.readline()
input_file.readline()

# next line is the end year
smap_folder = input_file.readline()[:-1]

# skip next 2 lines
input_file.readline()
input_file.readline()

# next line is the input path
smos_folder = input_file.readline()[:-1]

# skip next 2 lines
input_file.readline()
input_file.readline()

# next line is the output path
merged_folder = input_file.readline()[:-1]


#############Set flag for two merging schemes: TC or Simply average different products###################
merge_method_TC=True
merge_method_Mean=False


#####Function for Tripple collocation analysis: calculation of SNR, standadized error (err_var), and scaling factor (beta)#########
def tcol_snr(x, y, z, ref_ind=0):

    cov = np.cov(np.vstack((x, y, z)))

    ind = (0, 1, 2, 0, 1, 2)

    no_ref_ind = np.where(np.arange(3) != ref_ind)[0]

    liz = [((cov[i, i] * cov[ind[i + 1], ind[i + 2]]) /
                          (cov[i, ind[i + 1]] * cov[i, ind[i + 2]]) - 1) ** (-1)
                         for i in np.arange(3)]

    liz =map(abs, liz)

    snr = 10 * np.log10(liz)
    err_var = np.array([cov[i, i] -(cov[i, ind[i + 1]] * cov[i, ind[i + 2]]) / cov[ind[i + 1], ind[i + 2]]
        for i in np.arange(3)])

    beta = np.array([cov[ref_ind, no_ref_ind[no_ref_ind != i][0]] /
             cov[i, no_ref_ind[no_ref_ind != i][0]] if i != ref_ind
             else 1 for i in np.arange(3)])
    # pdb.set_trace()
    return snr,np.sqrt(err_var)*np.sqrt(err_var),beta


####Read soil moisture from different files#############
smos_file = smos_folder + 'smos_9km.nc'
smap_file = smap_folder + 'smap_9km.nc'
chess_file = chess_folder + 'chess_9km.nc'


dataset_smos = nc.Dataset(smos_file,'r')
smos = np.squeeze(np.array(dataset_smos.variables['sm']))
#read lat/lon data
lat = np.squeeze(np.array(dataset_smos.variables['lat']))
lon = np.squeeze(np.array(dataset_smos.variables['lon']))
tmp_date = np.asarray(dataset_smos.variables['time'])

dataset_smap = nc.Dataset(smap_file,'r')
smap = np.squeeze(np.array(dataset_smap.variables['sm']))

dataset_chess = nc.Dataset(chess_file,'r')
chess = np.squeeze(np.array(dataset_chess.variables['sm']))
#chess should be converted from percent to m3/m3
chess = chess/100


#masked out the values less than 0 and larger than 1
smos = np.ma.masked_where((smos<0)| (smos>1), smos)
smap = np.ma.masked_where((smap<0)| (smap>1),smap)
chess = np.ma.masked_where((chess<0)|(chess>1), chess)


#to make the three datasets have same sample size for each pixel
smos_tc = np.ma.masked_where((np.isnan(smap))|(np.isnan(chess))|(np.isnan(smos)), smos)
smap_tc = np.ma.masked_where(np.isnan(smos_tc), smap)
chess_tc = np.ma.masked_where(np.isnan(smos_tc), chess)

smos_merge = np.ma.filled(smos_tc.astype(float), np.nan)
smap_merge = np.ma.filled(smap_tc.astype(float), np.nan)
chess_merge = np.ma.filled(chess_tc.astype(float), np.nan)


########create merged file###########################
if merge_method_TC==True:
    dataset = nc.Dataset(merged_folder + 'merge_9km_tc.nc','w',format='NETCDF4')

if merge_method_Mean==True:
    dataset = nc.Dataset(merged_folder + 'merge_9km_mean.nc','w',format='NETCDF4')

# create dimensions
dataset.createDimension('lat',len(lat[:]))
dataset.createDimension('lon',len(lon[:]))
dataset.createDimension('time',len(tmp_date))
# create variables
lats = dataset.createVariable('lat',np.float32,('lat',))
lons = dataset.createVariable('lon',np.float32,('lon',))
times_out = dataset.createVariable('time',np.float32,('time',))

#sm_cal = dataset.createVariable('sm',np.float32,('time','lat','lon',),fill_value=9.96921e+36,zlib=True)
sm_cal = dataset.createVariable('sm',np.float32,('time','lat','lon',),fill_value=-999.,zlib=True)

# create Attributes
times_out.units = 'hours since 1970-01-01 00:00:00.0'
times_out.calendar = 'gregorian'
times_out.long_name = 'time'

sm_cal.units = 'm3/m3'
sm_cal.long_name = 'Soil moisture'

lats.standard_name = 'latitude'
lats.units = 'degrees_north'
lons.units = 'degrees_east'
lons.standard_name = 'longitude'


########merge different soil moisture pixel by pixel#######################

for i in range(len(lat[:])):
    print str(i+1) +' out of '+ str(len(lat[:]))
    for j in range(len(lon[:])):

        if merge_method_TC==True:
############################# merge soil moisture based on Tripple collocation ##########################################################
############################# set the sample size threshold, it should be 100 as suggested by Gruber et al.,2016  ##########################################################
            if len(smos_tc[:,i,j].compressed())>100:
                # pdb.set_trace()
                snr,err_var,beta = tcol_snr(smap_tc[:,i,j].compressed(),chess_tc[:,i,j].compressed(), smos_tc[:,i,j].compressed())

                ###scaling factor
                b1 = beta[0]
                b2 = beta[1]
                b3 = beta[2]

                ###weight for each product
                w1 = err_var[1]*err_var[2]/np.sum([err_var[0]*err_var[1],err_var[1]*err_var[2],err_var[0]*err_var[2]])
                w2 = err_var[0]*err_var[2]/np.sum([err_var[0]*err_var[1],err_var[1]*err_var[2],err_var[0]*err_var[2]])
                w3 = err_var[0]*err_var[1]/np.sum([err_var[0]*err_var[1],err_var[1]*err_var[2],err_var[0]*err_var[2]])

                weight = np.sum([w1 ,w2 , w3])

                if  (weight >0.9) & (w1<1.1):
                    sm_cal[:,i,j] = np.sum([smap_merge[:,i,j]*w1*b1,chess_merge[:,i,j]*w2*b2 ,smos_merge[:,i,j]*w3*b3 ],0)
                else:
                    sm_cal[:,i,j] =  chess_merge[:,i,j]      ##np.nanmean()

            elif (len(smos_tc[:,i,j].compressed())>0) & (len(smos_tc[:,i,j].compressed())<100):

                slope_ps, intercept_ps, r_value_ps, p_value_ps, std_err_ps = stats.linregress(smap_tc[:,i,j].compressed(),smos_tc[:,i,j].compressed())
                slope_pc, intercept_pc, r_value_pc, p_value_pc, std_err_pc = stats.linregress(smap_tc[:,i,j].compressed(),chess_tc[:,i,j].compressed())
                slope_sc, intercept_sc, r_value_sc, p_value_sc, std_err_sc = stats.linregress(chess_tc[:,i,j].compressed(),smos_tc[:,i,j].compressed())

                if (p_value_ps<=0.05) & (p_value_pc<=0.05)& (p_value_sc<=0.05):

                    if (r_value_ps >=0.5) & (r_value_pc>=0.5)& (r_value_sc>=0.5):
                        sm_cal[:,i,j] = np.mean([chess_merge[:,i,j],smap_merge[:,i,j] ,smos_merge[:,i,j] ],0)
                    elif (r_value_ps >=0.5) & (r_value_pc>=0.5)& (r_value_sc<0.5):
                        sm_cal[:,i,j] = smap_merge[:,i,j]
                    elif (r_value_ps <0.5) & (r_value_pc>=0.5)& (r_value_sc>=0.5):
                        sm_cal[:,i,j] = chess_merge[:,i,j]
                    elif (r_value_ps >=0.5) & (r_value_pc<0.5)& (r_value_sc>=0.5):
                        sm_cal[:,i,j] = smos_merge[:,i,j]
                    elif (r_value_ps >=0.5) & (r_value_pc<0.5)& (r_value_sc<0.5):
                        sm_cal[:,i,j] = np.mean([chess_merge[:,i,j],smap_merge[:,i,j]],0)
                    elif (r_value_ps <0.5) & (r_value_pc<0.5)& (r_value_sc>=0.5):
                        sm_cal[:,i,j] = np.mean([chess_merge[:,i,j],smos_merge[:,i,j]],0)
                    elif (r_value_ps >=0.5) & (r_value_pc<0.5)& (r_value_sc<0.5):
                        sm_cal[:,i,j] = np.mean([smap_merge[:,i,j],smos_merge[:,i,j]],0)
                    elif (r_value_ps <0.5) & (r_value_pc<0.5)& (r_value_sc<0.5):
                        sm_cal[:,i,j] = chess_merge[:,i,j]
                    else:
                        sm_cal[:,i,j] = chess_merge[:,i,j]
                else:
                    sm_cal[:,i,j] = chess_merge[:,i,j]
            else:
                # sm_cal[:,i,j] = np.nanmean([chess[:,i,j],smap[:,i,j] ,smos[:,i,j] ],0)
                sm_cal[:,i,j] = chess_merge[:,i,j]

############################# merge soil moisture based on average them ##########################################################
        if merge_method_Mean==True:
                sm_cal[:,i,j] = np.nanmean([chess[:,i,j],smap[:,i,j] ,smos[:,i,j] ],0)


sm_cal= np.ma.masked_where((sm_cal<0)| (sm_cal>1), sm_cal)
sm_cal=np.ma.filled(sm_cal.astype(float), np.nan)
lats[:]= lat[:]
lons[:]=lon[:]
times_out[:] = tmp_date[:]
# close file
dataset.close()
os.chmod(merged_folder + 'merge_9km_tc.nc',0664)


