## merge data based on Triple collocation error statistics

# to derive mean smap value, I used this command:
# cdo -b F64 -timmean -setmissval,nan smap_1km.nc smap_1km_mean.nc

# to derive mean chess value, I used this command:
# cdo -b F64 -timmean -setmissval,nan chess_1km.nc chess_1km_mean.nc

outPutInterpolatedChess = False

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


from multiprocessing import Pool


# The control file should look like something like this:
'''
# path to folder with chess file:
/prj/hydrojules/data/soil_moisture/preprocessed/chess/chess_2d/merged/

# path to folder with smap file:
/prj/hydrojules/data/soil_moisture/preprocessed/smap/smap_merged/

# path to folder with ASCAT file:
/prj/hydrojules/data/soil_moisture/preprocessed/ascat/h115/GB/gridded_data/

# path to folder with b2 scaling factor:
/prj/hydrojules/data/soil_moisture/merged/weights/

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
b2_folder = input_file.readline()[:-1]

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
##    print len(x)
##    print len(y)
##    print len(z)
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

def tcol_error(x, y, z):
    """
    Triple collocation error estimate of three calibrated/scaled
    datasets.

    Parameters
    ----------
    x : numpy.ndarray
        1D numpy array to calculate the errors
    y : numpy.ndarray
        1D numpy array to calculate the errors
    z : numpy.ndarray
        1D numpy array to calculate the errors

    Returns
    -------
    e_x : float
        Triple collocation error for x.
    e_y : float
        Triple collocation error for y.
    e_z : float
        Triple collocation error for z.

    Notes
    -----
    This function estimates the triple collocation error based
    on already scaled/calibrated input data. It follows formula 4
    given in [Scipal2008]_.

    .. math:: \\sigma_{\\varepsilon_x}^2 = \\langle (x-y)(x-z) \\rangle

    .. math:: \\sigma_{\\varepsilon_y}^2 = \\langle (y-x)(y-z) \\rangle

    .. math:: \\sigma_{\\varepsilon_z}^2 = \\langle (z-x)(z-y) \\rangle

    where the :math:`\\langle\\rangle` brackets mean the temporal mean.

    References
    ----------
    .. [Scipal2008] Scipal, K., Holmes, T., De Jeu, R., Naeimi, V., & Wagner, W. (2008). A
       possible solution for the problem of estimating the error structure of global
       soil moisture data sets. Geophysical Research Letters, 35(24), .
    """
    e_x = np.sqrt(np.abs(np.mean((x - y) * (x - z))))
    e_y = np.sqrt(np.abs(np.mean((y - x) * (y - z))))
    e_z = np.sqrt(np.abs(np.mean((z - x) * (z - y))))

    return e_x, e_y, e_z



def TCAmerging(i):
    ########merge different soil moisture pixel by pixel#######################
##print 'Loop through lat/lon...'
##for i in range(len(lat[:])):
    print str(i+1) +' out of '+ str(len(lat[:]))
    for j in range(len(lon[:])):

        if merge_method_TC==True:
############################# merge soil moisture based on Tripple collocation ##########################################################
############################# set the sample size threshold, it should be 100 as suggested by Gruber et al.,2016  ##########################################################
            if len(chess[:,i,j].compressed())>0:
                if len(chess_tc[:,i,j].compressed())>0:
                    sm_cal_dic[str(i)+'_'+str(j)] = np.ones_like(sm_cal[:,i,j])*-999

                    # make the duo smos/chess (or ascat/chess)
                    smos_duo_1 = np.ma.masked_where(np.isnan(chess[:,i,j])|(np.isnan(smos[:,i,j])), smos[:,i,j])
                    chess_duo_1 = np.ma.masked_where(np.isnan(smos_duo_1), chess[:,i,j])

                    smos_merge_duo_1 = np.ma.filled(smos_duo_1.astype(float), np.nan)
                    chess_merge_duo_1 = np.ma.filled(chess_duo_1.astype(float), np.nan)

                    # make the duo smap/chess
                    smap_duo_2 = np.ma.masked_where(np.isnan(chess[:,i,j])|(np.isnan(smap[:,i,j])), smap[:,i,j])
                    chess_duo_2 = np.ma.masked_where(np.isnan(smap_duo_2), chess[:,i,j])

                    smap_merge_duo_2 = np.ma.filled(smap_duo_2.astype(float), np.nan)
                    chess_merge_duo_2 = np.ma.filled(chess_duo_2.astype(float), np.nan)

                    # make the duo smap/smos
                    smap_duo_3 = np.ma.masked_where(np.isnan(smos[:,i,j])|(np.isnan(smap[:,i,j])), smap[:,i,j])
                    smos_duo_3 = np.ma.masked_where(np.isnan(smap_duo_3), smos[:,i,j])

                    smap_merge_duo_3 = np.ma.filled(smap_duo_3.astype(float), np.nan)
                    smos_merge_duo_3 = np.ma.filled(smos_duo_3.astype(float), np.nan)


                    # pdb.set_trace()
                    snr_unscaled,err_var_unscaled,beta_unscaled = tcol_snr(smap_tc[:,i,j].compressed(),chess_tc[:,i,j].compressed(), smos_tc[:,i,j].compressed())

                    ###scaling factor
                    b1 = beta_unscaled[0]
                    b2 = beta_unscaled[1]
                    b3 = beta_unscaled[2]


                    # rescale products (smap = reference)
                    smap_mean = np.nanmean(smap_tc[:,i,j])
                    chess_mean = np.nanmean(chess_tc[:,i,j])
                    smos_mean = np.nanmean(smos_tc[:,i,j])

                    chess_scaled = b2*(chess_tc[:,i,j]-chess_mean)+smap_mean
                    smos_scaled = b3*(smos_tc[:,i,j]-smos_mean)+smap_mean

                    # scale both "duos" time series
                    chess_duo1_scaled = b2*(chess_duo_1-chess_mean)+smap_mean
                    chess_duo2_scaled =  b2*(chess_duo_2-chess_mean)+smap_mean
                    smos_duo1_scaled = b3*(smos_duo_1-smos_mean)+smap_mean
                    smos_duo3_scaled = b3*(smos_duo_3-smos_mean)+smap_mean

                    # full time series of scaled chess (for gap-filling)
                    chess_full_scaled = b2*(chess[:,i,j]-chess_mean)+smap_mean
                    smos_full_scaled = b3*(smos[:,i,j]-smos_mean)+smap_mean

                    # deal with values > 1
                    chess_scaled = np.where(chess_scaled>1,1,chess_scaled)
                    smos_scaled = np.where(smos_scaled>1,1,smos_scaled)
                    chess_duo1_scaled = np.where(chess_duo1_scaled>1,1,chess_duo1_scaled)
                    chess_duo2_scaled = np.where(chess_duo2_scaled>1,1,chess_duo2_scaled)
                    smos_duo1_scaled = np.where(smos_duo1_scaled>1,1,smos_duo1_scaled)
                    smos_duo3_scaled = np.where(smos_duo3_scaled>1,1,smos_duo3_scaled)
                    chess_full_scaled = np.where(chess_full_scaled>1,1,chess_full_scaled)

                    # deal with values < 0
                    chess_scaled = np.where(chess_scaled<0,0,chess_scaled)
                    smos_scaled = np.where(smos_scaled<0,0,smos_scaled)
                    chess_duo1_scaled = np.where(chess_duo1_scaled<0,0,chess_duo1_scaled)
                    chess_duo2_scaled = np.where(chess_duo2_scaled<0,0,chess_duo2_scaled)
                    smos_duo1_scaled = np.where(smos_duo1_scaled<0,0,smos_duo1_scaled)
                    smos_duo3_scaled = np.where(smos_duo3_scaled<0,0,smos_duo3_scaled)
                    chess_full_scaled = np.where(chess_full_scaled<0,0,chess_full_scaled)


                    # calculate err_var for rescaled products
                    #print smap_tc
                    #print chess_scaled
                    #print smos_scaled
                    chess_scaled = np.ma.masked_where((np.isnan(smap_tc[:,i,j]))|(np.isnan(chess_scaled))|(np.isnan(smos_scaled)), chess_scaled)
                    smos_scaled = np.ma.masked_where(np.isnan(chess_scaled), smos_scaled)
                    smap_tc[:,i,j] = np.ma.masked_where(np.isnan(chess_scaled), smap_tc[:,i,j])
                    snr,err_var,beta = tcol_snr(smap_tc[:,i,j].compressed(),chess_scaled.compressed(), smos_scaled.compressed())


                    #weight for each product
                    w1 = err_var[1]*err_var[2]/np.sum([err_var[0]*err_var[1],err_var[1]*err_var[2],err_var[0]*err_var[2]])
                    w2 = err_var[0]*err_var[2]/np.sum([err_var[0]*err_var[1],err_var[1]*err_var[2],err_var[0]*err_var[2]])
                    w3 = err_var[0]*err_var[1]/np.sum([err_var[0]*err_var[1],err_var[1]*err_var[2],err_var[0]*err_var[2]])

                    weight = np.sum([w1 ,w2 , w3])

                    if  (weight >0.9) & (w1<1.1):
                        sm_cal_dic[str(i)+'_'+str(j)] = np.sum([smap_merge[:,i,j]*w1,chess_scaled*w2 ,smos_scaled*w3 ],0)
                    else:
                        sm_cal_dic[str(i)+'_'+str(j)] =  chess_full_scaled      ##np.nanmean()

                    # calculate duo smos/chess
                    w2_duo_1 = err_var[1]/np.sum([err_var[1],err_var[2]])  #chess
                    w3_duo_1 = err_var[2]/np.sum([err_var[1],err_var[2]])  #smos

                    weight = np.sum([w2_duo_1,w3_duo_1])

                    if  (weight >0.9) & (w1<1.1):
                        # fill in where NaN
                        duo_1_sum = np.sum([chess_duo1_scaled*w2_duo_1 ,smos_duo1_scaled*w3_duo_1 ],0)
                        sm_cal_dic[str(i)+'_'+str(j)] = np.ma.filled(sm_cal_dic[str(i)+'_'+str(j)].astype(float), np.nan)
                        sm_cal_dic[str(i)+'_'+str(j)] = np.where(np.isnan(sm_cal_dic[str(i)+'_'+str(j)]),duo_1_sum,sm_cal_dic[str(i)+'_'+str(j)])

                    # calculate duo smap/chess
                    w2_duo_2 = err_var[1]/np.sum([err_var[1],err_var[0]])  #chess
                    w1_duo_2 = err_var[0]/np.sum([err_var[1],err_var[0]])  #smap

                    weight = np.sum([w2_duo_2,w1_duo_2])

                    if  (weight >0.9) & (w1<1.1):
                        # fill in where NaN
                        duo_2_sum = np.sum([chess_duo2_scaled*w2_duo_2 ,smap_duo_2*w1_duo_2],0)
                        sm_cal_dic[str(i)+'_'+str(j)] = np.ma.filled(sm_cal_dic[str(i)+'_'+str(j)].astype(float), np.nan)
                        sm_cal_dic[str(i)+'_'+str(j)] = np.where(np.isnan(sm_cal_dic[str(i)+'_'+str(j)]),duo_2_sum,sm_cal_dic[str(i)+'_'+str(j)])
                    slope_ps, intercept_ps, r_value_ps, p_value_ps, std_err_ps = stats.linregress(smap_tc[:,i,j].compressed(),smos_tc[:,i,j].compressed())
                    slope_pc, intercept_pc, r_value_pc, p_value_pc, std_err_pc = stats.linregress(smap_tc[:,i,j].compressed(),chess_tc[:,i,j].compressed())
                    slope_sc, intercept_sc, r_value_sc, p_value_sc, std_err_sc = stats.linregress(chess_tc[:,i,j].compressed(),smos_tc[:,i,j].compressed())

                    if (p_value_ps>=0.05) & (p_value_pc>=0.05)& (p_value_sc>=0.05):
                        sm_cal_dic[str(i)+'_'+str(j)] = np.ma.filled(sm_cal_dic[str(i)+'_'+str(j)].astype(float), np.nan)
                        sm_cal_dic[str(i)+'_'+str(j)] = chess_full_scaled    #?????????? Not sure if correct....
                    elif (p_value_ps<0.05) & (p_value_pc<0.05)& (p_value_sc>=0.05):
                        sm_cal_dic[str(i)+'_'+str(j)] = smap[:,i,j]
                    elif (p_value_ps<0.05) & (p_value_pc>=0.05)& (p_value_sc<0.05):
                        sm_cal_dic[str(i)+'_'+str(j)] = smos_full_scaled
                    elif (p_value_ps>=0.05) & (p_value_pc<0.05)& (p_value_sc<0.05):
                        sm_cal_dic[str(i)+'_'+str(j)] = chess_full_scaled
                    elif (p_value_ps>=0.05) & (p_value_pc<0.05)& (p_value_sc>=0.05):
                        sm_cal_dic[str(i)+'_'+str(j)] = np.mean([chess_duo2_scaled,smap_duo_2],0)
                    elif (p_value_ps<0.05) & (p_value_pc>=0.05)& (p_value_sc>=0.05):
                        sm_cal_dic[str(i)+'_'+str(j)] = np.mean([smap_duo_3,smos_duo3_scaled],0)
                    elif (p_value_ps>=0.05) & (p_value_pc>=0.05)& (p_value_sc<0.05):
                        sm_cal_dic[str(i)+'_'+str(j)] = np.mean([chess_duo1_scaled,smos_duo1_scaled],0)


                # gap fill with scaled chess whatever is left with Nan
                    sm_cal_dic[str(i)+'_'+str(j)] = np.ma.filled(sm_cal_dic[str(i)+'_'+str(j)].astype(float), np.nan)
                    sm_cal_dic[str(i)+'_'+str(j)] = np.where(np.isnan(sm_cal_dic[str(i)+'_'+str(j)]),chess_full_scaled,sm_cal_dic[str(i)+'_'+str(j)])
                sm_cal_dic[str(i)+'_'+str(j)] = np.ma.filled(sm_cal_dic[str(i)+'_'+str(j)].astype(float), np.nan)
                sm_cal_dic[str(i)+'_'+str(j)] = np.where(sm_cal_dic[str(i)+'_'+str(j)]<0,chess_scaled_interpolated[:,i,j],sm_cal_dic[str(i)+'_'+str(j)])
                sm_cal_dic[str(i)+'_'+str(j)] = np.where(sm_cal_dic[str(i)+'_'+str(j)]>1,chess_scaled_interpolated[:,i,j],sm_cal_dic[str(i)+'_'+str(j)])
                sm_cal_dic[str(i)+'_'+str(j)] = np.where(np.isnan(sm_cal_dic[str(i)+'_'+str(j)]),chess_scaled_interpolated[:,i,j],sm_cal_dic[str(i)+'_'+str(j)])

            else:

                sm_cal_dic[str(i)+'_'+str(j)] = np.ma.filled(sm_cal_dic[str(i)+'_'+str(j)].astype(float), np.nan)
                sm_cal_dic[str(i)+'_'+str(j)] = chess_scaled_interpolated[:,i,j]

####Read soil moisture from different files#############
smos_file = smos_folder + 'ascat_VWC_linear_T_BULK_DEN_porosity_Toth_et_al_1km.nc'
smap_file = smap_folder + 'smap_1km.nc'
chess_file = chess_folder + 'chess_1km.nc'
b2_file = b2_folder + 'b2_interpolated_1km.nc'
# to derive mean smap value, I used this command:
# cdo -b F64 -timmean -setmissval,nan smap_1km.nc smap_1km_mean.nc
# And then I interpolated using python code 0a3_interpolate_smap_mean.py
smap_mean_file = smap_folder + 'smap_1km_mean_interpolated.nc'
# to derive mean chess value, I used this command:
# cdo -b F64 -timmean -setmissval,nan chess_1km.nc chess_1km_mean.nc
chess_mean_file = chess_folder + 'chess_1km_mean.nc'

print 'Reading in ASCAT data...'
dataset_smos = nc.Dataset(smos_file,'r')
smos = np.squeeze(np.array(dataset_smos.variables['sm']))
#read lat/lon data
lat = np.squeeze(np.array(dataset_smos.variables['lat']))
lon = np.squeeze(np.array(dataset_smos.variables['lon']))
tmp_date = np.asarray(dataset_smos.variables['time'])

print 'Reading in SMAP data...'
dataset_smap = nc.Dataset(smap_file,'r')
smap = np.squeeze(np.array(dataset_smap.variables['sm']))

print 'Reading in CHESS data...'
dataset_chess = nc.Dataset(chess_file,'r')
chess = np.squeeze(np.array(dataset_chess.variables['sm']))
#chess should be converted from percent to m3/m3
chess = chess/100.0

print 'Reading in CHESS mean data...'
dataset_chess_mean = nc.Dataset(chess_mean_file,'r')
chess_mean_whole = np.squeeze(np.array(dataset_chess_mean.variables['sm']))
chess_mean_whole = chess_mean_whole/100.0

print 'Reading in SMAP mean data...'
dataset_smap_mean = nc.Dataset(smap_mean_file,'r')
smap_mean_whole = np.squeeze(np.array(dataset_smap_mean.variables['sm']))

print 'Reading in b2 scaling factor data...'
dataset_b2 = nc.Dataset(b2_file,'r')
b2_whole = np.squeeze(np.array(dataset_b2.variables['b2']))

print 'Masking data...'
#masked out the values less than 0 and larger than 1
smos = np.ma.masked_where((smos<0)| (smos>1), smos)
smap = np.ma.masked_where((smap<0)| (smap>1),smap)
chess = np.ma.masked_where((chess<0)|(chess>1), chess)

##smos = np.where(smos<0, np.nan, smos)
##smap = np.where(smap<0, np.nan, smap)
##chess = np.where(chess<0, np.nan, chess)
##
##smos = np.where(smos>1, np.nan, smos)
##smap = np.where(smap>1, np.nan, smap)
##chess = np.where(chess>1, np.nan, chess)

##chess_masked = np.ma.masked_where((chess<0)|(chess>1), chess)

#smap_mean_whole = np.ma.masked_where((smap_mean_whole<0)| (smap_mean_whole>1),smap_mean_whole)
#chess_mean_whole = np.ma.masked_where((chess_mean_whole<0)|(chess_mean_whole>1), chess_mean_whole)
#b2_whole = np.ma.masked_where(b2_whole<0, b2_whole)

print 'Identify collocated data...'
#to make the three datasets have same sample size for each pixel
smos_tc = np.ma.masked_where((np.isnan(smap))|(np.isnan(chess))|(np.isnan(smos)), smos)
smap_tc = np.ma.masked_where(np.isnan(smos_tc), smap)
chess_tc = np.ma.masked_where(np.isnan(smos_tc), chess)

print 'Fill in with NaN...'
smos_merge = np.ma.filled(smos_tc.astype(float), np.nan)
smap_merge = np.ma.filled(smap_tc.astype(float), np.nan)
chess_merge = np.ma.filled(chess_tc.astype(float), np.nan)

print 'Create interpolated rescaled CHESS data...'
# make an interpolated version of scaled chess
chess_scaled_interpolated = b2_whole*(chess-chess_mean_whole)+smap_mean_whole
chess_scaled_interpolated = np.where(chess_scaled_interpolated<0,0,chess_scaled_interpolated)
chess_scaled_interpolated = np.where(chess_scaled_interpolated>1,1,chess_scaled_interpolated)

# Either output interpolated chess data or not:
if outPutInterpolatedChess == True:
    chess_scaled_interpolated_dataset = nc.Dataset(chess_folder + 'chess_scaled_interpolated_1km.nc','w',format='NETCDF4')

    #*******************************************************************************************************
    #*******************************************************************************************************
    # create netcdf of chess_scaled_interpolated
    # create dimensions
    chess_scaled_interpolated_dataset.createDimension('lat',len(lat[:]))
    chess_scaled_interpolated_dataset.createDimension('lon',len(lon[:]))
    chess_scaled_interpolated_dataset.createDimension('time',len(tmp_date))
    # create variables
    lats = chess_scaled_interpolated_dataset.createVariable('lat',np.float32,('lat',))
    lons = chess_scaled_interpolated_dataset.createVariable('lon',np.float32,('lon',))
    times_out = chess_scaled_interpolated_dataset.createVariable('time',np.float32,('time',))

    #sm_cal = dataset.createVariable('sm',np.float32,('time','lat','lon',),fill_value=9.96921e+36,zlib=True)
    sm_cal = chess_scaled_interpolated_dataset.createVariable('sm',np.float32,('time','lat','lon',),fill_value=-999.,zlib=True)

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

    sm_cal[:]=chess_scaled_interpolated
    lats[:]= lat[:]
    lons[:]=lon[:]
    times_out[:] = tmp_date[:]
    # close file
    chess_scaled_interpolated_dataset.close()
    os.chmod(chess_folder + 'chess_scaled_interpolated_1km.nc',0664)
#*******************************************************************************************************
#*******************************************************************************************************

print 'Create empty NetCDF...'
########create merged file###########################
if merge_method_TC==True:
    dataset = nc.Dataset(merged_folder + 'merge_1km_tc_ref_smap_parallel.nc','w',format='NETCDF4')
##
##if merge_method_Mean==True:
##    dataset = nc.Dataset(merged_folder + 'merge_1km_mean.nc','w',format='NETCDF4')

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

# parallelised code for the TCA analysis
print 'Parallel processing starting here...'
sm_cal_dic = dict()

# I get memory error with multiprocessing module
p = Pool(6)
p.map(TCAmerging,range(len(lat[:])))

##from loky import get_reusable_executor
##executor = get_reusable_executor(max_workers=4)
##results = list(executor.map(TCAmerging, range(len(lat[:])), chunksize=16));

print "Fill in grid..."
for i in range(len(lat[:])):
    print str(i+1) +' out of '+ str(len(lat[:]))
    for j in range(len(lon[:])):
        try:
            sm_cal[:,i,j] = sm_cal_dic[str(i)+'_'+str(j)]
        except:
            dummy = 0

sm_interm = np.array(sm_cal)
sm_interm= np.ma.masked_where((sm_interm<0)| (sm_interm>1), sm_interm)
sm_interm=np.ma.filled(sm_interm.astype(float), np.nan)
sm_interm=np.where(np.isnan(sm_interm),chess_scaled_interpolated,sm_interm)
m_mask = np.ma.getmask(chess)
sm_interm= np.ma.masked_array(sm_interm,mask=m_mask)
sm_cal[:]=sm_interm
lats[:]= lat[:]
lons[:]=lon[:]
times_out[:] = tmp_date[:]
# close file
dataset.close()
os.chmod(merged_folder + 'merge_1km_tc_ref_smap_parallel.nc',0664)


