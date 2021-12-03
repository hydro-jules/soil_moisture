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

# path to folder with ascat file:
/prj/hydrojules/data/soil_moisture/preprocessed/ascat/h115/GB/gridded_data/

# path to folder where scaling factor b2 file should be created:
/prj/hydrojules/data/soil_moisture/merged/weights/
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
# this folder is called "smos_folder" for legacy reasons.
# smos dataset was initially used before it was replaced by ascat after the paper was reviewed.
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

####Read soil moisture from different files#############
smos_file = smos_folder + 'ascat_VWC_linear_T_BULK_DEN_porosity_Toth_et_al_1km.nc'
smap_file = smap_folder + 'smap_1km.nc'
chess_file = chess_folder + 'chess_1km.nc'

print 'Read in ASCAT data...'
dataset_smos = nc.Dataset(smos_file,'r')
smos = np.squeeze(np.array(dataset_smos.variables['sm']))
#read lat/lon data
lat = np.squeeze(np.array(dataset_smos.variables['lat']))
lon = np.squeeze(np.array(dataset_smos.variables['lon']))
tmp_date = np.asarray(dataset_smos.variables['time'])

print 'Read in SMAP data...'
dataset_smap = nc.Dataset(smap_file,'r')
smap = np.squeeze(np.array(dataset_smap.variables['sm']))

print 'Read in CHESS data...'
dataset_chess = nc.Dataset(chess_file,'r')
chess = np.squeeze(np.array(dataset_chess.variables['sm']))
#chess should be converted from percent to m3/m3
chess = chess/100.0

print 'Apply mask where outside [0,1] values...'
#masked out the values less than 0 and larger than 1
smos = np.ma.masked_where((smos<0)| (smos>1), smos)
smap = np.ma.masked_where((smap<0)| (smap>1),smap)
chess = np.ma.masked_where((chess<0)|(chess>1), chess)

print 'Apply mask where data not collocated...'
#to make the three datasets have same sample size for each pixel
smos_tc = np.ma.masked_where((np.isnan(smap))|(np.isnan(chess))|(np.isnan(smos)), smos)
smap_tc = np.ma.masked_where(np.isnan(smos_tc), smap)
chess_tc = np.ma.masked_where(np.isnan(smos_tc), chess)
##
##print 'Fill in with NaN...'
##smos_merge = np.ma.filled(smos_tc.astype(float), np.nan)
##smap_merge = np.ma.filled(smap_tc.astype(float), np.nan)
##chess_merge = np.ma.filled(chess_tc.astype(float), np.nan)
##
##print 'Create ASCAT/CHESS duo...'
### make the duo smos/chess (or ascat/chess)
##smos_duo_1 = np.ma.masked_where(np.isnan(chess)|(np.isnan(smos)), smos)
##chess_duo_1 = np.ma.masked_where(np.isnan(smos_duo_1), chess)
##
##smos_merge_duo_1 = np.ma.filled(smos_duo_1.astype(float), np.nan)
##chess_merge_duo_1 = np.ma.filled(chess_duo_1.astype(float), np.nan)
##
##print 'Create SMAP/CHESS duo...'
### make the duo smap/chess
##smap_duo_2 = np.ma.masked_where(np.isnan(chess)|(np.isnan(smap)), smap)
##chess_duo_2 = np.ma.masked_where(np.isnan(smap_duo_2), chess)
##
##smap_merge_duo_2 = np.ma.filled(smap_duo_2.astype(float), np.nan)
##chess_merge_duo_2 = np.ma.filled(chess_duo_2.astype(float), np.nan)
##
##print 'Create ASCAT/SMAP duo...'
### make the duo smap/smos
##smap_duo_3 = np.ma.masked_where(np.isnan(smos)|(np.isnan(smap)), smap)
##smos_duo_3 = np.ma.masked_where(np.isnan(smap_duo_3), smos)
##
##smap_merge_duo_3 = np.ma.filled(smap_duo_3.astype(float), np.nan)
##smos_merge_duo_3 = np.ma.filled(smos_duo_3.astype(float), np.nan)


print 'Create output netCDF file...'
########create merged file###########################
if merge_method_TC==True:
    dataset = nc.Dataset(merged_folder + 'scaling_factor_b2_1km.nc','w',format='NETCDF4')

##if merge_method_Mean==True:
##    dataset = nc.Dataset(merged_folder + 'merge_12.5km_mean.nc','w',format='NETCDF4')

# create dimensions
dataset.createDimension('lat',len(lat[:]))
dataset.createDimension('lon',len(lon[:]))
dataset.createDimension('time',len(tmp_date))
# create variables
lats = dataset.createVariable('lat',np.float32,('lat',))
lons = dataset.createVariable('lon',np.float32,('lon',))
#times_out = dataset.createVariable('time',np.float32,('time',))

#sm_cal = dataset.createVariable('sm',np.float32,('time','lat','lon',),fill_value=9.96921e+36,zlib=True)
sm_cal = dataset.createVariable('b2',np.float32,('lat','lon',),fill_value=-999.,zlib=True)

# create Attributes
##times_out.units = 'hours since 1970-01-01 00:00:00.0'
##times_out.calendar = 'gregorian'
##times_out.long_name = 'time'

sm_cal.units = '-'
sm_cal.long_name = 'b2'

lats.standard_name = 'latitude'
lats.units = 'degrees_north'
lons.units = 'degrees_east'
lons.standard_name = 'longitude'


########merge different soil moisture pixel by pixel#######################

print 'Loop through lat/lon...'
for i in range(len(lat[:])):
    print str(i+1) +' out of '+ str(len(lat[:]))
    for j in range(len(lon[:])):

        if merge_method_TC==True:
############################# merge soil moisture based on Tripple collocation ##########################################################
############################# set the sample size threshold, it should be 100 as suggested by Gruber et al.,2016  ##########################################################
            if len(chess[:,i,j].compressed())>0:
                if len(chess_tc[:,i,j].compressed())>0:
                    # pdb.set_trace()
                    snr_unscaled,err_var_unscaled,beta_unscaled = tcol_snr(smap_tc[:,i,j].compressed(),chess_tc[:,i,j].compressed(), smos_tc[:,i,j].compressed())

                    ###scaling factor
                    b1 = beta_unscaled[0]
                    b2 = beta_unscaled[1]
                    b3 = beta_unscaled[2]

                    sm_cal[i,j] = b2



            else:
                # sm_cal[:,i,j] = np.nanmean([chess[:,i,j],smap[:,i,j] ,smos[:,i,j] ],0)
                #sm_cal[:,i,j] = np.where(np.isnan(sm_cal[:,i,j]),chess_full_scaled,sm_cal[:,i,j])
                sm_cal[i,j] = -999

############################# merge soil moisture based on average them ##########################################################
##        if merge_method_Mean==True:
##                sm_cal[:,i,j] = np.nanmean([chess[:,i,j],smap[:,i,j] ,smos[:,i,j] ],0)



sm_interm = np.array(sm_cal)
sm_interm= np.ma.masked_where(sm_interm==-999, sm_interm)
sm_interm=np.ma.filled(sm_interm.astype(float), np.nan)
sm_cal[:]=sm_interm
lats[:]= lat[:]
lons[:]=lon[:]
#times_out[:] = tmp_date[:]
# close file
dataset.close()
os.chmod(merged_folder + 'scaling_factor_b2_1km.nc',0664)


