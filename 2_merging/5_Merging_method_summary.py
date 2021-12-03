## merge data based on Triple collocation error statistics

# to derive mean smap value, I used this command:
# cdo -b F64 -timmean -setmissval,nan smap_12.5km.nc smap_12.5km_mean.nc

# to derive mean chess value, I used this command:
# cdo -b F64 -timmean -setmissval,nan chess_12.5km.nc chess_12.5km_mean.nc


# Merging method (X = SMAP, Y = CHESS, Z = ASCAT)
# 0 = TCA weighted mean(X, Y, Z)
# 1 = X
# 2 = Z
# 3 = Y
# 4 = Arithmetic mean (X, Y)
# 5 = Arithmetic mean (X, Z)
# 6 = Arithmetic mean (Y, Z)
# 7 = Disregard



#-----NOT IN USE--------
# Merging method:
# 0 = TCA, 1 = CHESS only, 2 = SMOS(ASCAT)/CHESS, 3 = SMAP/CHESS, 4 = SMOS/SMAP
#-----------------------

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
smos_file = smos_folder + 'ascat_VWC_linear_T_BULK_DEN_porosity_Toth_et_al_12.5km.nc'
smap_file = smap_folder + 'smap_12.5km.nc'
chess_file = chess_folder + 'chess_12.5km.nc'
b2_file = merged_folder + 'weights/b2_interpolated_12.5km.nc'
# to derive mean smap value, I used this command:
# cdo -b F64 -timmean -setmissval,nan smap_12.5km.nc smap_12.5km_mean.nc
# And then I interpolated using python code 0a3_interpolate_smap_mean.py
smap_mean_file = smap_folder + 'smap_12.5km_mean_interpolated.nc'
# to derive mean chess value, I used this command:
# cdo -b F64 -timmean -setmissval,nan chess_12.5km.nc chess_12.5km_mean.nc
chess_mean_file = chess_folder + 'chess_12.5km_mean.nc'

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
chess = chess/100.0

dataset_chess_mean = nc.Dataset(chess_mean_file,'r')
chess_mean_whole = np.squeeze(np.array(dataset_chess_mean.variables['sm']))
chess_mean_whole = chess_mean_whole/100.0

dataset_smap_mean = nc.Dataset(smap_mean_file,'r')
smap_mean_whole = np.squeeze(np.array(dataset_smap_mean.variables['sm']))

dataset_b2 = nc.Dataset(b2_file,'r')
b2_whole = np.squeeze(np.array(dataset_b2.variables['b2']))

#masked out the values less than 0 and larger than 1
smos = np.ma.masked_where((smos<0)| (smos>1), smos)
smap = np.ma.masked_where((smap<0)| (smap>1),smap)
chess = np.ma.masked_where((chess<0)|(chess>1), chess)

#smap_mean_whole = np.ma.masked_where((smap_mean_whole<0)| (smap_mean_whole>1),smap_mean_whole)
#chess_mean_whole = np.ma.masked_where((chess_mean_whole<0)|(chess_mean_whole>1), chess_mean_whole)
#b2_whole = np.ma.masked_where(b2_whole<0, b2_whole)

#to make the three datasets have same sample size for each pixel
smos_tc = np.ma.masked_where((np.isnan(smap))|(np.isnan(chess))|(np.isnan(smos)), smos)
smap_tc = np.ma.masked_where(np.isnan(smos_tc), smap)
chess_tc = np.ma.masked_where(np.isnan(smos_tc), chess)


smos_merge = np.ma.filled(smos_tc.astype(float), np.nan)
smap_merge = np.ma.filled(smap_tc.astype(float), np.nan)
chess_merge = np.ma.filled(chess_tc.astype(float), np.nan)

# make the duo smos/chess (or ascat/chess)
smos_duo_1 = np.ma.masked_where(np.isnan(chess)|(np.isnan(smos)), smos)
chess_duo_1 = np.ma.masked_where(np.isnan(smos_duo_1), chess)

smos_merge_duo_1 = np.ma.filled(smos_duo_1.astype(float), np.nan)
chess_merge_duo_1 = np.ma.filled(chess_duo_1.astype(float), np.nan)

# make the duo smap/chess
smap_duo_2 = np.ma.masked_where(np.isnan(chess)|(np.isnan(smap)), smap)
chess_duo_2 = np.ma.masked_where(np.isnan(smap_duo_2), chess)

smap_merge_duo_2 = np.ma.filled(smap_duo_2.astype(float), np.nan)
chess_merge_duo_2 = np.ma.filled(chess_duo_2.astype(float), np.nan)

# make the duo smap/smos
smap_duo_3 = np.ma.masked_where(np.isnan(smos)|(np.isnan(smap)), smap)
smos_duo_3 = np.ma.masked_where(np.isnan(smap_duo_3), smos)

smap_merge_duo_3 = np.ma.filled(smap_duo_3.astype(float), np.nan)
smos_merge_duo_3 = np.ma.filled(smos_duo_3.astype(float), np.nan)

# make an interpolated version of scaled chess
chess_scaled_interpolated = b2_whole*(chess-chess_mean_whole)+smap_mean_whole
chess_scaled_interpolated = np.where(chess_scaled_interpolated<0,0,chess_scaled_interpolated)
chess_scaled_interpolated = np.where(chess_scaled_interpolated>1,1,chess_scaled_interpolated)


########create merged file###########################
if merge_method_TC==True:
    dataset = nc.Dataset(merged_folder + 'merging_method_summary_12.5km.nc','w',format='NETCDF4')


# create dimensions
dataset.createDimension('lat',len(lat[:]))
dataset.createDimension('lon',len(lon[:]))

# create variables
lats = dataset.createVariable('lat',np.float32,('lat',))
lons = dataset.createVariable('lon',np.float32,('lon',))


#sm_cal = dataset.createVariable('sm',np.float32,('time','lat','lon',),fill_value=9.96921e+36,zlib=True)
m_meth = dataset.createVariable('merging_method',np.float32,('lat','lon',),fill_value=-999.,zlib=True)

# create Attributes
# Merging method (X = SMAP, Y = CHESS, Z = ASCAT)
# 0 = TCA weighted mean(X, Y, Z)
# 1 = X
# 2 = Z
# 3 = Y
# 4 = Arithmetic mean (X, Y)
# 5 = Arithmetic mean (X, Z)
# 6 = Arithmetic mean (Y, Z)
# 7 = Disregard
m_meth.units = '-'
m_meth.long_name = 'Merging Method'
m_meth.comment = '0 = TCA, 1 = SMAP only, 2 = CHESS only, 3 = ASCAT only, \
4 = Arithmetic mean (SMAP, CHESS), 5 = Arithmetic mean (SMAP, ASCAT),\
6 = Arithmetic mean (CHESS, ASCAT)'

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
            if len(chess[:,i,j].compressed())>0:
                if len(chess_tc[:,i,j].compressed())>0:
                    # pdb.set_trace()



                    slope_ps, intercept_ps, r_value_ps, p_value_ps, std_err_ps = stats.linregress(smap_tc[:,i,j].compressed(),smos_tc[:,i,j].compressed())
                    slope_pc, intercept_pc, r_value_pc, p_value_pc, std_err_pc = stats.linregress(smap_tc[:,i,j].compressed(),chess_tc[:,i,j].compressed())
                    slope_sc, intercept_sc, r_value_sc, p_value_sc, std_err_sc = stats.linregress(chess_tc[:,i,j].compressed(),smos_tc[:,i,j].compressed())


                    # Merging method (X = SMAP, Y = CHESS, Z = ASCAT)
                    # 0 = TCA weighted mean(X, Y, Z)
                    # 1 = X
                    # 2 = Z
                    # 3 = Y
                    # 4 = Arithmetic mean (X, Y)
                    # 5 = Arithmetic mean (X, Z)
                    # 6 = Arithmetic mean (Y, Z)
                    # 7 = Disregard
                    if (p_value_ps>=0.05) & (p_value_pc>=0.05)& (p_value_sc>=0.05):
                        m_meth[i,j] = 2
                    elif (p_value_ps<0.05) & (p_value_pc<0.05)& (p_value_sc>=0.05):
                        m_meth[i,j] =  1
                    elif (p_value_ps<0.05) & (p_value_pc>=0.05)& (p_value_sc<0.05):
                        m_meth[i,j] =  3
                    elif (p_value_ps>=0.05) & (p_value_pc<0.05)& (p_value_sc<0.05):
                        m_meth[i,j] =  2
                    elif (p_value_ps>=0.05) & (p_value_pc<0.05)& (p_value_sc>=0.05):
                        m_meth[i,j] =  4
                    elif (p_value_ps<0.05) & (p_value_pc>=0.05)& (p_value_sc>=0.05):
                        m_meth[i,j] =  5
                    elif (p_value_ps>=0.05) & (p_value_pc>=0.05)& (p_value_sc<0.05):
                        m_meth[i,j] =  6
                    elif (p_value_ps<0.05) & (p_value_pc<0.05)& (p_value_sc<0.05):
                        m_meth[i,j] = 0
                    else:
                        m_meth[i,j] = 2



##                    if (p_value_ps>=0.05) & (p_value_pc>=0.05)& (p_value_sc>=0.05):
##
##                        m_meth[i,j] = 1
##                        #print 'case1'
##                    elif (p_value_ps<0.05) & (p_value_pc>=0.05)& (p_value_sc>=0.05):
##                     # calculate duo smos/smap
##                        m_meth[i,j] = 4
##                        #print 'case4'
##                    elif (p_value_ps>=0.05) & (p_value_pc<0.05)& (p_value_sc>=0.05):
##                        m_meth[i,j] = 3
##                        #print 'case3'
##                    elif (p_value_ps>=0.05) & (p_value_pc>=0.05)& (p_value_sc<0.05):
##                        m_meth[i,j] = 2
##                        #print 'case2'
##                # gap fill with scaled chess whatever is left with Nan
##                    elif (p_value_ps<0.05) & (p_value_pc<0.05)& (p_value_sc<0.05):
##                        m_meth[i,j] = 0
##                    else:
##                        m_meth[i,j] = 1
                else:
                    m_meth[i,j] = 2


            else:
                # sm_cal[:,i,j] = np.nanmean([chess[:,i,j],smap[:,i,j] ,smos[:,i,j] ],0)
                #sm_cal[:,i,j] = np.where(np.isnan(sm_cal[:,i,j]),chess_full_scaled,sm_cal[:,i,j])

                m_meth[i,j] = 2

############################# merge soil moisture based on average them ##########################################################



sm_interm = np.array(m_meth)
sm_interm= np.ma.masked_where((sm_interm<0)| (sm_interm>4), sm_interm)
#sm_interm=np.ma.filled(sm_interm.astype(float), np.nan)
#sm_interm=np.where(np.isnan(sm_interm),chess_scaled_interpolated,sm_interm)
m_mask = np.ma.getmask(chess[0,:,:])
sm_interm= np.ma.masked_array(sm_interm,mask=m_mask)
m_meth[:]=sm_interm
lats[:]= lat[:]
lons[:]=lon[:]
#times_out[:] = tmp_date[:]
# close file
dataset.close()
os.chmod(merged_folder + 'merging_method_summary_12.5km.nc',0664)


