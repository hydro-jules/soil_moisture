set -e
#
# Script to add metadata to the relevant netcdf files.
# Adds global attributes to the netcdf files.
#
# Make sure weights files and merging method file have been merged into one NetCDF
# before running this script.
# For instructions on how to do that, look at documentation, section 6.
#


# 1) add metadata to merged dataset 12.5km (TCA)
# ----------------------------------------------

merged1=/prj/hydrojules/data/soil_moisture/merged/merge_12.5km_tc_ref_smap.nc
echo $merged1

ncatted -a institution,global,o,c,"UK Centre for Ecology & Hydrology (UKCEH)" -h $merged1

ncatted -a title,global,o,c,"HydroJULES Soil moisture product - Merged data 12.5km resolution" -h $merged1

ncatted -a summary,global,o,c,"Based on Peng et al. (2021) study, a combined soil moisture product was generated to merge soil moisture data using triple collocation (TC) error
estimation and least-squares mergind scheme, from three sources: remote sensing data SMAP L3 and ASCAT, and land surface model JULES output for Great Britain. Details of the
methodology can be found in Peng et al. (2021).The dataset comprises: (i) a merged product based on TC analysis (12.5km version); (ii) a merged product based on TC analysis (1km
version); (iii) weights used in the merging and map of merging method used (for 12.5km version only); (iv) a merged produced based on a plain mean. This NetCDF file contains (i)." -h $merged1

ncatted -a cdm_data_type,global,o,c,"Grid" -h $merged1

ncatted -a spatial_resolution_distance,global,o,f,0.0125 -h $merged1

ncatted -a spatial_resolution_unit,global,o,c,"urn:ogc:def:uom:EPSG::9102" -h $merged1

ncatted -a standard_name_vocabulary,global,o,c,"CF Standard Name Table v70, http://cfconventions.org/standard-names.html" -h $merged1

ncatted -a standard_name_url_vocabulary,global,o,c,"NERC Vocabulary Server, https://vocab.nerc.ac.uk/standard_name/" -h $merged1

ncatted -a geospatial_lat_min,global,o,f,49.815 -h $merged1
ncatted -a geospatial_lat_max,global,o,f,59.19 -h $merged1
ncatted -a geospatial_lon_min,global,o,f,-7.46 -h $merged1
ncatted -a geospatial_lon_max,global,o,f,1.915 -h $merged1

ncatted -a time_coverage_start,global,o,c,"2015-04-01 00:00:00 UTC" -h $merged1
ncatted -a time_coverage_end,global,o,c,"2017-12-31 00:00:00 UTC" -h $merged1

ncatted -a time_coverage_resolution,global,o,c,"P1D" -h $merged1
ncatted -a time_coverage_duration,global,o,c,"P3Y" -h $merged1

ncatted -a keywords,global,o,c,"Soil moisture, SMAP, ASCAT, JULES, Triple collocation, merging" -h $merged1

ncatted -a history,global,o,c,"File created on $(date)" -h $merged1

ncatted -a references,global,o,c,"Peng, J., Tanguy, M., Robinson, E., Pinnington, E, Evans, J., Ellis, R., Cooper, E., Hannaford, J., Blyth, E., Dadson, S. (2021) Estimation and evaluation of high-resolution soil moisture from merged model and Earth observation data in the Great Britain, Remote Sensing of the Environment, 264, 112610, https://doi.org/10.1016/j.rse.2021.112610" -h $merged1

ncatted -a acknowledgment,global,o,c,"The work was supported by the UK Natural Environment Research Council (NE/S017380/1). We thank NASA for the generation and dissemination of SMAP soil moisture, the EUMETSAT Satellite Application Facility on Support to Operational Hydrology and Water Management (H SAF) for the dissemination of ASCAT soil moisture, and UKCEH for the JULES Dataset." -h $merged1

ncatted -a date_created,global,o,c,"$(date)" -h $merged1
ncatted -a creator_name,global,o,c,"Tanguy M." -h $merged1
ncatted -a creator_email,global,o,c,"enquiries@ceh.ac.uk" -h $merged1
ncatted -a creator_institution,global,o,c,"UK Centre for Ecology & Hydrology (UKCEH)" -h $merged1

ncatted -a contributor_name,global,o,c,"Peng, J., Tanguy, M., Robinson, E., Pinnington, E, Evans, J., Ellis, R., Cooper, E., Hannaford, J., Blyth, E., Dadson, S." -h $merged1

ncatted -a licence,global,o,c,"This dataset is available under the terms of the Open Government Licence https://eidc.ceh.ac.uk/licences/OGL/plain" -h $merged1
ncatted -a id,global,o,c,"TBC" -h $merged1
ncatted -a metadata_link,global,o,c," " -h $merged1
ncatted -a publisher_institution,global,o,c,"NERC Environmental Information Data Centre" -h $merged1
ncatted -a naming_authority,global,o,c,"DataCITE" -h $merged1
ncatted -a version,global,o,c,"v1" -h $merged1
ncatted -a Conventions,global,o,c,"CF-1.6" -h $merged1



# 2) add metadata to merged dataset 1km (TCA)
# ----------------------------------------------

merged1=/prj/hydrojules/data/soil_moisture/merged/merge_1km_tc_ref_smap.nc
echo $merged1

ncatted -a institution,global,o,c,"UK Centre for Ecology & Hydrology (UKCEH)" -h $merged1

ncatted -a title,global,o,c,"HydroJULES Soil moisture product - Merged data 1km resolution" -h $merged1

ncatted -a summary,global,o,c,"Based on Peng et al. (2021) study, a combined soil moisture product was generated to merge soil moisture data using triple collocation (TC) error
estimation and least-squares mergind scheme, from three sources: remote sensing data SMAP L3 and ASCAT, and land surface model JULES output for Great Britain. Details of the
methodology can be found in Peng et al. (2021).The dataset comprises: (i) a merged product based on TC analysis (12.5km version); (ii) a merged product based on TC analysis (1km
version); (iii) weights used in the merging and map of merging method used (for 12.5km version only); (iv) a merged produced based on a plain mean. This NetCDF file contains (ii)." -h $merged1


ncatted -a cdm_data_type,global,o,c,"Grid" -h $merged1

ncatted -a spatial_resolution_distance,global,o,f,0.001 -h $merged1

ncatted -a spatial_resolution_unit,global,o,c,"urn:ogc:def:uom:EPSG::9102" -h $merged1

ncatted -a standard_name_vocabulary,global,o,c,"CF Standard Name Table v70, http://cfconventions.org/standard-names.html" -h $merged1

ncatted -a standard_name_url_vocabulary,global,o,c,"NERC Vocabulary Server, https://vocab.nerc.ac.uk/standard_name/" -h $merged1

ncatted -a geospatial_lat_min,global,o,f,49.815 -h $merged1
ncatted -a geospatial_lat_max,global,o,f,59.19 -h $merged1
ncatted -a geospatial_lon_min,global,o,f,-7.46 -h $merged1
ncatted -a geospatial_lon_max,global,o,f,1.915 -h $merged1

ncatted -a time_coverage_start,global,o,c,"2015-04-01 00:00:00 UTC" -h $merged1
ncatted -a time_coverage_end,global,o,c,"2017-12-31 00:00:00 UTC" -h $merged1

ncatted -a time_coverage_resolution,global,o,c,"P1D" -h $merged1
ncatted -a time_coverage_duration,global,o,c,"P3Y" -h $merged1

ncatted -a keywords,global,o,c,"Soil moisture, SMAP, ASCAT, JULES, Triple collocation, merging" -h $merged1

ncatted -a history,global,o,c,"File created on $(date)" -h $merged1

ncatted -a references,global,o,c,"Peng, J., Tanguy, M., Robinson, E., Pinnington, E, Evans, J., Ellis, R., Cooper, E., Hannaford, J., Blyth, E., Dadson, S. (2021) Estimation and evaluation of high-resolution soil moisture from merged model and Earth observation data in the Great Britain, Remote Sensing of the Environment, 264, 112610, https://doi.org/10.1016/j.rse.2021.112610" -h $merged1

ncatted -a acknowledgment,global,o,c,"The work was supported by the UK Natural Environment Research Council (NE/S017380/1). We thank NASA for the generation and dissemination of SMAP soil moisture, the EUMETSAT Satellite Application Facility on Support to Operational Hydrology and Water Management (H SAF) for the dissemination of ASCAT soil moisture, and UKCEH for the JULES Dataset." -h $merged1

ncatted -a date_created,global,o,c,"$(date)" -h $merged1
ncatted -a creator_name,global,o,c,"Tanguy M." -h $merged1
ncatted -a creator_email,global,o,c,"enquiries@ceh.ac.uk" -h $merged1
ncatted -a creator_institution,global,o,c,"UK Centre for Ecology & Hydrology (UKCEH)" -h $merged1

ncatted -a contributor_name,global,o,c,"Peng, J., Tanguy, M., Robinson, E., Pinnington, E, Evans, J., Ellis, R., Cooper, E., Hannaford, J., Blyth, E., Dadson, S." -h $merged1

ncatted -a licence,global,o,c,"This dataset is available under the terms of the Open Government Licence https://eidc.ceh.ac.uk/licences/OGL/plain" -h $merged1
ncatted -a id,global,o,c,"TBC" -h $merged1
ncatted -a metadata_link,global,o,c," " -h $merged1
ncatted -a publisher_institution,global,o,c,"NERC Environmental Information Data Centre" -h $merged1
ncatted -a naming_authority,global,o,c,"DataCITE" -h $merged1
ncatted -a version,global,o,c,"v1" -h $merged1
ncatted -a Conventions,global,o,c,"CF-1.6" -h $merged1



# 3) add metadata to weights and method summary file (12.5km)
# -----------------------------------------------------------

merged1=/prj/hydrojules/data/soil_moisture/merged/weights/weights_and_method_summary_12.5km.nc
echo $merged1

ncatted -a institution,global,o,c,"UK Centre for Ecology & Hydrology (UKCEH)" -h $merged1

ncatted -a title,global,o,c,"HydroJULES Soil moisture product - Merged data 1km resolution" -h $merged1

ncatted -a summary,global,o,c,"Based on Peng et al. (2021) study, a combined soil moisture product was generated to merge soil moisture data using triple collocation (TC) error
estimation and least-squares mergind scheme, from three sources: remote sensing data SMAP L3 and ASCAT, and land surface model JULES output for Great Britain. Details of the
methodology can be found in Peng et al. (2021).The dataset comprises: (i) a merged product based on TC analysis (12.5km version); (ii) a merged product based on TC analysis (1km
version); (iii) weights used in the merging and map of merging method used (for 12.5km version only); (iv) a merged produced based on a plain mean. This NetCDF file contains (iii)." -h $merged1


ncatted -a cdm_data_type,global,o,c,"Grid" -h $merged1

ncatted -a spatial_resolution_distance,global,o,f,0.0125 -h $merged1

ncatted -a spatial_resolution_unit,global,o,c,"urn:ogc:def:uom:EPSG::9102" -h $merged1

ncatted -a standard_name_vocabulary,global,o,c,"CF Standard Name Table v70, http://cfconventions.org/standard-names.html" -h $merged1

ncatted -a standard_name_url_vocabulary,global,o,c,"NERC Vocabulary Server, https://vocab.nerc.ac.uk/standard_name/" -h $merged1

ncatted -a geospatial_lat_min,global,o,f,49.815 -h $merged1
ncatted -a geospatial_lat_max,global,o,f,59.19 -h $merged1
ncatted -a geospatial_lon_min,global,o,f,-7.46 -h $merged1
ncatted -a geospatial_lon_max,global,o,f,1.915 -h $merged1

ncatted -a keywords,global,o,c,"Soil moisture, SMAP, ASCAT, JULES, Triple collocation, merging" -h $merged1

ncatted -a history,global,o,c,"File created on $(date)" -h $merged1

ncatted -a references,global,o,c,"Peng, J., Tanguy, M., Robinson, E., Pinnington, E, Evans, J., Ellis, R., Cooper, E., Hannaford, J., Blyth, E., Dadson, S. (2021) Estimation and evaluation of high-resolution soil moisture from merged model and Earth observation data in the Great Britain, Remote Sensing of the Environment, 264, 112610, https://doi.org/10.1016/j.rse.2021.112610" -h $merged1

ncatted -a acknowledgment,global,o,c,"The work was supported by the UK Natural Environment Research Council (NE/S017380/1). We thank NASA for the generation and dissemination of SMAP soil moisture, the EUMETSAT Satellite Application Facility on Support to Operational Hydrology and Water Management (H SAF) for the dissemination of ASCAT soil moisture, and UKCEH for the JULES Dataset." -h $merged1

ncatted -a date_created,global,o,c,"$(date)" -h $merged1
ncatted -a creator_name,global,o,c,"Tanguy M." -h $merged1
ncatted -a creator_email,global,o,c,"enquiries@ceh.ac.uk" -h $merged1
ncatted -a creator_institution,global,o,c,"UK Centre for Ecology & Hydrology (UKCEH)" -h $merged1

ncatted -a contributor_name,global,o,c,"Peng, J., Tanguy, M., Robinson, E., Pinnington, E, Evans, J., Ellis, R., Cooper, E., Hannaford, J., Blyth, E., Dadson, S." -h $merged1

ncatted -a licence,global,o,c,"This dataset is available under the terms of the Open Government Licence https://eidc.ceh.ac.uk/licences/OGL/plain" -h $merged1
ncatted -a id,global,o,c,"TBC" -h $merged1
ncatted -a metadata_link,global,o,c," " -h $merged1
ncatted -a publisher_institution,global,o,c,"NERC Environmental Information Data Centre" -h $merged1
ncatted -a naming_authority,global,o,c,"DataCITE" -h $merged1
ncatted -a version,global,o,c,"v1" -h $merged1
ncatted -a Conventions,global,o,c,"CF-1.6" -h $merged1




# 4) add metadata to merged dataset 12.5km (mean)
# ----------------------------------------------

merged1=/prj/hydrojules/data/soil_moisture/merged/merge_12.5km_chess_smap_ascat_mean.nc
echo $merged1

ncatted -a institution,global,o,c,"UK Centre for Ecology & Hydrology (UKCEH)" -h $merged1

ncatted -a title,global,o,c,"HydroJULES Soil moisture product - Merged data 12.5km resolution" -h $merged1

ncatted -a summary,global,o,c,"Based on Peng et al. (2021) study, a combined soil moisture product was generated to merge soil moisture data using triple collocation (TC) error
estimation and least-squares mergind scheme, from three sources: remote sensing data SMAP L3 and ASCAT, and land surface model JULES output for Great Britain. Details of the
methodology can be found in Peng et al. (2021).The dataset comprises: (i) a merged product based on TC analysis (12.5km version); (ii) a merged product based on TC analysis (1km
version); (iii) weights used in the merging and map of merging method used (for 12.5km version only); (iv) a merged produced based on a plain mean. This NetCDF file contains (iv)." -h $merged1


ncatted -a cdm_data_type,global,o,c,"Grid" -h $merged1

ncatted -a spatial_resolution_distance,global,o,f,0.0125 -h $merged1

ncatted -a spatial_resolution_unit,global,o,c,"urn:ogc:def:uom:EPSG::9102" -h $merged1

ncatted -a standard_name_vocabulary,global,o,c,"CF Standard Name Table v70, http://cfconventions.org/standard-names.html" -h $merged1

ncatted -a standard_name_url_vocabulary,global,o,c,"NERC Vocabulary Server, https://vocab.nerc.ac.uk/standard_name/" -h $merged1

ncatted -a geospatial_lat_min,global,o,f,49.815 -h $merged1
ncatted -a geospatial_lat_max,global,o,f,59.19 -h $merged1
ncatted -a geospatial_lon_min,global,o,f,-7.46 -h $merged1
ncatted -a geospatial_lon_max,global,o,f,1.915 -h $merged1

ncatted -a time_coverage_start,global,o,c,"2015-04-01 00:00:00 UTC" -h $merged1
ncatted -a time_coverage_end,global,o,c,"2017-12-31 00:00:00 UTC" -h $merged1

ncatted -a time_coverage_resolution,global,o,c,"P1D" -h $merged1
ncatted -a time_coverage_duration,global,o,c,"P3Y" -h $merged1

ncatted -a keywords,global,o,c,"Soil moisture, SMAP, ASCAT, JULES, Triple collocation, merging" -h $merged1

ncatted -a history,global,o,c,"File created on $(date)" -h $merged1

ncatted -a references,global,o,c,"Peng, J., Tanguy, M., Robinson, E., Pinnington, E, Evans, J., Ellis, R., Cooper, E., Hannaford, J., Blyth, E., Dadson, S. (2021) Estimation and evaluation of high-resolution soil moisture from merged model and Earth observation data in the Great Britain, Remote Sensing of the Environment, 264, 112610, https://doi.org/10.1016/j.rse.2021.112610" -h $merged1

ncatted -a acknowledgment,global,o,c,"The work was supported by the UK Natural Environment Research Council (NE/S017380/1). We thank NASA for the generation and dissemination of SMAP soil moisture, the EUMETSAT Satellite Application Facility on Support to Operational Hydrology and Water Management (H SAF) for the dissemination of ASCAT soil moisture, and UKCEH for the JULES Dataset." -h $merged1

ncatted -a date_created,global,o,c,"$(date)" -h $merged1
ncatted -a creator_name,global,o,c,"Tanguy M." -h $merged1
ncatted -a creator_email,global,o,c,"enquiries@ceh.ac.uk" -h $merged1
ncatted -a creator_institution,global,o,c,"UK Centre for Ecology & Hydrology (UKCEH)" -h $merged1

ncatted -a contributor_name,global,o,c,"Peng, J., Tanguy, M., Robinson, E., Pinnington, E, Evans, J., Ellis, R., Cooper, E., Hannaford, J., Blyth, E., Dadson, S." -h $merged1

ncatted -a licence,global,o,c,"This dataset is available under the terms of the Open Government Licence https://eidc.ceh.ac.uk/licences/OGL/plain" -h $merged1
ncatted -a id,global,o,c,"TBC" -h $merged1
ncatted -a metadata_link,global,o,c," " -h $merged1
ncatted -a publisher_institution,global,o,c,"NERC Environmental Information Data Centre" -h $merged1
ncatted -a naming_authority,global,o,c,"DataCITE" -h $merged1
ncatted -a version,global,o,c,"v1" -h $merged1
ncatted -a Conventions,global,o,c,"CF-1.6" -h $merged1

