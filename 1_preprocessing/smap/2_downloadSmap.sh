#!/bin/sh

#USAGE: downloadSmap.sh <file_name>
#<file_name> is the file with the list of files to download.
#This file can be downloaded using the following command:
# wget --no-check-certificate --user=<username> --password=<password> https://n5eil02u.ecs.nsidc.org/esir/<request_id>.txt
#Or using the script getListFileSmap.sh

targetFolder="/prj/hydrojules/data/soil_moisture/preprocessed/smap/smap_tif/"

read -p "Earthdata username: " USER
echo -e "\n"
read -s -p "Password for $USER: " PASS
echo -e "\n"

while read p; do
	echo "$p"
	wget --no-check-certificate --user=$USER --password=$PASS -P $targetFolder $p
	
	
done <$1
files=$targetFolder"*.tif"
chmod 664 $files

