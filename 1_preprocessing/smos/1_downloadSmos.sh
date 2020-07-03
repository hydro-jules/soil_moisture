#!/bin/sh

targetFolder="/prj/hydrojules/data/soil_moisture/preprocessed/smos/smos_nc/"

#input_start='2020-5-1'
#input_end='2020-5-23'

read -p "BEC username: " USER
echo -e "\n"
read -s -p "Password for $USER: " PASS
echo -e "\n"

read -p "start date (format: YYYY-MM-DD): " input_start
echo -e "\n"
read -p "end date (format: YYYY-MM-DD): " input_end




dir="/data/LAND/SM/SMOS/EUROPE_and_MEDITERRANEAN/v5.0/L4/REPROCESSED/daily/ASC/"
#BEC_SM____SMOS__EUM_L4__A_20200415T033219_001km_1d_REP_v5.0.nc

#convert to dates
startdate=$(date -d $input_start +%Y%m%d)
enddate=$(date -d $input_end +%Y%m%d)

# Loop through dates
d=$startdate
echo $d


while [[ $d -le $enddate ]]

do
	dd=$(echo "$d" | cut -c7-8)

	# get day 
	#DD=$(date -d "$d" '+%d')
	#dd=$( printf '%02d' $DD )
	#echo $dd

	# get month 
	# NOTE: I had problems with august and september when using $MM
	# Had to replace with ${MM#0}. This is why: (from stackoverflow)
	# Why does this help? Well, a number literal starting with 
	# 0 but having no x at the 2nd place is interpreted as octal value.
	# Octal value only have the digits 0..7, 8 and 9 are unknown.
	# "${a#0}" strips one leading 0. The resulting value can be fed to 
	# printf then, which prints it appropriately, with 0 prefixed, in 4 digits.
	# https://stackoverflow.com/questions/8078167/
	# bizarre-issue-with-printf-in-bash-script09-and-08-are-invalid-numbers-07
	
	MM=$(date -d "$d" '+%m')
	mm=$( printf '%02d' ${MM#0} )
	
	# get year
	YY=$(date -d "$d" '+%Y')
	yy=$( printf '%02d' $YY )
	#echo $yy
	echo -e "\n"
	echo ${yy}-${mm}-${dd}
		
	#f="${dir}${yy}/BEC_SM____SMOS__EUM_L4__A_${yy}${mm}${dd}T030315_001km_1d_REP_v5.0.nc"
	f="${dir}${yy}/BEC_SM____SMOS__EUM_L4__A_${yy}${mm}${dd}T*_001km_1d_REP_v5.0.nc"
	com="$USER@becftp.icm.csic.es"

{
expect -c "
spawn sftp -P27500 $USER@becftp.icm.csic.es
expect \"password\"
send \"$PASS\r\"
expect \"sftp>\"
send \"get $f $targetFolder\r\"
expect \"sftp>\"
send \"quit\r\"
"
} || {
	echo "something went wrong for ${yy}-${mm}-${dd}"
}

	
	#f="https://n5eil01u.ecs.nsidc.org/DP4/SMAP/SPL3SMP_E.003/${yy}.${mm}.${dd}/SMAP_L3_SM_P_E_${yy}${mm}${dd}_R16510_001.h5"
	#echo $f
	#wget --no-check-certificate --user=$USER --password=$PASS $f

	d=$(date -d"$d + 1 day" +"%Y%m%d")
	myfile=$targetFolder"BEC_SM____SMOS__EUM_L4__A_${yy}${mm}${dd}T*_001km_1d_REP_v5.0.nc"
	chmod 664 $myfile
	
done
echo -e "\n"
echo -e "FINISHED!!"
echo -e "\n"

