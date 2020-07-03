#!/bin/sh

# usage: getListFileSmap.sh <request_id>
# example: getListFileSmap.sh 5000000696489
# The request ID can be found in the email you receive from EarthData

read -p "Earthdata username: " USER
echo -e "\n"
read -s -p "Password for $USER: " PASS
echo -e "\n"

f="https://n5eil02u.ecs.nsidc.org/esir/$1.txt"
echo $f

wget --no-check-certificate --user=$USER --password=$PASS $f
chmod 664 "$1.txt"
