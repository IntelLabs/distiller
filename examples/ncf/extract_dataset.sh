#!/bin/bash

echo "unzip ml-20m.zip"
if unzip -u ml-20m.zip
then
    echo "Start processing ml-20m/ratings.csv"
	python convert.py ml-20m/ratings.csv ml-20m --negatives 999
else
	echo "Problem unzipping ml-20.zip"
	echo "Please run 'download_data.sh && verify_datset.sh' first"
fi
