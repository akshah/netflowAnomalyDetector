#!/usr/bin/bash

date=`echo $1 | sed 's/\///g' | sed 's/keyvalue_archive//g'`
date_with_slashes=`echo $1 | sed 's/keyvalue_archive\///g'`
dataset=$2

src_path=/raid1/akshah/Scanning_Behavior_Analysis/src/

hour_array=(00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23)

for hour in "${hour_array[@]}"; do
	result_file=results/"$date"_"$dataset"_anomalous_flows_h"$hour"
	input_file=results/"$date"_"$dataset"_feature_matrix_h"$hour"
	if [ -f $result_file ];
	then
		rm $result_file
	fi
	python ../../src/pca_on_feature_matrix.py $input_file $date_with_slashes | sort -n > $result_file	
done
