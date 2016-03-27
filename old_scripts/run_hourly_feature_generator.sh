#!/usr/bin/bash

date=`echo $1 | sed 's/\///g' | sed 's/keyvalue_archive//g'`
dataset=$2

src_path=/raid1/akshah/Scanning_Behavior_Analysis/src/

hour_array=(00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23)

for hour in "${hour_array[@]}"; do
	result_file=results/"$date"_"$dataset"_feature_matrix_h"$hour"
	if [ -f $result_file ];
	then
		rm $result_file
	fi
	for i in $( ls -d $1/$hour/* ); do	
		cat $i | python $src_path/generate_features.py | python $src_path/reducer.py >> $result_file
	done
done

cat results/"$date"_"$dataset"_feature_matrix_h* > results/"$date"_"$dataset"_feature_matrix
