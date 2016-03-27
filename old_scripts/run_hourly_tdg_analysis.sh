#!/usr/bin/bash

date=`echo $1 | sed 's/\///g' | sed 's/keyvalue_archive//g'`
date_with_slashes=`echo $1 | sed 's/keyvalue_archive\///g'`
dataset=$2

src_path=/raid1/akshah/Scanning_Behavior_Analysis/src/

hour_array=(00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23)

for hour in "${hour_array[@]}"; do
	input_file=results/"$date"_"$dataset"_anomalous_flows_h"$hour"
	result_file=results/"$date"_"$dataset"_ranks_h"$hour"
	if [ -f $result_file ];
	then
		rm $result_file
	fi
	cat $input_file | python ../../src/feed_anomalous_flows.py | python ../../src/flows_to_graph.py | sort -nr | grep -vf nba.conf > $result_file

done

cat results/"$date"_"$dataset"_ranks_h* | python ../../src/generate_final_rank_list.py | sort -nr > results/"$date"_"$dataset"_ranks
