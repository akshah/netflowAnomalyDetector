#!/usr/bin/bash
port=$2
for i in $( ls $1 | awk -F'.' '{print$1"."$2"."$3"*"}' | sort | uniq); do
  cat $1/$i | awk -v port="$port" -F',' '{if($6==port)print$4}'| sort | uniq | wc -l
done
