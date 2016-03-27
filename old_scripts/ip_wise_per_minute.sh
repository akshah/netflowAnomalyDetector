#!/usr/bin/bash
ip=$2
for i in $( ls $1 | awk -F'.' '{print$1"."$2"."$3"*"}' | sort | uniq); do
  cat $1/$i | grep "$2" | awk -F',' '{print$5}'| sort | uniq | wc -l
done
