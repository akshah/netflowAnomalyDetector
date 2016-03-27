#!/usr/bin/bash

date=$1
minutes=$2

ls -d archive/$date/00/* | sh /raid1/akshah/Scanning_Behavior_Analysis/src/get_argus_keyval.sh &
ls -d archive/$date/01/* | sh /raid1/akshah/Scanning_Behavior_Analysis/src/get_argus_keyval.sh &
ls -d archive/$date/02/* | sh /raid1/akshah/Scanning_Behavior_Analysis/src/get_argus_keyval.sh &
ls -d archive/$date/03/* | sh /raid1/akshah/Scanning_Behavior_Analysis/src/get_argus_keyval.sh &
ls -d archive/$date/04/* | sh /raid1/akshah/Scanning_Behavior_Analysis/src/get_argus_keyval.sh &
sleep $minutes
ls -d archive/$date/05/* | sh /raid1/akshah/Scanning_Behavior_Analysis/src/get_argus_keyval.sh &
ls -d archive/$date/06/* | sh /raid1/akshah/Scanning_Behavior_Analysis/src/get_argus_keyval.sh &
ls -d archive/$date/07/* | sh /raid1/akshah/Scanning_Behavior_Analysis/src/get_argus_keyval.sh &
ls -d archive/$date/08/* | sh /raid1/akshah/Scanning_Behavior_Analysis/src/get_argus_keyval.sh &
ls -d archive/$date/09/* | sh /raid1/akshah/Scanning_Behavior_Analysis/src/get_argus_keyval.sh &
sleep $minutes
ls -d archive/$date/10/* | sh /raid1/akshah/Scanning_Behavior_Analysis/src/get_argus_keyval.sh &
ls -d archive/$date/11/* | sh /raid1/akshah/Scanning_Behavior_Analysis/src/get_argus_keyval.sh &
ls -d archive/$date/12/* | sh /raid1/akshah/Scanning_Behavior_Analysis/src/get_argus_keyval.sh &
sleep $minutes
ls -d archive/$date/13/* | sh /raid1/akshah/Scanning_Behavior_Analysis/src/get_argus_keyval.sh &
ls -d archive/$date/14/* | sh /raid1/akshah/Scanning_Behavior_Analysis/src/get_argus_keyval.sh &
ls -d archive/$date/15/* | sh /raid1/akshah/Scanning_Behavior_Analysis/src/get_argus_keyval.sh &
sleep $minutes
ls -d archive/$date/16/* | sh /raid1/akshah/Scanning_Behavior_Analysis/src/get_argus_keyval.sh &
ls -d archive/$date/17/* | sh /raid1/akshah/Scanning_Behavior_Analysis/src/get_argus_keyval.sh &
ls -d archive/$date/18/* | sh /raid1/akshah/Scanning_Behavior_Analysis/src/get_argus_keyval.sh &
ls -d archive/$date/19/* | sh /raid1/akshah/Scanning_Behavior_Analysis/src/get_argus_keyval.sh &
sleep $minutes
ls -d archive/$date/20/* | sh /raid1/akshah/Scanning_Behavior_Analysis/src/get_argus_keyval.sh &
ls -d archive/$date/21/* | sh /raid1/akshah/Scanning_Behavior_Analysis/src/get_argus_keyval.sh &
ls -d archive/$date/22/* | sh /raid1/akshah/Scanning_Behavior_Analysis/src/get_argus_keyval.sh &
ls -d archive/$date/23/* | sh /raid1/akshah/Scanning_Behavior_Analysis/src/get_argus_keyval.sh &
