while read line
do
	grep $line csv_for_entropy | awk -F',' '{print$2}' | python /raid/akshah/Captures/argus_files/lander4/scripts/entropy.py
done
