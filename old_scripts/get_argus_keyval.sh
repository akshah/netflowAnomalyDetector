while read line
do	
	#Key is the frist timestamp in the file
	key=`ra -nnr $line -u -s stime proto saddr daddr sport dport dur tcprtt spkts dpkts loss sbytes dbytes load -c',' -L-1 -N1 - udp| awk -F'.' '{print$1}'`
	path=${line%argus*}
	mkdir -p keyvalue_$path
	racluster -nnr $line -u -s stime proto saddr daddr sport dport dur tcprtt spkts dpkts loss sbytes dbytes load -c',' -L-1 - udp | awk -v key=$key '{printf(key"\t"$_"\n")}' > keyvalue_$line.data
done
