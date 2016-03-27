
cat keyvalue_archive/2014/01/15/$1/* | grep  129.82.138.50 | awk -F',' '{print$5}' | sort | uniq | grep -vf results/already_appreared_ports > results/ports.15$1 && cat results/ports.15$1 >> results/already_appreared_ports
wc=` tail -n1 results/20140115_netsec_uniq_ports_129.82.138.50|tr -d '\n'`
wc2=`cat results/ports.15$1 | wc -l|tr -d '\n'`
echo "$wc $wc2" | awk '{print $1+$2}' >> results/20140115_netsec_uniq_ports_129.82.138.50
