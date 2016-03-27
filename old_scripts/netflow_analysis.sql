use nsas;
show tables;
desc ahtflows;
select INET_NTOA(srcaddr) from nsas.ahtflows;
select count(distinct dstaddr) from nsas.ahtflows;	
select TIMESTAMP(FROM_UNIXTIME(min(timestamp))),TIMESTAMP(FROM_UNIXTIME(max(timestamp))) from nsas.ahtflows;
select * from nsas.ahtflows order by srcaddr;	
SELECT INET_NTOA('2130706433');
