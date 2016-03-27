#!/usr/bin/python
import MySQLdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.optimize import curve_fit


def standard(data):
	mean=np.mean(data)
	std=np.std(data)
	return (data - mean)/std

def calculate_tot(data_vals,num_elements):
	#tot=[]
	#for x in range(0, num_elements-1):
	#	tot[x]=data_vals[x]
	return data_vals

db = MySQLdb.connect(host="172.17.0.12", # your host, usually localhost
		             user="chouser", # your username
					 passwd="chot3sti9", # your password
					 db="choslm") # name of the data base

# you must create a Cursor object. It will let
#  you execute all the queries you need
cur = db.cursor() 


time='2014-08-01 00%'
# Use all the SQL you like
cur.execute("select samplevalue from choslm.rn_qos_data_0011 where table_id='20' and sampletime like '2014-08-01 00%';")
#cur.execute("select * from choslm.s_qos_data;")

result_array=[]

# print all the first cell of all the rows
row = ''
num_entries=0
row=cur.fetchone()
while row:
	result_array.append(row[0])
	num_entries+=1
	row=cur.fetchone()

# Use all the SQL you like
cur.execute("select samplevalue from choslm.rn_qos_data_0003 where table_id='12' and sampletime like '2014-08-01 00%';")
#cur.execute("select * from choslm.s_qos_data;")

result_array_mem=[]

# print all the first cell of all the rows
row = ''
num_entries_mem=0
row=cur.fetchone()
while row:
	result_array_mem.append(row[0])
	num_entries_mem+=1
	row=cur.fetchone()

# Use all the SQL you like
cur.execute("select samplevalue from choslm.rn_qos_data_0016 where table_id='2' and sampletime like '2014-08-01 00%';")
#cur.execute("select * from choslm.s_qos_data;")

result_array_cpu=[]

# print all the first cell of all the rows
row = ''
num_entries_cpu=0
row=cur.fetchone()
while row:
	result_array_cpu.append(row[0])
	num_entries_mem+=1
	row=cur.fetchone()

# Use all the SQL you like
cur.execute("select samplevalue from choslm.rn_qos_data_0067 where table_id='62274' and sampletime like '2014-08-01 00%';")

result_array_if0=[]

# print all the first cell of all the rows
row = ''
num_entries_cpu=0
row=cur.fetchone()
while row:
	result_array_if0.append(row[0])
	num_entries_mem+=1
	row=cur.fetchone()

#print result_array
threshold=90000	
th_array=[threshold] * num_entries
tot=[]
tot=calculate_tot(result_array[:],num_entries)
xdata = np.array(np.arange(20))
ydata = np.array(result_array[:])

plt.figure(1)
plt.suptitle('Host 172.17.0.10')
plt.subplot(4,1,1)
plt.plot(result_array,'-b')
plt.subplot(4,1,2)
plt.plot(result_array_mem,'-g')
plt.subplot(4,1,3)
plt.plot(result_array_cpu,'-r')
plt.subplot(4,1,4)
plt.plot(result_array_if0,'-c')
#plt.plot(th_array,'-g')
#plt.title('Time to Threshold')
plt.show()
