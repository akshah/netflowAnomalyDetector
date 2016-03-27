import numpy as np
import sys
import matplotlib.pyplot as plt	
from copy import deepcopy

def ewma(inmat):	
	alpha=0.87
	smooth_mat=deepcopy(inmat)
	for i in range(inmat.shape[0]):
		if(i!=0):
			smooth_mat[i]=(1-alpha)*inmat[i]+alpha*smooth_mat[i-1]	
	return smooth_mat

def smooth(x,window_len=5,window='hanning'):
	if x.ndim != 1:
		raise ValueError, "smooth only accepts 1 dimension arrays."
	if x.size < window_len:
		raise ValueError, "Input vector needs to be bigger than window size."
	if window_len<3:
		return x
	
	s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]

	if window == 'flat': #moving average
		w=np.ones(window_len,'d')
	else:
		w=eval('np.'+window+'(window_len)')
	
	y=np.convolve(w/w.sum(),s,mode='valid')
	return y

def standard(origX):
        means = origX.mean(axis=0)
        stds = origX.std(axis=0)
        return (origX - means) / stds

data_file=sys.argv[1]
nrow_list=[]
markers_on = []

def get_anomalous_secs(inX):
	###This method needs to be changed to work with PCA###
	stds = inX.std(axis=0)
	mean = inX.mean(axis=0)
	for i in range(nrow):
		if(inX[i] >= (mean+6*stds) or inX[i] <= (mean-4*stds) ):
		
			hour=(i)/60
			minute=((i)%60)
			
			anomalous_flow='%02d/argus.%02d.gz.data' % (hour,minute)
			nrow_list.append(anomalous_flow)
			markers_on.append(i)

names=['HProto','HSrcAddr','HDstAddr','HSport','HDport','AvgDur','AvgTcpRtt','TInPkts','TOutPkts','TPktsDrpd','TInBytes','TOutBytes','TBytesDrpd','AvgPktSize','NumFlows','StdOutDegree','StdRatioDegree']

data= np.loadtxt(data_file,delimiter=',',skiprows=0,dtype='float')
# Looking at data
(nrow,ncol) = data.shape


#Look for anomalous values in HDstAddr
get_anomalous_secs(data[:,1])
#get_anomalous_secs(data[:,2])
myfile = open(data_file+'_anomalies', "w")
for item in nrow_list:
	print>>myfile,item 
myfile.close()	

#plt.ion()
plt.figure(1,figsize=(8, 30))

dates=data_file.split('_')
date=dates[0]
title=date
plt.suptitle(title)

#plt.subplots_adjust(top=3)

#plt.clf()

total_subplots=ncol
subnum=1

plt.subplot(total_subplots,1,subnum)
dummy=[]
p1 = plt.plot(dummy,'c-')
p2 = plt.plot(dummy,'k-')
#for x in p1:
#	x.set_visible(False)
#for x in p2:
#	p2.set_visible(False)
l=plt.legend(('observed','smoothed'),'lower right')
l.set_visible(True)
plt.tick_params(\
   axis='both',          # changes apply to the both
   which='both',      # both major and minor ticks are affected
   bottom='off',      # ticks along the bottom edge are off
   top='off',         # ticks along the top edge are off
   labelbottom='off') # labels along the bottom edge are off
plt.axis('off')
subnum+=1




for c in range(ncol):
	#Incurrent Case they are not contributing much so no plot
	if names[c]=='HProto' or names[c]=='AvgTcpRtt' or names[c]=='TPktsDrpd' or names[c]=='TBytesDrpd':
		continue
	plt.subplot(total_subplots,1,subnum)
	mat=deepcopy(data[:,c])
	plt.plot(mat,'c-')
	plt.plot(ewma(mat),'k-')
	for ent in markers_on:
		val=mat[ent]
		plt.plot(ent,val,'ro')
	plt.ylabel(names[c])
	plt.tick_params(\
   	axis='x',          # changes apply to the x-axis
   	which='both',      # both major and minor ticks are affected
   	bottom='off',      # ticks along the bottom edge are off
   	top='off',         # ticks along the top edge are off
   	labelbottom='off') # labels along the bottom edge are off	
	subnum+=1	

'''
#HSrcAddr
plt.subplot(total_subplots,1,subnum)
mat=deepcopy(data[:,1])
plt.plot(mat,'c-')
plt.plot(ewma(mat),'k-')
for ent in markers_on:
	val=mat[ent]
	plt.plot(ent,val,'ro')
plt.ylabel(names[1])
plt.tick_params(\
   axis='x',          # changes apply to the x-axis
   which='both',      # both major and minor ticks are affected
   bottom='off',      # ticks along the bottom edge are off
   top='off',         # ticks along the top edge are off
   labelbottom='off') # labels along the bottom edge are off
#plt.legend(('Observed Entropy','Smoothed Entropy'),'best')
#xticks=['','1/08','1/09']
#plt.xticks(np.arange(1,nrow,86400),xticks)
subnum+=1

#HDstAddr
mat=deepcopy(data[:,2])
plt.subplot(total_subplots,1,subnum)
plt.plot(mat,'c-')
plt.plot(ewma(mat),'k-')
for ent in markers_on:
	val=mat[ent]
	plt.plot(ent,val,'ro')
plt.ylabel(names[2])
plt.tick_params(\
   axis='x',          # changes apply to the x-axis
   which='both',      # both major and minor ticks are affected
   bottom='off',      # ticks along the bottom edge are off
   top='off',         # ticks along the top edge are off
   labelbottom='off') # labels along the bottom edge are off
#plt.legend(('Observed Entropy','Smoothed Entropy'),'best')
#xticks=['','1/16','1/17']
#plt.xticks(np.arange(1,nrow,143),xticks)
subnum+=1


#HSport
mat=deepcopy(data[:,3])
plt.subplot(total_subplots,1,subnum)
plt.plot(mat,'c-')
plt.plot(ewma(mat),'k-')
for ent in markers_on:
	val=mat[ent]
	plt.plot(ent,val,'ro')
plt.ylabel(names[3])
plt.tick_params(\
   axis='x',          # changes apply to the x-axis
   which='both',      # both major and minor ticks are affected
   bottom='off',      # ticks along the bottom edge are off
   top='off',         # ticks along the top edge are off
   labelbottom='off') # labels along the bottom edge are off
#plt.legend(('Observed Entropy','Smoothed Entropy'),'best')
#xticks=['','1/16','1/17']
#plt.xticks(np.arange(1,nrow,143),xticks)
subnum+=1


#HDport
mat=deepcopy(data[:,4])
plt.subplot(total_subplots,1,subnum)
plt.plot(mat,'c-')
plt.plot(ewma(mat),'k-')
for ent in markers_on:
	val=mat[ent]
	plt.plot(ent,val,'ro')
plt.ylabel(names[4])
plt.tick_params(\
   axis='x',          # changes apply to the x-axis
   which='both',      # both major and minor ticks are affected
   bottom='off',      # ticks along the bottom edge are off
   top='off',         # ticks along the top edge are off
   labelbottom='off') # labels along the bottom edge are off
#plt.legend(('Observed Entropy','Smoothed Entropy'),'best')
#xticks=['','1/16','1/17']
#plt.xticks(np.arange(1,nrow,143),xticks)
subnum+=1


#NumFlows
plt.subplot(total_subplots,1,subnum)
mat=deepcopy(data[:,13])
plt.plot(mat,'c-')
plt.plot(ewma(mat),'k-')
for ent in markers_on:
	val=mat[ent]
	plt.plot(ent,val,'ro')
plt.ylabel(names[13])
plt.tick_params(\
   axis='x',          # changes apply to the x-axis
   which='both',      # both major and minor ticks are affected
   bottom='off',      # ticks along the bottom edge are off
   top='off',         # ticks along the top edge are off
   labelbottom='off') # labels along the bottom edge are off
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
subnum+=1

#AvgPktSize
plt.subplot(total_subplots,1,subnum)
mat_src_bytes=deepcopy(data[:,7])
mat_dst_bytes=deepcopy(data[:,8])
mat_src_pkts=deepcopy(data[:,10])
mat_dst_pkts=deepcopy(data[:,11])

mat_bytes=mat_src_bytes+mat_dst_bytes
mat_pkts=mat_src_pkts+mat_dst_pkts

avg_pkt_size=mat_pkts/mat_bytes

plt.plot(avg_pkt_size,'c-')
for ent in markers_on:
	val=avg_pkt_size[ent]
	plt.plot(ent,val,'ro')
plt.plot(ewma(avg_pkt_size),'k-')
#plt.yscale('log')
plt.ylabel('AvgPktSize')
'''

xticks=['','3am','6pm','9pm','']
plt.xticks(np.arange(1,nrow,nrow/24*6),xticks)

#plt.tight_layout()

fnames=data_file.split('/')
plt.savefig('pics/flow_features_'+fnames[1]+'.png')

