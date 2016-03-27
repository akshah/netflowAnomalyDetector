import matplotlib.mlab as mlab
import copy as cp
import matplotlib.pyplot as plt

import sys
import numpy as np

from sklearn.decomposition import PCA as sklearnPCA
from scipy.interpolate import spline
from scipy.interpolate import interp1d


def standard(data):
    return mlab.center_matrix(data)

def standardize(inX):	
	nrow=len(inX)
	stds = np.std(inX)
	mean = np.mean(inX)
	return (inX-mean)/stds

def scale(val, src, dst):
    """
    Scale the given value from the scale of src to the scale of dst.
    """
    return ((val - src[0]) / (src[1]-src[0])) * (dst[1]-dst[0]) + dst[0]


markers_on=[]
nrow_list={}
def mark_anomalous_instances(inX,data_with_time): 
    nrow=inX.shape[0]
    stds = inX.std(axis=0)
    mean = inX.mean(axis=0)
    for i in range(nrow):
        if(inX[i] >= (mean+3*stds) or inX[i] <= (mean-2*stds) ): 
            markers_on.append(i)
            anomalous_flow_time=int(data_with_time[i,0])
            if anomalous_flow_time not in nrow_list:
                nrow_list[anomalous_flow_time]=1;

def mark_anomalous_instances_array(inX,data_with_time):
    nrow=len(inX)
    stds = np.std(inX)
    mean = np.mean(inX)
    for i in range(nrow):
        if(inX[i] >= (mean+3*stds) or inX[i] <= (mean-2*stds) ):
            markers_on.append(i)	 
            anomalous_flow_time=int(data_with_time[i,0])
            if anomalous_flow_time not in nrow_list:
                nrow_list[anomalous_flow_time]=1;

def perform_pca(input_data,names):
	data_with_time=np.array(input_data)
	data = cp.deepcopy(data_with_time[:,1:])
	(nrow,ncol)=data.shape
	if nrow < ncol:
		print('Data not in right format. #Rows cannot be less than columns')
		exit(0)
	
	n_comp=ncol
	standard_data=standard(data)
	#mlab_pca = mlab.PCA((mlab.center_matrix(data)))
	sklearn_pca_all = sklearnPCA(n_components=n_comp)
	sklearn_transf_all = sklearn_pca_all.fit_transform(standard_data)

	#Calculate variance captured by each PC
	PC_Variances=[]

	for i in range(n_comp):
	    PC_Variances.append(np.var(sklearn_transf_all[:,i]))
		#print "Variance of PC ",i," is ",PC_Variances[i]

##########Variance Plot
	x=np.arange(n_comp)
	xnew = np.linspace(x.min(),x.max(),n_comp)
	smooth_variance = spline(x,PC_Variances,xnew)

	plt.figure(1)
	plt.plot(smooth_variance,label='PC Variance')

	plt.xticks(range(n_comp))

	plt.xlabel('PC')
	plt.ylabel('Variance')
	plt.ylim(-0.1,1.2)
	plt.legend()
	plt.title('Variance Captured by Principal Components')
	#plt.annotate('Most variance captured by first 2 PC', xy=(1,smooth_variance[1]),xytext=(2,smooth_variance[0]/2),fontsize=15,arrowprops=dict(facecolor='black', shrink=0.05),)
	#plt.draw()
	plt.savefig('variance_captured.png')
	#plt.show()
###############

	#####Calculate Anomaly Score
	#####Sum of PC3 to end

	anomaly_score=[]
	#Figure out possible max and min 
	pc_from=2
	pc_to=ncol
	
	pc_current=pc_from
	for pc in range(pc_from,pc_to):
	    mark_anomalous_instances(sklearn_transf_all[:,pc],data_with_time)	

	for i in range(nrow):
	    abs_array=np.absolute(sklearn_transf_all[i,pc_from:pc_to])#Excluding first two principal components
	    score=np.sum(abs_array)#sum
	    anomaly_score.append(score)	

	min_val=np.min(anomaly_score)
	max_val=np.max(anomaly_score)		
	mean_score=np.mean(anomaly_score)
	alpha=0.5
	mark_anomalous_instances_array(anomaly_score,data_with_time)	

	for v in markers_on:
	    #if anomaly_score[v] > mean_score:
	    #	print "here"
	    #else:
	    weighted_score=scale(anomaly_score[v],(min_val,max_val),((max_val+min_val)/2,max_val))
	    anomaly_score[v]=weighted_score

	min_val=np.min(anomaly_score)
	max_val=np.max(anomaly_score)		
	for v in range(nrow):
	    anomaly_score[v]=scale(anomaly_score[v],(min_val,max_val),(1,10))


	
######################Entropy Plots
	plt.figure(2,figsize=(20, 16))
	plt.suptitle('NEWS Flow Forensics: Netflow from a CA Customer',fontsize=14)
	total_subplot=6
	subnum=1

	plt.subplot(total_subplot,1,subnum)
	subnum+=1
	for c in range(1,5):
		plt.plot(standardize(data[:,c]),label=names[c])
	
	plt.ylabel('Entropy')
	#plt.ylim(1,15)#Find min and max here
	plt.legend(loc='lower right')
#	plt.tick_params(\
#	    axis='x',          # changes apply to the x-axis
#	     which='both',      # both major and minor ticks are affected
#	    bottom='off',      # ticks along the bottom edge are off
#	    top='off',         # ticks along the top edge are off
	  #  labelbottom='off'
#	  ) # labels along the bottom edge are off  


	######################################################COMMENTED
	plt.subplot(total_subplot,1,subnum)
	subnum+=1
	for c in range(5,7):
		plt.plot(standardize(data[:,c]),label=names[c])
	plt.legend(loc='lower right')


	plt.subplot(total_subplot,1,subnum)
	subnum+=1
	for c in range(7,8):
	    plt.plot(standardize(data[:,c]),label=names[c])
	plt.ylabel('Avg Packet Size')
	plt.legend(loc='lower right')
	#plt.tick_params(\
		#    axis='x',          # changes apply to the x-axis
	#    which='both',      # both major and minor ticks are affected
	#    bottom='off',      # ticks along the bottom edge are off
	#    top='off',         # ticks along the top edge are off
	#    labelbottom='off') # labels along the bottom edge are off  

	plt.subplot(total_subplot,1,subnum)
	subnum+=1
	for c in range(8,10):
	    plt.plot(standardize(data[:,c]),label=names[c])
	plt.legend(loc='lower right')
	####################################################END COMMENTED
	#plt.subplot(total_subplot,1,subnum)
	#subnum+=1
	#plt.plot(data[:,14],label=names[14])
	#plt.ylabel('Bytes')
	#plt.legend(loc='lower right')
	#plt.tick_params(\
		#    axis='x',          # changes apply to the x-axis
	#    which='both',      # both major and minor ticks are affected
	#    bottom='off',      # ticks along the bottom edge are off
	#    top='off',         # ticks along the top edge are off
	#    labelbottom='off') # labels along the bottom edge are off     

	plt.subplot(total_subplot,1,subnum)
	subnum+=1
	plt.plot(standardize(data[:,10]),label=names[10])
	plt.legend(loc='lower right')
	plt.ylabel('Numflows/min')
	#plt.tick_params(\
		#    axis='x',          # changes apply to the x-axis
	#    which='both',      # both major and minor ticks are affected
	#    bottom='off',      # ticks along the bottom edge are off
	#    top='off',         # ticks along the top edge are off
	#    labelbottom='off') # labels along the bottom edge are off   

	#plt.subplot(total_subplot,1,subnum)
	#subnum+=1
	#for c in range(16,17):
	#    plt.plot(standard_data[:,c],label=names[c])
	#plt.ylabel('StdDev Out Degree')
	#plt.legend(loc='lower right')
	#plt.tick_params(\
		#    axis='x',          # changes apply to the x-axis
	#    which='both',      # both major and minor ticks are affected
	#    bottom='off',      # ticks along the bottom edge are off
	#    top='off',         # ticks along the top edge are off
	#    labelbottom='off') # labels along the bottom edge are off  

	plt.subplot(total_subplot,1,subnum)
	subnum+=1
	plt.plot(anomaly_score,'-',label='Anomaly Score')
	for mark in markers_on:
		val=anomaly_score[mark]
		plt.plot(mark,val,'ro')
	plt.ylim(0.1,15)
	plt.ylabel('Score')
	#plt.xlabel('24 Hours')
	plt.legend(loc='lower right')
	#xticks=['','3am','6pm','9pm','']
	#plt.xticks(np.arange(1,nrow,nrow/24*6),xticks)
	plt.savefig('anomaly_score.png')
	#plt.show()

	#for flow_id in nrow_list:
	#    print '%d' % (flow_id)
	
	return nrow_list #List of anomalous time instances
