#!/usr/bin/env python
import sys
import time
import random
import numpy as np
import networkx as nx

import matplotlib.mlab as mlab
import copy as cp
from matplotlib import pyplot as plt

import os
import networkx as nx

from sklearn.decomposition import PCA as sklearnPCA
from scipy.interpolate import spline
from scipy.interpolate import interp1d


#Universal Windows to threshold on
Wup=4 ## Wup x StdDev
Wdn=3

#Hold anomalous times and their indexes
markers_on=[]
nrow_list={}

#Definition of Calculation functions

def get_entropy_old(*args):
    xy = zip(*args)
    # probs
    proba = [ float(xy.count(c)) / len(xy) for c in dict.fromkeys(list(xy)) ]
    entropy= - sum([ p * np.log2(p) for p in proba ])
    return entropy

def get_entropy(*args):
    xy = list(zip(*args))
    # probs
    proba = [ float(xy.count(c)) / len(xy) for c in dict.fromkeys(xy) ]
    entropy= - sum([ p * np.log2(p) for p in proba ])
    return entropy


def get_avg(lt):
	return np.mean(lt)

def get_sum(lt):
	return np.sum(lt)


def get_graph_features(SrcAddrL,DstAddrL,NumFlows):
	#Initialize DiGraph
	G=nx.DiGraph()
	for index in range(0,NumFlows):
		G.add_edge(SrcAddrL[index],DstAddrL[index])

	out_degree_dict=G.out_degree()
	in_degree_dict=G.in_degree()
	out_degrees=[]
	ratio_degree_dict=G.in_degree()
	ratio_degree=[]
	in_degrees=[]
	for node in out_degree_dict:
		out_degrees.append(out_degree_dict[node])
		in_degrees.append(in_degree_dict[node])
		ratio_degree_dict[node]=(out_degree_dict[node]+0.001)/(in_degree_dict[node]+0.001)
		ratio_degree.append((out_degree_dict[node]+0.001)/(in_degree_dict[node]+0.001))
	
	
	std_out_degree=np.std(out_degrees)
	std_ratio_degree=np.std(ratio_degree)

	return std_out_degree, std_ratio_degree

def get_features(flow_data_row):

	
	SrcAddrL,DstAddrL,TimestampL,InPktsL,InBytesL,DstPortL,SrcPortL,OutIfL,InIfL,RouterL,TCPL,ProtocolL,NextHopL,NumFlowsL=flow_data_row

	#Feature Values
	HProto = None
	HSrcAddr = None
	HDstAddr = None
	HSport = None
	HDport = None
	TInPkts = None
	TInBytes = None
	AvgPktSize= 0
	StdOutDegree = 0
	StdRatioDegree=0
	NumFlows=NumFlowsL[0]
	Timestamp=TimestampL[0]#Begining of the block

	#Start Calculation of features
	#print('Calculating Entropy for Protocol')
	HProto=get_entropy(ProtocolL)
	#print('Calculating Entropy for Src Addr')
	HSrcAddr=get_entropy(SrcAddrL)
	#print('Calculating Entropy for Dst Addr')
	HDstAddr=get_entropy(DstAddrL)
	#print('Calculating Entropy for Src Port')
	HSport=get_entropy(SrcPortL)
	#print('Calculating Entropy for Dst Port')
	HDport=get_entropy(DstPortL)
	#print('Calculating Volume Metrics')
	TInPkts = get_sum(InPktsL)
	TInBytes = get_sum(InBytesL)
	AvgPktSize = (TInBytes/TInPkts)
	#NumFlows is same as passed
	#print('Calculating graph Metrics')
	(StdOutDegree,StdRatioDegree) = get_graph_features(SrcAddrL,DstAddrL,NumFlows)

	return Timestamp,HProto,HSrcAddr,HDstAddr,HSport,HDport,TInPkts,TInBytes,AvgPktSize,StdOutDegree,StdRatioDegree,NumFlows


def standard(data):
    return mlab.center_matrix(data)

def standardize(inX):	
    nrow=len(inX)
    stds = np.std(inX)
    mean = np.mean(inX)
    if(stds == 0):
        stds = 0.0001 #Hack in case the values are all same 
    return (inX-mean)/stds

def scale(val, src, dst):
    """
    Scale the given value from the scale of src to the scale of dst.
    """
    return ((val - src[0]) / (src[1]-src[0])) * (dst[1]-dst[0]) + dst[0]


def mark_anomalous_instances(inX,data_with_time): 
    nrow=inX.shape[0]
    stds = inX.std(axis=0)
    mean = inX.mean(axis=0)
    for i in range(nrow):
        if(inX[i] >= (mean+Wup*stds) or inX[i] <= (mean-Wdn*stds) ):
            if i not in markers_on:
                markers_on.append(i)
            anomalous_flow_time=int(data_with_time[i,0])
            if anomalous_flow_time not in nrow_list:
                nrow_list[anomalous_flow_time]=1;


def mark_anomalous_instances_array(inX,data_with_time):
    nrow=len(inX)
    stds = np.std(inX)
    mean = np.mean(inX)
    for i in range(nrow):
        if(inX[i] >= (mean+Wup*stds) or inX[i] <= (mean-Wdn*stds) ):
            if i not in markers_on:
                markers_on.append(i)	 
            anomalous_flow_time=int(data_with_time[i,0])
            if anomalous_flow_time not in nrow_list:
                nrow_list[anomalous_flow_time]=1;


def perform_pca(input_data,names,time_filename):

	#Hold anomalous times and their indexes
	del markers_on[:]
	nrow_list.clear()

	data_with_time=np.array(input_data)
	data = cp.deepcopy(data_with_time[:,1:])
	(nrow,ncol)=data.shape
	if nrow < ncol:
		print('Data not in right format. #Rows cannot be less than #Columns')
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

	#plt.figure(1)
	#plt.plot(smooth_variance,label='PC Variance')

	#plt.xticks(range(n_comp))

	#plt.xlabel('PC')
	#plt.ylabel('Variance')
	#plt.ylim(-0.1,1.2)
	#plt.legend()
	#plt.title('Variance Captured by Principal Components')
	#plt.annotate('Most variance captured by first 2 PC', xy=(1,smooth_variance[1]),xytext=(2,smooth_variance[0]/2),fontsize=15,arrowprops=dict(facecolor='black', shrink=0.05),)
	#plt.draw()
	#pic_name="variance_captured_"+str(time_filename)+"png"
	#plt.savefig(pic_name)
	#plt.show()
###############

	#####Calculate Anomaly Score
	#####Sum of PC3 to end


	anomaly_score=[]
	#Figure out possible max and min 
	pc_from=1
	pc_to=ncol
	
	pc_current=pc_from
	for pc in range(pc_from,pc_to):
		mark_anomalous_instances(sklearn_transf_all[:,pc],data_with_time)	
		#print(markers_on)

	for i in range(nrow):
	    abs_array=np.absolute(sklearn_transf_all[i,pc_from:pc_to])#Excluding first few principal components
	    score=np.sum(abs_array)#sum
	    anomaly_score.append(score)	

	min_val=np.min(anomaly_score)
	max_val=np.max(anomaly_score)		
	mean_score=np.mean(anomaly_score)
	alpha=0.5
	mark_anomalous_instances_array(anomaly_score,data_with_time)	

	
	for v in markers_on:
		#print(len(anomaly_score),v)
        #if anomaly_score[v] > mean_score:
	    #	print "here"
	    #else:
		weighted_score=scale(anomaly_score[v],(min_val,max_val),((max_val+min_val)/1.2,max_val))
		anomaly_score[v]=weighted_score

	min_val=np.min(anomaly_score)
	max_val=np.max(anomaly_score)
	
	for v in range(nrow):
		if v not in markers_on:
			anomaly_score[v]=scale(anomaly_score[v],(min_val,max_val),(min_val,(max_val+min_val)/1.2))
		anomaly_score[v]=scale(anomaly_score[v],(min_val,max_val),(1,100))
	

    ######################Plots Start
    
    ######################Plots End
	#fname1="anomaly_score_"+str(time_filename)+"png"
	#plt.savefig(fname1)
	return nrow_list, anomaly_score,markers_on #List of anomalous time instances


def extract_suspicious_nodes(G):	

	out_degree_dict=G.out_degree()
	in_degree_dict=G.in_degree()
	#pageranks=nx.pagerank_numpy(G.reverse())#Attackers will get higher rank
	out_degrees=[]
	#pg_rank_array=[]
	ratio_degree_dict=G.in_degree()
	ratio_degree=[]
	in_degrees=[]
	connected_suspicious_nodes=[]
	suspicious={}
	for node in out_degree_dict:  
		out_degrees.append(out_degree_dict[node])	
		in_degrees.append(in_degree_dict[node])
		#pg_rank_array.append(pageranks[node])
		
		rt_cal=0
		if in_degree_dict[node] > 0:
			rt_cal=(out_degree_dict[node])/(in_degree_dict[node])
		else:
			rt_cal=(out_degree_dict[node])
		ratio_degree_dict[node]=(rt_cal)
		ratio_degree.append(rt_cal)
	

	std_out_degree=np.std(out_degrees)
	std_ratio_degree=np.std(ratio_degree)
	#std_pg_rank=np.std(pg_rank_array)
	mean_out_degree=np.mean(out_degrees)
	mean_ratio_degree=np.mean(ratio_degree)
	#mean_pg_rank=np.mean(pg_rank_array)

	for node in out_degree_dict:  
		#print pageranks[node]," ",node
		if out_degree_dict[node] > (mean_out_degree+Wup*std_out_degree) or ratio_degree_dict[node]>(mean_ratio_degree+Wdn*std_ratio_degree):

			suspicious[node]=out_degree_dict[node]
			#Get all nodes connected to this suspicious node
			#connected_suspicious_nodes=nx.node_connected_component(G.to_undirected(),node)
			#connected_suspicious_nodes.append(node)#Add yourself
			'''			
			connected_suspicious_nodes=G.predecessors(node)#If you are suspected, so are your predecessors
		
			for connected_nodename in connected_suspicious_nodes:
				if connected_nodename not in suspicious or pageranks[node] > (mean_pg_rank+2*std_pg_rank):
					#suspicious[connected_nodename]=pageranks[connected_nodename]
					suspicious[connected_nodename]=out_degree_dict[connected_nodename]
			'''	
	return suspicious			

def process_suspicious_nodes(suspicious_nodes):
	for node in suspicious_nodes:
		print(suspicious_nodes[node],"\t",node)


def get_tdg(SrcAddrL,DstAddrL,TimestampL,InPktsL,InBytesL,DstPortL,SrcPortL,OutIfL,InIfL,RouterL,TCPL,ProtocolL,NextHopL,NumFlowsL):
	#Initialize graph
	NumFlows=NumFlowsL[0]
	G=nx.DiGraph()
	for index in range(0,NumFlows):
		SrcAddr=str(SrcAddrL[index])
		Sport=str(SrcPortL[index])
		DstAddr=str(DstAddrL[index])
		Dport=str(DstPortL[index])

		#To capture many ports to many ports
		node_src=SrcAddr+":"+Sport
		node_dst=DstAddr+":"+Dport
		
		SrcAddr_octates=SrcAddr.split(".")
		DstAddr_octates=DstAddr.split(".")

		if len(SrcAddr_octates)<4 or len(DstAddr_octates)<4:
			continue
		src_subnet=SrcAddr_octates[0]+"."+SrcAddr_octates[1]+"."+SrcAddr_octates[2]+".0/24"
		dst_subnet=DstAddr_octates[0]+"."+DstAddr_octates[1]+"."+DstAddr_octates[2]+".0/24"

		G.add_edge(src_subnet,SrcAddr)
		G.add_edge(SrcAddr,node_src)
		G.add_edge(node_src,node_dst)
		G.add_edge(node_dst,DstAddr)
		G.add_edge(DstAddr,dst_subnet)
		
	suspicious_nodes=extract_suspicious_nodes(G)
	#process_suspicious_nodes(suspicious_nodes)
	return suspicious_nodes,G


def check_node_activity(node_time_series):
	inX=[]#Data
	inT=[]#Times
	for values in node_time_series:
		inX.append(values[1])
		inT.append(values[0])
	nrow=len(inX)
	stds = np.std(inX)
	mean = np.mean(inX)
	anomalous_times=[]
	for i in range(nrow):
		if(inX[i] > (mean+3*stds) or inX[i] < (mean-2*stds) ):
			anomalous_times.append(inT[i])
	if len(anomalous_times) > 0:
		return 1,anomalous_times
	else:
		return 0,anomalous_times#Found nothing


def plot_tdg(dotfilename,G):
    #TODO: use pydot 
	nx.write_dot(G,dotfilename+'.dot')
	cmd="sfdp -Goverlap=prism -Tpng "+dotfilename+".dot > "+dotfilename+".png"
	os.system(cmd)
    
def makeLLS(X,T):
    X = np.hstack((np.ones((X.shape[0],1)),X))
    wt = np.linalg.solve(np.dot(X.T,X), np.dot(X.T,T))
    return wt

def useLLS(X,W):
    X = np.hstack((np.ones((X.shape[0],1)),X))
    pred = np.dot(X,W)
    return pred

def perform_lls_train(flow_matrix,names):    
    
    dataOriginal=np.array(flow_matrix,dtype=float)
    dataOriginal=dataOriginal[:,1:]
    #print(dataOriginal[0])
    #dataOriginal=np.transpose(dataOriginal)
    ########################Training########################
    #dataOriginal= np.loadtxt(sys.argv[1],delimiter=',',skiprows=1)
    data= dataOriginal[:,:dataOriginal.shape[1]-1]
    #data=forensics.standard(data)
    (rnum,cnum)=data.shape
    
    T=dataOriginal[:,dataOriginal.shape[1]-1:dataOriginal.shape[1]]
    #print(names)
    Tname=names[dataOriginal.shape[1]-1]
    #print(Tname)
    X=data[:,:dataOriginal.shape[1]-1]
    Xname=names[0:dataOriginal.shape[1]]
    
    #Standardize
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    #X=standard(X)
    
    rmseTrainavg=0
    rmseTestavg=0
    
    # Partition into training and testing sets.
    PARTITION=0.6
    #print("PARTITION: %s" % (PARTITION))
    
    nSamples =  X.shape[0]
    allI = range(nSamples)                     # indices for all data rows
    nTrain = int(round(nSamples*PARTITION))           # number of training samples
    #nTest = nSamples - nTrain
    #nTest=nSamples #use all samples for testing
    trainI = list(set(random.sample(allI,nTrain))) # row indices for training samples
    #testI = list(set(allI).union(set(trainI))) # row indices for testing samples
    Xtrain = X[trainI,:]
    Ttrain = T[trainI,:]
    #Xtest = X[testI,:]
    #Ttest = T[testI,:]
    
    #Add constant 1 attribute column and attribute name
    Xtrain = np.hstack((np.ones((nTrain,1)), Xtrain))
    #Xtest = np.hstack((np.ones((nTest,1)), Xtest))
    Xname=np.insert(names,0,'bias')
    # Make the linear model, which consists of just the column vector of weights.
    w = np.linalg.solve(np.dot(Xtrain.T,Xtrain), np.dot(Xtrain.T, Ttrain))
    
    return w
    
    
def perform_lls_predict(w,flow_matrix,names):
    ########################Testing#########################
    
    dataOriginal=np.array(flow_matrix,dtype=float)
    dataOriginal=dataOriginal[:,1:]
    #dataOriginal= np.loadtxt(sys.argv[2],delimiter=',',skiprows=1)
    #data= dataOriginal[:,:dataOriginal.shape[1]-1]
    data= dataOriginal[:,:dataOriginal.shape[1]-1]
    (rnum,cnum)=data.shape
    
    T=dataOriginal[:,dataOriginal.shape[1]-1:dataOriginal.shape[1]]
    Tname=names[dataOriginal.shape[1]-1]
    #print Tname
    X=data[:,:dataOriginal.shape[1]-1]
    Xname=names[0:dataOriginal.shape[1]]
    
    #Standardize
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    #X=standard(X)
    
    #Setup Testing Matrix
    # Partition into training and testing sets.
    PARTITION=1
    #print "PARTITION: %s" % (PARTITION)
    
    nSamples =  X.shape[0]
    allI = range(nSamples)                     # indices for all data rows
    nTrain = int(round(nSamples*PARTITION))           # number of training samples
    #nTest = nSamples - nTrain
    nTest=nSamples #use all samples for testing
    trainI = list(set(random.sample(allI,nTrain))) # row indices for training samples
    testI = list(set(allI).union(set(trainI))) # row indices for testing samples
    #Xtrain = X[trainI,:]
    #Ttrain = T[trainI,:]
    Xtest = X[testI,:]
    Ttest = T[testI,:]
    
    #Add constant 1 attribute column and attribute name
    #Xtrain = np.hstack((np.ones((nTrain,1)), Xtrain))
    Xtest = np.hstack((np.ones((nTest,1)), Xtest))
    
    #plt.figure(2,figsize=(20,20))
    #plt.clf()
    #for z in range(X.shape[1]):
    #        plt.subplot(5,len(names)/5+1,z+1)
    #        plt.plot(X[:,z],T,'.')
    #        plt.xlabel(Xname[z])
    #        plt.ylabel(Tname)
        #a = max(min(X[:,z]),min(T))
        #b = min(max(X[:,z]),max(T))
        #plt.plot([a,b],[a,b],'r-',linewidth=3)
    
    #plt.savefig('linear_data_vs_target.png')
    
    # Apply the model to the train and test sets.  Plot predicted versus target values of updrs.
    #TtrainPredicted = np.dot(Xtrain,w)
    TtestPredicted = np.dot(Xtest,w)
    return TtestPredicted[:,0]   
    
#     plt.figure(3)
#     plt.clf()
#     plt.subplot(1,2,1)
#     plt.plot(Ttrain,TtrainPredicted,'.')
#     plt.xlabel('Observed Dest Entropy') 
#     plt.ylabel('Predicted Dest Entropy')
#     plt.title('Training Day')
#     a = max(min(Ttrain),min(TtrainPredicted))
#     b = min(max(Ttrain),max(TtrainPredicted))
#     plt.plot([a,b],[a,b],'r-',linewidth=3)
#     #plt.savefig('assgnmnt1fig3.pdf')
#     
#     plt.figure(3)
#     plt.subplot(1,2,2)
#     plt.plot(Ttest,TtestPredicted,'.')
#     plt.xlabel('Original Dest Entropy')
#     plt.ylabel('Predicted Dest Entropy')
#     plt.title('Testing Day')
#     a = max(min(Ttest),min(TtestPredicted))
#     b = min(max(Ttest),max(TtestPredicted))
#     plt.plot([a,b],[a,b],'r-',linewidth=3)
#     
#     
#     #plt.show()
#     rmseTrain = np.sqrt(np.mean((Ttrain - TtrainPredicted)**2))
#     rmseTest = np.sqrt(np.mean((Ttest - TtestPredicted)**2))
#     print('RMSE: Train %.2f %s, Test %.2f %s' % (rmseTrain,Tname, rmseTest,Tname))
#     
#     sortedOrder = abs(w).argsort(axis=0).flatten()
#     sortedarray = sortedOrder[::-1]
#     #for (ni,wi) in zip(np.array(Xname)[sortedOrder], w[sortedarray]):
#     #print '%15s %4.4f' % (ni,wi[0])
#     
#     
#     '''
#     plt.figure(5)
#     plt.plot(np.arange(Ttrain.shape[0]),Ttrain,'.')
#     plt.plot(np.arange(Ttrain.shape[0]),TtrainPredicted,'.')
#     plt.legend(('Train','TrainPredicted'),'upper center')
#     #plt.savefig('linear_train_vs_train_predicted.png')
#     
#     plt.figure(6)
#     plt.plot(np.arange(Ttest.shape[0]),Ttest,'.')
#     plt.plot(np.arange(Ttest.shape[0]),TtestPredicted,'.')
#     plt.legend(('Test','TestPredicted'),'upper center')
#     #plt.savefig('linear_test_and_test_vs_predicted_.png')
#     '''
#     
#     plt.figure(7)
#     plt.plot(np.arange(Ttest.shape[0]),Ttest[:,0])
#     plt.plot(np.arange(Ttest.shape[0]),TtestPredicted[:,0])
#     plt.legend(('Observed Dest Entropy','Predicted Dest Entropy'),'best')
#     plt.xlabel('Samples')
#     plt.ylabel('Entropy')
#     plt.title('Prediction of Dest Entropy')
#     
#     
#     plt.show()
    

