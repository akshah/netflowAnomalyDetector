from __future__ import print_function

from DBConn.nff_db import netflow_connection as conn
from Logger.Logger import Logger
import Forensics.nff_forensics as forensics
#import forensics.LLS.lls_flow as lls
from ThreadPool.TPool import TPool
#import threading
import os
import sys
import time
import random
import numpy as np
import networkx as nx
from collections import defaultdict
from collections import OrderedDict
from matplotlib import pyplot as plt


#CONFIG
#TODO: Read from a config file
interval_secs=1
block_secs=60
start_time=1410931777
NUM_THREADS=1


#Logger
scriptname=sys.argv[0].split('.')
logfilename=scriptname[0]+'.log'
logger=Logger(logfilename)

showPlots=True

#dbname,table_name,host,port,user,passwd,interval_sec
fetch=conn('nsas','ahtflows','172.19.5.24',3307,'nsas','nqnsas',interval_secs)
names=['HProto','HSrcAddr','HDstAddr','HSport','HDport','TInPkts','TInBytes','AvgPktSize','StdOutDegree','StdRatioDegree','NumFlows','Score','Marker','NumFlowsPredicted']
names_for_lls=['HProto','HSrcAddr','HDstAddr','HSport','HDport','TInPkts','TInBytes','AvgPktSize','StdOutDegree','StdRatioDegree','NumFlows']
output_file='3news_out.txt'

#Create Figure to show features
(FHProto,FHSrcAddr,FHDstAddr,FHSport,FHDport,FTInPkts,FTInBytes,FAvgPktSize,FStdOutDegree,FStdRatioDegree,FNumFlows,FScore,FNumFlowsP)=([1]*(int(block_secs/interval_secs)) for i in range(len(names)-1))
FMarker=[-30]*(int(block_secs/interval_secs))
#dat=[0]*(block_secs/interval_secs)
#dat2=[0]*(block_secs/interval_secs)
total_subplot=8
subplot_col=1
subnum=1

fig = plt.figure(figsize=(20, 16))
ax_entropy = fig.add_subplot(total_subplot,subplot_col,subnum)
subnum+=1
ax_tpkts = fig.add_subplot(total_subplot,subplot_col, subnum)
subnum+=1
ax_pktsize = fig.add_subplot(total_subplot,subplot_col, subnum)
subnum+=1
ax_numflows = fig.add_subplot(total_subplot,subplot_col, subnum)
subnum+=1
ax_numflowsp = fig.add_subplot(total_subplot,subplot_col, subnum)
subnum+=1
ax_tdg = fig.add_subplot(total_subplot,subplot_col, subnum)
subnum+=1
ax_score = fig.add_subplot(total_subplot,subplot_col, subnum)

Ln_entropySrcAddr, = ax_entropy.plot(FHSrcAddr,label="SrcAddr Entropy")
Ln_entropyDstAddr, = ax_entropy.plot(FHDstAddr,label="DstAddr Entropy")
Ln_entropySrcPort, = ax_entropy.plot(FHSport,label="SrcPort Entropy")
Ln_entropyDstPort, = ax_entropy.plot(FHDport,label="DstPort Entropy")

Ln_Itpkts, = ax_tpkts.plot(FTInPkts,label="#Incoming\nPackets")
#Ln_Ibytes, = ax_tpkts.plot(FTInBytes,label="#Incoming Bytes")

Ln_avgpktsize, = ax_pktsize.plot(FAvgPktSize,label="Average Pkt Size",color='g')

Ln_numflows, = ax_numflows.plot(FNumFlows,label="NumFlows",color='c')
Ln_numflowsp, = ax_numflowsp.plot(FNumFlowsP,label="NumFlows\nPredicted",color='m')

#Ln_tdgod, = ax_tdg.plot(FStdOutDegree,label="StdDev FanOut")
Ln_tdgrd, = ax_tdg.plot(FStdRatioDegree,label="StdDev I/O Ratio")

Ln_score, = ax_score.plot(FScore,label="Anomaly Score")
Ln_marker, = ax_score.plot(FMarker,'ro')


ax_entropy.set_xlim([0,(block_secs/interval_secs)])

if showPlots:
	
	plt.ion()
	plt.show()   


def plot_features(feature_matrix):
		
	for features in feature_matrix:
		Timestamp,HProto,HSrcAddr,HDstAddr,HSport,HDport,TInPkts,TInBytes,AvgPktSize,StdOutDegree,StdRatioDegree,NumFlows,Score,Marker,NumFlowsP=features		
	
		#Show Entropies
		ymin = float(min(min(FHSrcAddr),min(FHDstAddr),min(FHSport),min(FHDport)))-3
		ymax = float(max(max(FHSrcAddr),max(FHDstAddr),max(FHSport),max(FHDport)))+3
		ax_entropy.set_ylim([ymin,ymax])
		
		FHSrcAddr.append(HSrcAddr)
		del FHSrcAddr[0]
		Ln_entropySrcAddr.set_ydata(FHSrcAddr)
		Ln_entropySrcAddr.set_xdata(range(len(FHSrcAddr)))
		FHDstAddr.append(HDstAddr)
		del FHDstAddr[0]
		Ln_entropyDstAddr.set_ydata(FHDstAddr)
		Ln_entropyDstAddr.set_xdata(range(len(FHDstAddr)))
		FHSport.append(HSport)
		del FHSport[0]
		Ln_entropySrcPort.set_ydata(FHSport)
		Ln_entropySrcPort.set_xdata(range(len(FHSport)))
		FHDport.append(HDport)
		del FHDport[0]
		Ln_entropyDstPort.set_ydata(FHDport)
		Ln_entropyDstPort.set_xdata(range(len(FHDport)))
		
		ax_entropy.set_ylabel('Entropy')
		ax_entropy.legend(loc='lower right')
		ax_entropy.tick_params(\
		axis='x',          # changes apply to the x-axis
		which='both',      # both major and minor ticks are affected
		bottom='off',      # ticks along the bottom edge are off
		top='off',         # ticks along the top edge are off
		labelbottom='off'
		) # labels along the bottom edge are off  

		
		
		
		FTInPkts.append(TInPkts)
		del FTInPkts[0]
		Ln_Itpkts.set_ydata(FTInPkts)
		#FTInBytes.append(TInBytes)
		#del FTInBytes[0]
		#Ln_Ibytes.set_ydata((FTInBytes))
			
		#Show Incoming Volume
		ymin = float(min(FTInPkts))-(min(FTInPkts)/10)
		ymax = float(max(FTInPkts))+(max(FTInPkts)/10)
		ax_tpkts.set_ylim([-100,ymax])
		
		ax_tpkts.set_ylabel('#Packets\n(log scale)')
		ax_tpkts.set_yscale('log')
		ax_tpkts.legend(loc='lower right')
		ax_tpkts.tick_params(\
		axis='x',          # changes apply to the x-axis
		which='both',      # both major and minor ticks are affected
		bottom='off',      # ticks along the bottom edge are off
		top='off',         # ticks along the top edge are off
		labelbottom='off'
		) # labels along the bottom edge are off  
		
		
		#Show Number of Flows
		ymin = float(min(FNumFlows))-100
		ymax = float(max(FNumFlows))+100
		ax_numflows.set_ylim([ymin,ymax])
		FNumFlows.append(NumFlows)
		#ax_numflows.set_yscale('log')
		del FNumFlows[0]
		Ln_numflows.set_ydata(FNumFlows)
		Ln_numflows.set_xdata(range(len(FNumFlows)))
		ax_numflows.set_ylabel('#Flows')
		ax_numflows.legend(loc='lower right')
		ax_numflows.tick_params(\
		axis='x',          # changes apply to the x-axis
		which='both',      # both major and minor ticks are affected
		bottom='off',      # ticks along the bottom edge are off
		top='off',         # ticks along the top edge are off
		labelbottom='off'
		) # labels along the bottom edge are off  

		#Show NumFlows Predicted
		ymin = float(min(FNumFlowsP))-100
		ymax = float(max(FNumFlowsP))+100
		ax_numflowsp.set_ylim([ymin,ymax])
		FNumFlowsP.append(NumFlowsP)
		#ax_numflowsp.set_yscale('log')
		del FNumFlowsP[0]
		Ln_numflowsp.set_ydata(FNumFlowsP)
		Ln_numflowsp.set_xdata(range(len(FNumFlowsP)))
		ax_numflowsp.set_ylabel('#Flows\nPredicted')
		ax_numflowsp.legend(loc='lower right')
		ax_numflowsp.tick_params(\
		axis='x',          # changes apply to the x-axis
		which='both',      # both major and minor ticks are affected
		bottom='off',      # ticks along the bottom edge are off
		top='off',         # ticks along the top edge are off
		labelbottom='off'
		) # labels along the bottom edge are off  

		
		#Show Average Pkt Size
		ymin = float(min(FAvgPktSize))-10
		ymax = float(max(FAvgPktSize))+10
		ax_pktsize.set_ylim([ymin,ymax])
		FAvgPktSize.append(AvgPktSize)
		del FAvgPktSize[0]
		Ln_avgpktsize.set_ydata(FAvgPktSize)
		ax_pktsize.set_ylabel('Avg Pkt Size')
		ax_pktsize.legend(loc='lower right')
		ax_pktsize.tick_params(\
		axis='x',          # changes apply to the x-axis
		which='both',      # both major and minor ticks are affected
		bottom='off',      # ticks along the bottom edge are off
		top='off',         # ticks along the top edge are off
		labelbottom='off'
		) # labels along the bottom edge are off  
		
		#Show TDG Std Dev
		ymin = float(min(FStdRatioDegree))-10
		ymax = float(max(FStdRatioDegree))+10
		ax_tdg.set_ylim([ymin,ymax])
		
		#FStdOutDegree.append(StdOutDegree)
		#print(StdOutDegree)
		#del FStdOutDegree[0]
		#Ln_tdgod.set_ydata(FStdOutDegree)
		
		FStdRatioDegree.append(StdRatioDegree)
		del FStdRatioDegree[0]
		Ln_tdgrd.set_ydata(FStdRatioDegree)
		
		
		ax_tdg.set_ylabel('Std Deviation')
		ax_tdg.legend(loc='lower right')
		ax_tdg.tick_params(\
		axis='x',          # changes apply to the x-axis
		which='both',      # both major and minor ticks are affected
		bottom='off',      # ticks along the bottom edge are off
		top='off',         # ticks along the top edge are off
		labelbottom='off'
		) # labels along the bottom edge are off  
		
		#Show Anomaly Score
		#Show Average Pkt Size
		#ymin = float(min(FScore))-10
		#ymax = float(max(FScore))+10
		ax_score.set_ylim([0,120])
		#ax_score.set_xlim([0,120])
		FScore.append(Score)
		del FScore[0]
		Ln_score.set_ydata(FScore)
		FMarker.append(Marker)
		del FMarker[0]
		Ln_marker.set_ydata(FMarker)
		ax_score.set_ylabel('Score')
		ax_score.set_xlabel('Last 5 minutes')
		ax_score.legend(loc='lower right')
		#ax_score.tick_params(\
		#axis='x',          # changes apply to the x-axis
		#which='both',      # both major and minor ticks are affected
		#bottom='off',      # ticks along the bottom edge are off
		#top='off',         # ticks along the top edge are off
		#labelbottom='off'
		#) # labels along the bottom edge are off  
		
		plt.pause(0.25)
		
#Fetch the first timestamp
#If no start time is give use min time
if start_time:
	timestamp=start_time
else:	
	timestamp=fetch.getMinTime()
	
if not timestamp:
	print("Start time unknown")
	exit(0)	
	
#Fetch the last time
maxtimestamp=fetch.getMaxTime()
#Round number
round=0

feature_matrix_previous=[]
flow_data=[]
feature_matrix=[]
nodes_to_track=defaultdict(list)
def_malicious_nodes=defaultdict(list)

while timestamp<=maxtimestamp:
	#Variables to hold this set of flow data	
	del flow_data[:]
	del feature_matrix[:]
	nodes_to_track.clear()
	def_malicious_nodes.clear()
	TDG=nx.DiGraph()

	time_filename=timestamp#Will be used to save result images
	#Loop for block
	logger.print_log('Fetching data')
	for looper in range(0,int(block_secs/interval_secs)):
		#TODO: Use threads
		#Use DB object to get Next set of flows
		#SrcAddrL,DstAddrL,TimestampL,InPktsL,InBytesL,DstPortL,SrcPortL,OutIfL,InIfL,RouterL,TCPL,ProtocolL,NextHopL,NumFlowsL=fetch.getNextFlows(timestamp,interval_secs)
		timestamp+=interval_secs
		flow_data_row=[]	
		flow_data_row=fetch.getNextFlows(timestamp,interval_secs)
 	
		#print(flow_data_row[13][0])
		if flow_data_row[13][0] == 0:#If NumFlows  = 0
			continue
		if not flow_data_row[13][0]:#If NumFlows  is not defined
			continue
		flow_data.append(flow_data_row)
	
	if(len(flow_data)==0):
  		continue
  	
	round+=1#We are going forward with this round
	print(round)
  	
	logger.print_log('Generating Features')
	#Use Forensics object to get features	
	#Timestamp,HProto,HSrcAddr,HDstAddr,HSport,HDport,TInPkts,TInBytes,AvgPktSize,StdOutDegree,StdRatioDegree,NumFlows
	pool=TPool(numThreads=NUM_THREADS)
	feature_matrix=pool.getResultViaThreads(forensics.get_features,flow_data)
	#for d in flow_data:
	#	tmp=forensics.get_features(d)
	#	feature_matrix.append(tmp)
  	
	NumFlowsPL=[0]*len(feature_matrix)
	if(round>1):
		logger.print_log('Performing LLS')
		weights=forensics.perform_lls_train(feature_matrix_previous,names_for_lls)
		NumFlowsPL=forensics.perform_lls_predict(weights,feature_matrix,names_for_lls)
		
	del feature_matrix_previous[:]
	feature_matrix_previous=feature_matrix[:]

	logger.print_log('Performing PCA')
	#Get anomalous times using PCA and create TDG for those set if flows
	anomalous_flows_times,anomaly_scores,markers_on=forensics.perform_pca(feature_matrix,names,time_filename)
	
	markers=[-30]*len(feature_matrix)
	for tagval in markers_on:
		#print(len(feature_matrix),tagval)
		markers[tagval]=anomaly_scores[tagval]
		
	feature_matrix_with_scores=[]
	for v in range (0,len(feature_matrix)):
		Timestamp,HProto,HSrcAddr,HDstAddr,HSport,HDport,TInPkts,TInBytes,AvgPktSize,StdOutDegree,StdRatioDegree,NumFlows=feature_matrix[v]
		add_tupple=(Timestamp,HProto,HSrcAddr,HDstAddr,HSport,HDport,TInPkts,TInBytes,AvgPktSize,StdOutDegree,StdRatioDegree,NumFlows,anomaly_scores[v],markers[v],NumFlowsPL[v])
		feature_matrix_with_scores.append(add_tupple)
	
	
	if showPlots:
		#threading.Thread(target=plot_features, args=feature_matrix_with_scores)
		plot_features(feature_matrix_with_scores)
 
	logger.print_log('Generating Traffic Distribution Graphs for anomalous seconds')
 
	for ptime in anomalous_flows_times:
		SrcAddrL,DstAddrL,TimestampL,InPktsL,InBytesL,DstPortL,SrcPortL,OutIfL,InIfL,RouterL,TCPL,ProtocolL,NextHopL,NumFlowsL=fetch.getNextFlows(ptime,interval_secs)
		suspicious_nodes,TDG=forensics.get_tdg(SrcAddrL,DstAddrL,TimestampL,InPktsL,InBytesL,DstPortL,SrcPortL,OutIfL,InIfL,RouterL,TCPL,ProtocolL,NextHopL,NumFlowsL)
 		
		for node in suspicious_nodes:
			tm_val=[]
			tm_val.append(ptime)
			tm_val.append(suspicious_nodes[node])
			nodes_to_track[node].append(tm_val)
 
 
 
	logger.print_log('Checking suspicious node activity')
	#--------------Get timeseries of out-degree of each suspected IP
	for node in nodes_to_track:
		node_timeseries=[]
		for val in nodes_to_track[node]:
			node_timeseries.append(val)
		is_malicious,anomalous_times=forensics.check_node_activity(node_timeseries)
		if(is_malicious):
			def_malicious_nodes[node].append(anomalous_times)
 
 
	#Sort to get the most malicious nodes
	sorted_def_malicious_nodes = OrderedDict(sorted(def_malicious_nodes.items(), key=lambda x: len(x[1])))
 
	outfile=open(output_file,'a')
	for node in sorted_def_malicious_nodes:
		outfile.write('['+str(time_filename)+']	'+node+'\n')
	outfile.close()
	#exit(0)	
	logger.print_log('Finished Processing')

	


