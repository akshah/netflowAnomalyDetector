#!/usr/bin/env python
import sys
import numpy as np
import networkx as nx


#Definition of Calculation functions

def get_entropy(*args):
    xy = zip(*args)
    # probs
    proba = [ float(xy.count(c)) / len(xy) for c in dict.fromkeys(list(xy)) ]
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

def get_features(SrcAddrL,DstAddrL,TimestampL,InPktsL,InBytesL,DstPortL,SrcPortL,OutIfL,InIfL,RouterL,TCPL,ProtocolL,NextHopL,NumFlows):

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

