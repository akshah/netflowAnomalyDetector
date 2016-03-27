#!/usr/bin/python

import sys
import os
import networkx as nx
import numpy as np


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
		if out_degree_dict[node] > (mean_out_degree+3*std_out_degree) or ratio_degree_dict[node]>(mean_ratio_degree+2*std_ratio_degree):

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
		print suspicious_nodes[node],"\t",node


def get_tdg(SrcAddrL,DstAddrL,TimestampL,InPktsL,InBytesL,DstPortL,SrcPortL,OutIfL,InIfL,RouterL,TCPL,ProtocolL,NextHopL,NumFlows):
	#Initialize graph
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
	nx.write_dot(G,dotfilename+'.dot')
	cmd="sfdp -Goverlap=prism -Tpng "+dotfilename+".dot > "+dotfilename+".png"
	os.system(cmd)
