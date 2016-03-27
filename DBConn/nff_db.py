#!/usr/bin/python

#Read the next set of flows


import pymysql
import numpy as np

class netflow_connection(object):
	def __init__(self,dbname,table_name,host,port,user,passwd,interval):
		self.dbname=dbname
		self.table_name=table_name
		self.time_interval=interval
		self.db = pymysql.connect(host=host, # your host, usually localhost
		            port=port, 
					user=user, # your username
					 passwd=passwd, # your password
					 db=dbname) # name of the data base


	def getMinTime(self):
		qcur = self.db.cursor()
		qcur.execute("select min(timestamp) FROM {0}.{1}".format(self.dbname,self.table_name))
		row=qcur.fetchone()
		return row[0]

	def getMaxTime(self):
		qcur = self.db.cursor()
		qcur.execute("select max(timestamp) FROM {0}.{1}".format(self.dbname,self.table_name))
		row=qcur.fetchone()
		return row[0]

	def getColumnNames(self):
		qcur = self.db.cursor()
		qcur.execute("SHOW COLUMNS FROM {0}.{1}".format(self.dbname,self.table_name))
		cols=[]
		row=qcur.fetchone()
		while row:
			cols.append(row[0])
			row=qcur.fetchone()
		qcur.execute("SELECT COUNT(*) AS Columns FROM INFORMATION_SCHEMA.COLUMNS WHERE table_schema = '{0}' AND table_name = '{1}'".format(self.dbname,self.table_name))
		col_count=qcur.fetchone()
		return cols,int(col_count[0])

	def getNextFlows(self,old_pointer_time,interval_secs):
		pointer_time=old_pointer_time+interval_secs
		# Use all the SQL you like
		cur = self.db.cursor() 
		cur.execute("select INET_NTOA(srcaddr),INET_NTOA(dstaddr),timestamp,inpkts,inbytes,dstport,srcport,OutIf,InIf,router,tcpflags,Protocol,nextHop from {0}.{1} where timestamp > '{2}' and timestamp <= '{3}';".format(self.dbname,self.table_name,old_pointer_time,pointer_time))
		columns,numcols=self.getColumnNames()
		SrcAddrL,DstAddrL,TimestampL,InPktsL,InBytesL,DstPortL,SrcPortL,OutIfL,InIfL,RouterL,TCPL,ProtocolL,NextHopL = ([] for i in range(numcols))
		# print all the first cell of all the rows
		row = ''
		NumFlows=0
		row=cur.fetchone()
		while row:
			SrcAddrL.append(row[0])
			DstAddrL.append(row[1])
			TimestampL.append(int(row[2]))
			InPktsL.append(float(row[3]))
			InBytesL.append(float(row[4]))
			DstPortL.append(row[5])
			SrcPortL.append(row[6])
			OutIfL.append(row[7])
			InIfL.append(row[8])
			RouterL.append(row[9])
			TCPL.append(row[10])
			ProtocolL.append(row[11])
			NextHopL.append(row[12])

			NumFlows+=1
			row=cur.fetchone()

		NumFlowsL=[]
		NumFlowsL.append(NumFlows)
		return SrcAddrL,DstAddrL,TimestampL,InPktsL,InBytesL,DstPortL,SrcPortL,OutIfL,InIfL,RouterL,TCPL,ProtocolL,NextHopL,NumFlowsL

	def getNextFlowsFloats(self,old_pointer_time,interval_secs):
		pointer_time=old_pointer_time+interval_secs
		# Use all the SQL you like
		cur = self.db.cursor() 
		cur.execute("select srcaddr,dstaddr,timestamp,inpkts,inbytes,dstport,srcport,OutIf,InIf,router,tcpflags,Protocol,nextHop from {0}.{1} where timestamp > '{2}' and timestamp <= '{3}';".format(self.dbname,self.table_name,old_pointer_time,pointer_time))
		columns,numcols=self.getColumnNames()
		SrcAddrL,DstAddrL,TimestampL,InPktsL,InBytesL,DstPortL,SrcPortL,OutIfL,InIfL,RouterL,TCPL,ProtocolL,NextHopL = ([] for i in range(numcols))
		# print all the first cell of all the rows
		row = ''
		NumFlows=0
		row=cur.fetchone()
		while row:
			SrcAddrL.append(row[0])
			DstAddrL.append(row[1])
			TimestampL.append(int(row[2]))
			InPktsL.append(float(row[3]))
			InBytesL.append(float(row[4]))
			DstPortL.append(row[5])
			SrcPortL.append(row[6])
			OutIfL.append(row[7])
			InIfL.append(row[8])
			RouterL.append(row[9])
			TCPL.append(row[10])
			ProtocolL.append(row[11])
			NextHopL.append(row[12])

			NumFlows+=1
			row=cur.fetchone()

		NumFlowsL=[]
		NumFlowsL.append(NumFlows)
		return SrcAddrL,DstAddrL,TimestampL,InPktsL,InBytesL,DstPortL,SrcPortL,OutIfL,InIfL,RouterL,TCPL,ProtocolL,NextHopL,NumFlowsL
