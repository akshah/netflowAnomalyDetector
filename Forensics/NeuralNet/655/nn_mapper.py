#!/usr/bin/env python
#
# Adapted from an example by Michael G. Noll at:
#
# http://www.michael-noll.com/wiki/Writing_An_Hadoop_MapReduce_Program_In_Python
#
'''
import sys, urllib, re
import numpy as np
# Read pairs as lines of input from STDIN
#for line in sys.stdin:
    # We assume that we are fed a series of URLs, one per line
#    url = line.strip()
    # Fetch the content and output the title (pairs are tab-delimited)
dataOriginal= np.loadtxt('wine.data')
print "Shah","\t", "Anant"
'''

import hashlib
import Image
import sys

data=list()
imagelist=list()
imgname=list()
for line in sys.stdin:
	dataline = line.strip()
	imgname=dataline.split("\t")	
	#imagelist.append(dataline)	
	imagelist.append(imgname[1])	
	#print "Keyr1","\tThis is length",len(imagelist)
	#print "Keyr1","\t",name[1]

def md5Checksum(filePath):
    fh = open(filePath, 'rb')
    m = hashlib.md5()
    while True:
        data = fh.read(8192)
        if not data:
            break
        m.update(data)
    return m.hexdigest()

for z in range(len(imagelist)):
	#print 'The MD5 checksum of ',imagelist[i],'is ', md5Checksum(imagelist[i])
	im = Image.open(imagelist[z])
	pix= im.load()
	size= im.size
	#print 'Column: ',size[0],'Rows :',size[1]
	#print pix[100,100][1]

	#calculate row mean
	rowmeanR=list()
	rowmeanG=list()
	rowmeanB=list()
	csumr=0
	csumg=0
	csumb=0	
	for r in range(size[1]):
		csumr=0
		csumg=0
		csumb=0	
		for c in range(size[0]):
			csumr=csumr+pix[c,r][0]
			csumg=csumg+pix[c,r][1]
			csumb=csumb+pix[c,r][2]	
		rowmeanR.append(csumr/size[0])
		rowmeanG.append(csumg/size[0])
		rowmeanB.append(csumb/size[0])
	csumr=0
	csumg=0
	csumb=0	
	for i in range(len(rowmeanR)):
		csumr=csumr+rowmeanR[i]
	for i in range(len(rowmeanG)):
		csumg=csumg+rowmeanG[i]
	for i in range(len(rowmeanB)):
		csumb=csumb+rowmeanB[i]
	ROWMEAN_R=csumr/len(rowmeanR)
	ROWMEAN_G=csumg/len(rowmeanG)
	ROWMEAN_B=csumb/len(rowmeanB)

	#calculate column mean
	colmeanR=list()
	colmeanG=list()
	colmeanB=list()
	rsumr=0
	rsumg=0
	rsumb=0	
	for c in range(size[0]):
		rsumr=0
		rsumg=0
		rsumb=0	
		for r in range(size[1]):
			rsumr=rsumr+pix[c,r][0]
			rsumg=rsumg+pix[c,r][1]
			rsumb=rsumb+pix[c,r][2]	
		colmeanR.append(rsumr/size[1])
		colmeanG.append(rsumg/size[1])
		colmeanB.append(rsumb/size[1])
	rsumr=0
	rsumg=0
	rsumb=0	
	for i in range(len(colmeanR)):
		rsumr=rsumr+colmeanR[i]
	for i in range(len(colmeanG)):
		rsumg=rsumg+colmeanG[i]
	for i in range(len(colmeanB)):
		rsumb=rsumb+colmeanB[i]
	COLMEAN_R=rsumr/len(colmeanR)
	COLMEAN_G=rsumg/len(colmeanG)
	COLMEAN_B=rsumb/len(colmeanB)

	#calculate DCT Row mean
	dctrowmeanR=list()
	dctrowmeanG=list()
	dctrowmeanB=list()
	csumr=0
	csumg=0
	csumb=0	
	for r in range(size[1]):
		csumr=0
		csumg=0
		csumb=0	
		for c in range(size[0]):
			csumr=csumr+pix[c,r][0]
			csumg=csumg+pix[c,r][1]
			csumb=csumb+pix[c,r][2]	
		dctrowmeanR.append(csumr/size[0])
		dctrowmeanG.append(csumg/size[0])
		dctrowmeanB.append(csumb/size[0])
	csumr=0
	csumg=0
	csumb=0	
	for i in range(len(dctrowmeanR)):
		csumr=csumr+dctrowmeanR[i]
	for i in range(len(dctrowmeanG)):
		csumg=csumg+dctrowmeanG[i]
	for i in range(len(dctrowmeanB)):
		csumb=csumb+dctrowmeanB[i]
	DCT_ROWMEAN_R=csumr/len(dctrowmeanR)
	DCT_ROWMEAN_G=csumg/len(dctrowmeanG)
	DCT_ROWMEAN_B=csumb/len(dctrowmeanB)
	
	#select category
	category=1
	tokens = imagelist[z].split('.')
	if int(tokens[0]) >= 1:
		if int(tokens[0]) < 6:
			category=1
		if (int(tokens[0]) >= 6) and (int(tokens[0]) < 11):
			category=2
		if (int(tokens[0]) >= 11) and (int(tokens[0]) < 16):
			category=3

	#print features of this image
	#print category,"\t",category,",",ROWMEAN_R,",",ROWMEAN_G,",",ROWMEAN_B,",",COLMEAN_R,",",COLMEAN_G,",",COLMEAN_B
	
	outstr=str(category)+","+str(ROWMEAN_R)+","+str(ROWMEAN_G)+","+str(ROWMEAN_B)+","+str(COLMEAN_R)+","+str(COLMEAN_G)+","+str(COLMEAN_B)
	print "keyr1","\t",outstr
	print "keyr2","\t",outstr
	print "keyr3","\t",outstr
	


	



