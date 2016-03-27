#!/usr/bin/env python
#
# Adapted from an example by Michael G. Noll at:
#
# http://www.michael-noll.com/wiki/Writing_An_Hadoop_MapReduce_Program_In_Python
#
from __future__ import with_statement 
from operator import itemgetter
import sys
import re

writerow=list()
for line in sys.stdin:
    line = line.strip()
    datarow = line.split("\t")
    writerow.append(datarow[1]) 
    #print datarow[1]
    #print line
  
with open("img.out","a+") as f:
    for item in writerow:
	f.write("%s\n" % item)
f.closed

'''
#File access
with open("img.out","r") as f:
	print "Opened the file for reading contents"	
	for ln in f:
    		print ln,
f.closed
#FILE = open("img.out","a")
#FILE.write(line)
#FILE.close()
'''
import gradientDescent as gd
import numpy as np
import random

class NeuralNetClassifier:
    def __init__(self,ni,nh,no):
        self.V = np.random.uniform(-0.1,0.1,size=(1+ni,nh))
        self.W = np.random.uniform(-0.1,0.1,size=(1+nh,no))
        self.ni,self.nh,self.no = ni,nh,no
        self.standardize = None
        self.standardizeT = None
        self.unstandardizeT = None
        #self.makeIndicatorVars= None
    
    def makeStandardize(self,X):
        means = X.mean(axis=0)
        stds = X.std(axis=0)
        def standardize(origX):
            return (origX - means) / stds
        def unStandardize(stdX):
            return stds * stdX + means
        return (standardize, unStandardize)

    def train(self,X,T,nIterations,weightPrecision=1.0e-4,errorPrecision=1.0e-4):
        if self.standardize is None:
            self.standardize,_ = self.makeStandardize(X)
        X = self.standardize(X)
        X1 = self.addOnes(X)

        if self.standardizeT is None:
            self.standardizeT,self.unstandardizeT = self.makeStandardize(T)
        T = self.standardizeT(T)

	def makeIndicatorVars(T):
		if T.ndim == 1:
    			T = T.reshape((-1,1))    
		return (T == np.unique(T)).astype(int)

        # Local functions used by gradientDescent.scg()
        def pack(V,W):
            return np.hstack((V.flat,W.flat))

        def unpack(w):
            self.V[:] = w[:(self.ni+1)*self.nh].reshape((self.ni+1,self.nh))
            self.W[:] = w[(self.ni+1)*self.nh:].reshape((self.nh+1,self.no))

	T = makeIndicatorVars(T)

        def objectiveF(w):
		unpack(w)
		Y = np.dot(self.addOnes(np.tanh(np.dot(X1,self.V))), self.W)
		expY = np.exp(Y)
		denom = 1 + np.sum(expY,axis=1).reshape((-1,1))
		Y = np.hstack((expY / denom, 1/denom))
		return -np.sum(T * np.log(Y))

        def gradF(w):
		unpack(w)
		Z = np.tanh(np.dot( X1, self.V ))
		Z1 = self.addOnes(Z)
		Y = np.dot( Z1, self.W )
		expY = np.exp(Y)
		denom = 1 + np.sum(expY,axis=1).reshape((-1,1))
		Y = np.hstack((expY /denom , 1.0/denom))
		error = (T[:,:-1] - Y[:,:-1])
		dV = -np.dot( X1.T, np.dot( error, self.W[1:,:].T) * (1-Z**2))
		dW = -np.dot( Z1.T, error) 
		return pack(dV,dW)

        scgresult = gd.scg(pack(self.V,self.W), objectiveF, gradF,xPrecision = weightPrecision,fPrecision = errorPrecision,nIterations = nIterations,ftracep=True)

        unpack(scgresult['x'])
        self.reason = scgresult['reason']
        self.errorTrace = scgresult['ftrace']
        self.numberOfIterations = len(self.errorTrace)

    def getNumberOfIterations(self):
        return self.numberOfIterations
        
    def use(self,X,allOutputs=False):
	X = self.standardize(X)
        X1 = self.addOnes(X)
        Z = np.tanh(np.dot( X1, self.V ))
	Z1 = self.addOnes(Z)
	fs = np.exp(np.dot(Z1, self.W))  # N x K-1
        denom = (1 + np.sum(fs,axis=1)).reshape((-1,1))
        gs = fs / denom
        return np.hstack((gs,1/denom))
    
    def plotErrors(self):
        plt.plot(self.errorTrace)
        plt.ylabel("Train MSE")
        plt.xlabel("Epochs")

    def addOnes(self,X):
        return np.hstack((np.ones((X.shape[0],1)),X))

data= np.loadtxt('img.out', delimiter=',')
#print data.shape

X = data[:,1:]
T = data[:,0].reshape((-1,1))

trainf = 0.8
c1I,_ = np.where(T == 1)
c2I,_ = np.where(T == 2)
c3I,_ = np.where(T == 3)

c1I = np.random.permutation(c1I)
c2I = np.random.permutation(c2I)
c3I = np.random.permutation(c3I)

nc1 = len(c1I)
nc2 = len(c2I)
nc3 = len(c3I)

n = round(trainf*len(c1I))
rows = c1I[:n]
Xtrain = X[rows,:]
Ttrain = T[rows,:]
rows = c1I[n:]
Xtest =  X[rows,:]
Ttest =  T[rows,:]

n = round(trainf*len(c2I))
rows = c2I[:n]
Xtrain = np.vstack((Xtrain, X[rows,:]))
Ttrain = np.vstack((Ttrain, T[rows,:]))
rows = c2I[n:]
Xtest = np.vstack((Xtest, X[rows,:]))
Ttest = np.vstack((Ttest, T[rows,:]))

n = round(trainf*len(c3I))
rows = c3I[:n]
Xtrain = np.vstack((Xtrain, X[rows,:]))
Ttrain = np.vstack((Ttrain, T[rows,:]))
rows = c3I[n:]
Xtest = np.vstack((Xtest, X[rows,:]))
Ttest = np.vstack((Ttest, T[rows,:]))

nHidden = 50
nnet = NeuralNetClassifier(6,nHidden,2) # 3 classes, will actually make 2-unit output layer

nnet.train(Xtrain,Ttrain,nIterations=1000, weightPrecision=0,errorPrecision=0)

#print "SCG stopped after",len(nnet.errorTrace),"iterations:",nnet.reason
Ytrain = nnet.use(Xtrain)
predTrain=np.argmax(Ytrain, axis=1)+1
Ytest = nnet.use(Xtest)
predTest=np.argmax(Ytest, axis=1)+1

def percentCorrect(p,t):
	return np.sum(p.ravel()==t.ravel()) / float(len(t)) * 100

#print "Percent correct(Non Linear Logistic Regression):"
print "Train",percentCorrect(predTrain,Ttrain),"Test",percentCorrect(predTest,Ttest)
#print Ttest
#print data

