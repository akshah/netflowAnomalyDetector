import os
import sys
import time
import numpy as np

def makeLLS(X,T):
	X = np.hstack((np.ones((X.shape[0],1)),X))
	wt = np.linalg.solve(np.dot(X.T,X), np.dot(X.T,T))
	return wt

def useLLS(X,W):
	X = np.hstack((np.ones((X.shape[0],1)),X))
	pred = np.dot(X,W)
	return pred

def perform_lls(flow_matrix,names):	
	
	dataOriginal=np.array(flow_matrix[0:len(flow_data_row)-1],dtype=float)
	dataOriginal=np.transpose(dataOriginal)
	########################Training########################
	#dataOriginal= np.loadtxt(sys.argv[1],delimiter=',',skiprows=1)
	data= dataOriginal[:,:dataOriginal.shape[1]-1]
	#data=forensics.standard(data)
	(rnum,cnum)=data.shape
	
	T=dataOriginal[:,dataOriginal.shape[1]-1:dataOriginal.shape[1]]
	Tname=names[dataOriginal.shape[1]-1]
	print(Tname)
	X=data[:,:dataOriginal.shape[1]-1]
	Xname=names[0:dataOriginal.shape[1]]
	
	#Standardize
	#X = (X - X.mean(axis=0)) / X.std(axis=0)
	forensics.standard(X)
	
	rmseTrainavg=0
	rmseTestavg=0
	
	# Partition into training and testing sets.
	PARTITION=0.3
	print("PARTITION: %s" % (PARTITION))
	
	nSamples =  X.shape[0]
	allI = xrange(nSamples)                     # indices for all data rows
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
	
	
	########################Testing#########################
	
	#dataOriginal= np.loadtxt(sys.argv[2],delimiter=',',skiprows=1)
	#data= dataOriginal[:,:dataOriginal.shape[1]-1]
	
	(rnum,cnum)=data.shape
	
	T=dataOriginal[:,dataOriginal.shape[1]-1:dataOriginal.shape[1]]
	Tname=names[dataOriginal.shape[1]-1]
	#print Tname
	X=data[:,:dataOriginal.shape[1]-1]
	Xname=names[0:dataOriginal.shape[1]]
	
	#Standardize
	#X = (X - X.mean(axis=0)) / X.std(axis=0)
	forensics.standard(X)
	
	#Setup Testing Matrix
	# Partition into training and testing sets.
	PARTITION=0.3
	#print "PARTITION: %s" % (PARTITION)
	
	nSamples =  X.shape[0]
	allI = xrange(nSamples)                     # indices for all data rows
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
	
	plt.figure(2,figsize=(20,20))
	plt.clf()
	for z in range(X.shape[1]):
	        plt.subplot(5,len(names)/5+1,z+1)
	        plt.plot(X[:,z],T,'.')
		plt.xlabel(Xname[z])
	        plt.ylabel(Tname)
		#a = max(min(X[:,z]),min(T))
		#b = min(max(X[:,z]),max(T))
		#plt.plot([a,b],[a,b],'r-',linewidth=3)
	
	#plt.savefig('linear_data_vs_target.png')
	
	# Apply the model to the train and test sets.  Plot predicted versus target values of updrs.
	TtrainPredicted = np.dot(Xtrain,w)
	TtestPredicted = np.dot(Xtest,w)
	
	
	plt.figure(3)
	plt.clf()
	plt.subplot(1,2,1)
	plt.plot(Ttrain,TtrainPredicted,'.')
	plt.xlabel('Observed Dest Entropy') 
	plt.ylabel('Predicted Dest Entropy')
	plt.title('Training Day')
	a = max(min(Ttrain),min(TtrainPredicted))
	b = min(max(Ttrain),max(TtrainPredicted))
	plt.plot([a,b],[a,b],'r-',linewidth=3)
	#plt.savefig('assgnmnt1fig3.pdf')
	
	plt.figure(3)
	plt.subplot(1,2,2)
	plt.plot(Ttest,TtestPredicted,'.')
	plt.xlabel('Original Dest Entropy')
	plt.ylabel('Predicted Dest Entropy')
	plt.title('Testing Day')
	a = max(min(Ttest),min(TtestPredicted))
	b = min(max(Ttest),max(TtestPredicted))
	plt.plot([a,b],[a,b],'r-',linewidth=3)
	
	
	#plt.show()
	rmseTrain = np.sqrt(np.mean((Ttrain - TtrainPredicted)**2))
	rmseTest = np.sqrt(np.mean((Ttest - TtestPredicted)**2))
	print('RMSE: Train %.2f %s, Test %.2f %s' % (rmseTrain,Tname, rmseTest,Tname))
	
	sortedOrder = abs(w).argsort(axis=0).flatten()
	sortedarray = sortedOrder[::-1]
	#for (ni,wi) in zip(np.array(Xname)[sortedOrder], w[sortedarray]):
	#print '%15s %4.4f' % (ni,wi[0])
	
	
	'''
	plt.figure(5)
	plt.plot(np.arange(Ttrain.shape[0]),Ttrain,'.')
	plt.plot(np.arange(Ttrain.shape[0]),TtrainPredicted,'.')
	plt.legend(('Train','TrainPredicted'),'upper center')
	#plt.savefig('linear_train_vs_train_predicted.png')
	
	plt.figure(6)
	plt.plot(np.arange(Ttest.shape[0]),Ttest,'.')
	plt.plot(np.arange(Ttest.shape[0]),TtestPredicted,'.')
	plt.legend(('Test','TestPredicted'),'upper center')
	#plt.savefig('linear_test_and_test_vs_predicted_.png')
	'''
	
	plt.figure(7)
	plt.plot(np.arange(Ttest.shape[0]),Ttest[:,0])
	plt.plot(np.arange(Ttest.shape[0]),TtestPredicted[:,0])
	plt.legend(('Observed Dest Entropy','Predicted Dest Entropy'),'best')
	plt.xlabel('Samples')
	plt.ylabel('Entropy')
	plt.title('Prediction of Dest Entropy')
	
	
	plt.show()
