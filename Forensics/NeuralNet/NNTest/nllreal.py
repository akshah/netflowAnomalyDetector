import gradientDescent as gd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import sys

class NeuralNetClassifier:
    """ Neural network with one hidden layer.
 For nonlinear regression (prediction of real-valued outputs)
   net = NeuralNet(ni,nh,no)       # ni is number of attributes each sample,
                                   # nh is number of hidden units,
                                   # no is number of output components
   net.train(X,T,                  # X is nSamples x ni, T is nSamples x no
             nIterations=1000,     # maximum number of SCG iterations
             weightPrecision=1e-8, # SCG terminates when weight change magnitudes fall below this,
             errorPrecision=1e-8)  # SCG terminates when objective function change magnitudes fall below this
   Y,Z = net.use(Xtest,allOutputs=True)  # Y is nSamples x no, Z is nSamples x nh
"""
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

    def draw(self, inputNames = None, outputNames = None, gray = False):
        def isOdd(x):
            return x % 2 != 0

        W = [self.V, self.W]
        nLayers = 2

        # calculate xlim and ylim for whole network plot
        #  Assume 4 characters fit between each wire
        #  -0.5 is to leave 0.5 spacing before first wire
        xlim = max(map(len,inputNames))/4.0 if inputNames else 1
        ylim = 0
    
        for li in range(nLayers):
            ni,no = W[li].shape  #no means number outputs this layer
            if not isOdd(li):
                ylim += ni + 0.5
            else:
                xlim += ni + 0.5

        ni,no = W[nLayers-1].shape  #no means number outputs this layer
        if isOdd(nLayers):
            xlim += no + 0.5
        else:
            ylim += no + 0.5

        # Add space for output names
        if outputNames:
            if isOdd(nLayers):
                ylim += 0.25
            else:
                xlim += round(max(map(len,outputNames))/4.0)

        ax = plt.gca()

        x0 = 1
        y0 = 0 # to allow for constant input to first layer
        # First Layer
        if inputNames:
            #addx = max(map(len,inputNames))*0.1
            y = 0.55
            for n in inputNames:
                y += 1
                ax.text(x0-len(n)*0.2, y, n)
                x0 = max([1,max(map(len,inputNames))/4.0])

        for li in range(nLayers):
            Wi = W[li]
            ni,no = Wi.shape
            if not isOdd(li):
                # Odd layer index. Vertical layer. Origin is upper left.
                # Constant input
                ax.text(x0-0.2, y0+0.5, '1')
                for li in range(ni):
                    ax.plot((x0,x0+no-0.5), (y0+li+0.5, y0+li+0.5),color='gray')
                # output lines
                for li in range(no):
                    ax.plot((x0+1+li-0.5, x0+1+li-0.5), (y0, y0+ni+1),color='gray')
                # cell "bodies"
                xs = x0 + np.arange(no) + 0.5
                ys = np.array([y0+ni+0.5]*no)
                ax.scatter(xs,ys,marker='v',s=1000,c='gray')
                # weights
                if gray:
                    colors = np.array(["black","gray"])[(Wi.flat >= 0)+0]
                else:
                    colors = np.array(["red","green"])[(Wi.flat >= 0)+0]
                xs = np.arange(no)+ x0+0.5
                ys = np.arange(ni)+ y0 + 0.5
                aWi = abs(Wi)
                aWi = aWi / np.max(aWi) * 50
                coords = np.meshgrid(xs,ys)
                #ax.scatter(coords[0],coords[1],marker='o',s=2*np.pi*aWi**2,c=colors)
                ax.scatter(coords[0],coords[1],marker='s',s=aWi**2,c=colors)
                y0 += ni + 1
                x0 += -1 ## shift for next layer's constant input
            else:
                # Even layer index. Horizontal layer. Origin is upper left.
                # Constant input
                ax.text(x0+0.5, y0-0.2, '1')
                # input lines
                for li in range(ni):
                    ax.plot((x0+li+0.5,  x0+li+0.5), (y0,y0+no-0.5),color='gray')
                # output lines
                for li in range(no):
                    ax.plot((x0, x0+ni+1), (y0+li+0.5, y0+li+0.5),color='gray')
                # cell "bodies"
                xs = np.array([x0 + ni + 0.5]*no)
                ys = y0 + 0.5 + np.arange(no)
                ax.scatter(xs,ys,marker='>',s=1000,c='gray')
                # weights
                Wiflat = Wi.T.flatten()
                if gray:
                    colors = np.array(["black","gray"])[(Wiflat >= 0)+0]
                else:
                    colors = np.array(["red","green"])[(Wiflat >= 0)+0]
                xs = np.arange(ni)+x0 + 0.5
                ys = np.arange(no)+y0 + 0.5
                coords = np.meshgrid(xs,ys)
                aWi = abs(Wiflat)
                aWi = aWi / np.max(aWi) * 50
                #ax.scatter(coords[0],coords[1],marker='o',s=2*np.pi*aWi**2,c=colors)
                ax.scatter(coords[0],coords[1],marker='s',s=aWi**2,c=colors)
                x0 += ni + 1
                y0 -= 1 ##shift to allow for next layer's constant input

        # Last layer output labels 
        if outputNames:
            if isOdd(nLayers):
                x = x0+1.5
                for n in outputNames:
                    x += 1
                    ax.text(x, y0+0.5, n)
            else:
                y = y0+0.6
                for n in outputNames:
                    y += 1
                    ax.text(x0+0.2, y, n)
        ax.axis([0,xlim, ylim,0])
        ax.axis('off')

if __name__== "__main__":
	mylist=[]	
	D=[]
	cnt=0
	for line in sys.stdin:
		dataline = line.strip()
		mylist=dataline.split(",")
		tmplist=[]
		ncol=-1#Only X not T
		for ele in mylist:
		    tmplist.append(float(ele))
		    ncol=ncol+1
		D.append(tmplist)
		cnt=cnt+1
			

	data=np.array(D)

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

	nHidden = 3
	nnet = NeuralNetClassifier(ncol,nHidden,1) # 3 classes, will actually make 2-unit output layer

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


	
