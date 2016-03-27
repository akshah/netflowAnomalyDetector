from ThreadPool.TPool import TPool

def Afunc(val):
	print(val)

somedata=[1,2,3,4,5,6,7,8,9,0]

pool=TPool(numThreads=2)

returns=pool.getResultViaThreads(Afunc,somedata)

