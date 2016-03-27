from multiprocessing import Pool as ThreadPool
from multiprocessing import cpu_count


class TPool(object):
    '''
    Thread Pools that are meant to be used only once as they will be cleaned up
    '''
    numberOfThreads = cpu_count()

    def __init__(self, numThreads):
        '''
        Constructor
        '''
        if numThreads is not None:
            self.numThreads = numThreads
        else:
            self.numThreads = TPool.numberOfThreads
	
        self.pool = self._setUpP()
    
    '''
    Threading pices
    '''
    # Creates the thread pool
    def _setUpP(self):
        pool = ThreadPool(processes=self.numThreads)
        return pool
    
    # Will run the function with the arguments
    def getResultViaThreads(self, functionCall, args):
        toReturn = self.pool.map(functionCall, args) 
        self.pool.close()
        self.pool.join()
        return toReturn
