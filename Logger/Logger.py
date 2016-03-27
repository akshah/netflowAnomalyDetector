
import time
import sys
import threading

class Logger():
	def current_time(self):
		    return int(time.time()),time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())

	def __init__(self,logfilename):
		self.lock = threading.RLock()
		self.logfilename = logfilename

	def print_log(self,msg):
		self.lock.acquire()
		try:

			#Log file
			logfile = open(self.logfilename,'a')

			_,localtime=self.current_time()
			time='['+localtime+']  '
			logfile.write(time+' INFO: '+msg+'\n')
			logfile.close()

		finally:
			self.lock.release()

