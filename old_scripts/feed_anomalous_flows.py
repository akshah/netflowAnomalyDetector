#!/usr/bin/python 

import time
import sys
import os

for timestamp in sys.stdin:
    timestamp = timestamp.strip()   
    ts="%s.00" % (timestamp)
    path=time.strftime('keyvalue_archive/%Y/%m/%d/%H/argus.%H.%M.%S.gz.data', time.localtime(float(ts)))
    cmd="cat %s" % (path)
    os.system(cmd)    
