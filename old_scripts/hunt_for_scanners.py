#!/usr/bin/env python

import sys
import os
import numpy as np
for line in sys.stdin:
	line = line.strip()	
	cmd="cat keyvalue_archive/2014/01/17/argus.* | grep %s | awk -F',' '{print$4}' | python scripts/entropy.py" % (line)
	os.system(cmd)

