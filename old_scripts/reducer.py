#!/usr/bin/env python

#Get feature matrix strip out key and print

import sys
import numpy as np
for line in sys.stdin:
	line = line.strip()
	key,value=line.split("\t")
	print "%s" % (value)

