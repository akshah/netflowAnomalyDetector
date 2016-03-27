#!/usr/bin/python 

import sys
import numpy as np

ranks_map={}
for line in sys.stdin:
	line = line.strip()
	rank,node=line.split("\t")
        if node not in ranks_map:
            ranks_map[node]=rank
        else:
            new_rank=float(ranks_map[node])+float(rank)
            ranks_map[node]=new_rank

for node in ranks_map:
    print '%s %s' % (ranks_map[node],node)

