import fileinput
from numpy import *
def entropy(*args):
    xy = zip(*args)
    # probs
    proba = [ float(xy.count(c)) / len(xy) for c in dict.fromkeys(list(xy)) ]
    entropy= - sum([ p * log2(p) for p in proba ])
    return entropy


data=list()
for line in fileinput.input():
	data.append(line)

ent=entropy(data)
print ent
