#quick import stuff

import math
from squareThings import *
from sampleGen import *
def saveM(gen,t,name = "save.aiff"):
    saveSample(monoChannelWrapper(extent(gen,t)),name,1,1)

print("run(cycle([0,1]),fof(rand24(),lambda x: int(math.log(256/(x+1),2))))")
def cycle(l):
    while 1:
        for i in l:
            yield i
def runNoise(b=2,e=1):
    b = makeGen(b)
    e = makeGen(e)
    return run(cycle([0,1]),fof(rand24(),lambda x: int(1+math.log(256/(x+1),next(b))**next(e))))
