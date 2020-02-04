from sampleGen import *
from squareThings import *


def shift(g,d):
    for i in g:
        yield i+d
def step(g,d):
    for i in g:
        d -= 1;
        if d <= 0:
            break
    for i in g:
        yield i

def fof(g,l = lambda x: x):
    for i in g:
        yield l(i)
        
def prod(g1,g2):
    for i in g1:
        yield i*next(g2)

def add(g1,g2):
    for i in g1:
        yield i+next(g2)

def integrate(g,k=1,d = 0):
    v = 0
    for i in g:
        v += k*i - d*v
        yield v

def derivitive(g):
    prev = 0
    for i in g:
        yield i-prev
        prev = i



        
def count(rate=1,sps=48000):
    t = 0
    rate /= sps
    while 1:
        yield t
        t += rate
        
path = "chiptuneSamples/"

import math
#untzs
for i in range(24):
    f = 440*(2**(i/12-2))
    saveSample(
        monoChannelWrapper(
            extent(
                prod(
                    fof(
                        exponential(1,.9999)
                        ,lambda x: math.sin(x*math.pi*2/4.8*f)
                    ),
                    exponential(1,.9999)
                ),
                1
            )
        ),path+"untz"+str(i)+".aiff",1,1,48000)

#wooshes
#saveSample(monoChannelWrapper(comparator(shift(comparator(rand16(),shift(exponential(258,.99996),-1)),.1),comparator(resample(rand24(),0.998,1),rand24()))),path+"woosh.aiff",1,1,48000)


#cymbal things
for j in range(6):
    l = 1<<j
    for i in range(12):
        f = 2**(i/3)
        out = open(path+["hihat_","hihat_long_","cymbal_","cymbal_long_","gong_base_","gong_base_long_"][j]+str(i)+".aiff","w+b")
        saveSample(monoChannelWrapper(extent(comparator(shift(exponential(255,1-.001*f/l),-1),rand24()),0.25*l,48000/f)),out,1,1,48000/f)


#drums



