from distort import *
from filters import *
import fmsynth as fm
import random
from itools import *

sr = 48000

class addSynthInst:
    def __init__(self,freqs=[1+1j]):
        self.freqs = freqs
    def __call__(self,f):
        pass

def toFreq(s,o=64,p=12):
    return 440*2**((s-o)/p)


def sawsSum(fs = [(440,1+1j)],sr=sr):
    return (sum((fm.nsaw(i*f[0]/sr)*f[1] for f in fs)) for i in fm.integ(fm.const(1)))


class sawPluck:
    def __init__(self,n=4,d=1,df=0):
        self.n=n
        self.d=d
        self.df=df
    def __call__(self,f,v=1):
        return sawsSum([((f+(random.random()-.5)*self.d)*(1+(random.random()-.5)*self.df),(1+1j)*v/self.n) for i in range(self.n)])
        

#some pythagorean chords:

#  # #  # # #  # #  # # #  # #  # # #  # #  # # #  #
# _#_#__#_#_#__#_#__#_#_#__#_#__#_#_#__#_#__#_#_#__#
# 1           2      3    4   5  6  7 8

p_M_tri = [1,5/4,3/2]
p_m_tri = [1,6/5,3/2]
p_lm_tri = [1,7/6,3/2]

toF = lambda *x: [2**(i/12) for i in x]

e_M_tri = toF(0,4,7)
e_m_tri = toF(0,3,7)

def seq(notes,tpn=.25):
    res = [[i*tpn,[notes[i]]] for i in range(len(notes))]
    p = fm.PiecewizePoly(res,res[-1][0]+tpn).integ()
    return p

    
def h(g):
    for i in g:
        yield i
    while 1:
        yield i

def toneAcc(notes,times,o=0,sr=sr):
    a = 0
    t = 0
    for i in range(len(notes)):
        d = toFreq(notes[i],o)/sr
        t += times[i]*sr
        while t > 0:
            t -= 1
            yield a
            a += d
    
#[0,3,0,-4,-9,3,0,-4,0,3,0,-4,-9,3,0,-4]
#[3,-64,-4,0,3,-64,0,-9,0,-64]
def play(sr=sr):
    r = (it(toneAcc(([0,7,0,3]*4+[-4,3,0,7,-4,0,3,7]*2)*2,[.125]*4*8*8)).l(lm(fm.ppar)*(1+1j)))



    return r









