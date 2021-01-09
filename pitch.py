#file for pitch estimation, pitch changing, other vocoding stuff


import math
import numpy as np
import scipy as sp

def vprod(v,o):
    return v.real*o.real+1j*(v.imag*o.imag)

def nsinc(x):
    if x == 0:
        return 1
    return math.sin(math.pi*x)/(math.pi*x)
def gaussian(x):
    return math.exp(-x*x)
class autocor:
    def __init__(self,d=0.01,l=128,df=2,kf=3,wl=3):
        self.i = 0
        self.d = d
        self.b = np.zeros((l,2),dtype=float)
        self.cb = np.zeros((l,2),dtype=float)
        self.kern = np.array([(lambda x: [x,x])(nsinc(i/kf-l/kf)*gaussian((i/(l-1)-.5)*wl)) for i in range(l)],dtype=float)
        self.l = l
        self.dc = 0
        self.df = df
    def __call__(self,v):
        self.chain = self.b[self.i][0]+self.b[self.i][1]*1j 
        self.cb *= 1-self.d
        va = np.array([v.real,v.imag])*self.d
        self.dc += 1
        if self.dc >= self.df:
            r = np.sum(self.b[self.i:]*self.kern[:self.l-self.i],axis=0)+\
                np.sum(self.b[:self.i]*self.kern[self.l-self.i:],axis=0)
        self.cb[:self.l-self.i] += self.b[self.i:]*va
        self.cb[self.l-self.i:] += self.b[:self.i]*va
        self.i = (self.i-1)%self.l
        self.b[self.i][0] = v.real
        self.b[self.i][1] = v.imag
        if self.dc >= self.df:
            self.dc -= self.df
            return r[0]+r[1]*1j
        
class autocor_cascade:
    def __init__(self,num=8,*cargs):
        self.cors = [autocor(*cargs) for i in range(num)]
    def __call__(self,v):
        i = 0
        while v != None and i < len(self.cors):
            v = self.cors[i](v)
            i += 1
        return v


class windower:
    def __init__(self,thing,size=256,window=lambda x:math.cos(x*math.pi)/2+1,mult=2):
        self.times=mult
        self.window=np.array([window((i/(size-1))*2-1) for i in range(size)],dtype=complex)
        self.buf=np.zeros(size,dtype=complex)
        self.windowed=np.zeros(size,dtype=complex)
        self.outbuf=np.zeros(size,dtype=complex)
        self.i=0
        self.oi=0
        self.cb = thing
    def __call__(self,v):
        #put into buf
        l = len(self.buf)
        if ((self.i*self.times)%l)+self.times >= l:
            #transform
            self.windowed[:l-self.i] = self.buf[self.i:]*self.window[:l-self.i]
            self.windowed[l-self.i:] = self.buf[:self.i]*self.window[l-self.i:]
            n = (self.i*self.times)//l

            self.outbuf[:l-self.oi] = self.outbuf[self.oi:]
            self.outbuf[l-self.oi:] = 0
            self.oi = 0
            self.outbuf += self.cb(self.windowed)
        #put into buf
        self.buf[self.i] = v
        self.i = (self.i+1)%len(self.buf)
        oi = self.oi
        self.oi += 1
        return self.outbuf[oi]
        
class ft_space:
    def __init__(self,thing,size=256,*args):
        self.windower = windower(self.cbm,size,*args)
        self.cb = thing
    def cbm(self,b):
        return sp.fft.ifft(self.cb(sp.fft.fft(b)))
    def __call__(self,v):
        return self.windower(v)
        
def split_ft(ft,fi):
    return ft[-fi:],ft[:fi],ft[fi:-fi]
def reshape_ft(ft):
    l = len(ft)
    res = np.zeros((l//2+((l^1)&1),2),dtype=complex)
    res[0,:] = ft[0]
    if l&1 == 0:
        res[-1,:] = ft[l//2]
    res[1:l//2,0] = ft[1:l//2]
    res[1:l//2,1] = ft[-1:-l//2:-1]
    return res
def unreshape_ft(ft,even=True):
    l = len(ft)*2-1-even
    res = np.zeros(l,dtype=complex)
    res[0] = (ft[0,0]+ft[0,1])/2
    if even:
        res[l//2] = (ft[-1,0]+ft[-1,1])/2
    res[1:l//2] = ft[1:-even,0]
    res[-1:-l//2:-1] = ft[1:-even,1]
    return res
    
        

        
    

    

class wavecutter:
    def __init__(self):
        #attempts to find the period of the wave and cut it there
        pass
