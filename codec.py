from random import random
from bisect import bisect_left
class codec:
    def __init__(self,*a):
        self.update(*a)
    def update(self,valids,res=16,low=-1,high=1):
        try:
            self.vals = sorted(valids)
        except:
            self.vals = sorted([valids(i/(res-1)*(high-low)+low) for i in range(res)])
        self.splits = [(self.vals[i]+self.vals[i+1])/2 for i in range(len(self.vals)-1)]
        self.difs = [0]+[self.vals[i+1]-self.vals[i] for i in range(len(self.vals)-1)]+[0]
    def __call__(self,v,dith=None):
        if type(v) is complex:
            return self(v.real,dith if dith is None else dith.real)+1j*self(v.imag,dith if dith is None else dith.imag)
        if dith is None:
            return self.vals[bisect_left(self.splits,v)]
        i = bisect_left(self.splits,v)
        dith *= self.difs[i+(dith>0)]
        return self.vals[bisect_left(self.splits,v+dith.real)]

def zeros():
    while 1:
        yield 0
    
class error_prop:
    def __init__(self,filt,efilt = lambda e:e,dith=zeros()):
        self.error = 0
        self.ef = efilt
        self.f = filt
        self.dith = dith
    def __call__(self,v):
        r = self.f(v+self.error+next(self.dith))
        self.error = self.ef(self.error + v - r)
        return r,self.error
    
class dith_error_prop:
    def __init__(self,filt,efilt = lambda e,n:n,dith=zeros()):
        self.error = 0
        self.ef = efilt
        self.f = filt
        self.dith = dith
    def __call__(self,v):
        r = self.f(v,self.error+next(self.dith))
        self.error = self.ef(self.error,v - r)
        return r,self.error
    

#some examples
# LOUD BOi
# cod = codec([-1,1])
# fl = filt.iir1l(-.9,.01)
# f = error_prop(lambda x:cod(fl(x*(abs(x)**0.01))),filt.iir1l(.9,.1))
# mix.out = (f(v)[0] for v in gen)
#
# another is
# fl = filt.iir1l(-.93,.5)
# f = error_prop(lambda x:cod(fl(x*(abs(x)**6))),filt.iir1l(-.99,.01))
