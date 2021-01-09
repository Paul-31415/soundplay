import numpy as np

import struct

def float32_dist(a,b):
    ia,ib = struct.unpack(">ii",np.array([a,b],dtype=np.float32).tobytes())
    return ia-ib

class PDist:
    def __init__(self,f,d,n="probability distribution"):
        self.f = f
        self.d = d
        self.name = n
    def __call__(self,*a):
        return self.f(*a)
    def gradient(self,valx,vald,fv=None):
        try:
            return self.d(valx,vald,fv)
        except:
            return self.d(valx,vald,self.f(valx,vald))
cdf = lambda x,d: (1+np.tanh(x/d))/2
pdf = lambda x,d: (1-np.tanh(x/d)**2)/2/d
grad = lambda x,d,p: np.array([-np.tanh(x/d)*d*p/d/d, (lambda f: (2*x*f*d*p+d*f*f-d)/2/d/d/d)(np.tanh(x/d))])    
dtanh = PDist(pdf,grad,"dtanh")

grad = lambda x,d,p: np.array([-x/d, (lambda f: (2*x*f*d*p+d*f*f-d)/2/d/d/d)(np.tanh(x/d))])
dtanh_fake = PDist(pdf,grad,"dtanh_fake")
grad = lambda x,d,p: np.array([-x/d, x*x-1/d])
dtanh_fake2 = PDist(pdf,grad,"dtanh_fake2")

cdf = lambda x,d: (1+np.tanh(x*np.exp(-d))/2)
pdf = lambda x,d: (lambda ed:(1-np.tanh(x*ed)**2)*ed/2)(np.exp(-d))
grad = lambda x,d,p: (lambda ed:np.array([-ed*ed*np.tanh(x*ed)*p, (lambda f: ed*ed*x*f*p+f*f*ed/2-ed/2)(np.tanh(x*ed))]))(np.exp(-d))
dtanhe = PDist(pdf,grad,"dtanhe")

grad = lambda x,d,p: (lambda ed:np.array([-x*ed, (lambda f: ed*ed*x*f*p+f*f*ed/2-ed/2)(np.tanh(x*ed))]))(np.exp(-d))
dtanhe_fake = PDist(pdf,grad,"dtanhe_fake")

def entropy(val,guess,deviation,dist=dtanhe,base=1/(1<<64)):
    prob = dist(val-guess,deviation+base)
    #gradient of entropy = ∂dist
    return prob*(1-base)+base,(1-base)*dist.gradient(val-guess,deviation+base,prob)
