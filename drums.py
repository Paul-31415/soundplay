import prng
rand = prng.rand()

from itools import it,lm,tseq

def shift(g,t,sr=48000):
    for i in range(int(t*sr)):
        yield 0
    for i in g:
        yield i

def ssDelay(g,d=0):
    for i in g:
        yield i#todo

        
class Dist:
    def __init__(self):
        pass
    def sample(self):
        return next(rand)
class BDist:
    def __init__(self):
        pass
    def sample(self):
        return next(rand)*2-1
class CDist:
    def __init__(self,n=0):
        self.v = n
    def sample(self):
        return self.v
    
class click:
    def __init__(self,freq=1000,sr=48000):
        self.n = sr/freq
    def __iter__(self):
        for i in range(int(self.n+next(rand))):
            yield 1
    
class particleDrum:
    def __init__(self,tdist=Dist(),vdist=BDist(),ndist=CDist(80),cb=click(),tscale=1,vscale=1,nscale=1):
        self.td = tdist
        self.vd = vdist
        self.nd = ndist
        self.cb = cb
        self.tscale = tscale
        self.vscale = vscale
        self.nscale = nscale
    def __iter__(self,sr = 48000):
        n = int(self.nd.sample()*self.nscale)
        t = [self.td.sample()*self.tscale*sr for i in range(n)]
        v = [self.vd.sample()*self.vscale for i in range(n)]
        t.sort()
        for i in tseq([it(ssDelay(iter(self.cb),t[i]%1))*v[i] for i in range(len(v))],[int(t[i]) for i in range(len(t))]):
            yield i
        

            
            


