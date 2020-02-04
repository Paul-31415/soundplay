import math
def fir(g,taps = [1]):
    buf = [next(g) for i in taps]
    ind = 0
    while 1:
        yield sum((taps[i]*buf[(i+ind)%len(buf)] for i in range(len(taps))))
        buf[ind] = next(g)
        ind = (ind + 1)%len(buf)

def sfir(g,taps = [(0,1)]):
    offs = min((i[0] for i in taps))
    buf = [next(g) for i in range(offs+1+max((i[0] for i in taps)))]
    ind = 0
    while 1:
        yield sum((i[1]*buf[(offs+i[0]+ind)%len(buf)] for i in taps))
        buf[ind] = next(g)
        ind = (ind + 1)%len(buf)


class Integ():
    def __init__(self,integWeights = [],derivWeights=[],passweight=1,oint=[],oder=[],passo=1):
        self.integ = integWeights
        self.deriv = derivWeights
        self.ointeg = oint
        self.oderiv = oder
        self.pregain = passweight
        self.postgain = passo
    def __call__(self,g):
        ibuf = [0 for i in range(max(len(self.integ),len(self.ointeg)))]
        dbuf = [0 for i in range(max(len(self.deriv),len(self.oderiv)))]
        p = 0
        for v in g:
            r = v*self.pregain
            for i in range(len(self.integ)):
                r += ibuf[i]*self.integ[i]
            for i in range(len(self.deriv)):
                r += dbuf[i]*self.deriv[i]

            if len(self.integ):
                ibuf[0] += r
                for i in range(1,len(self.integ)):
                    ibuf[i] += ibuf[i-1]
            if len(self.deriv):
                n = dbuf[0]
                dbuf[0] = r-p
                for i in range(1,len(self.deriv)):
                    n,dbuf[i] = dbuf[i],dbuf[i]-n

            o = r*self.postgain
            for i in range(len(self.ointeg)):
                o += ibuf[i]*self.ointeg[i]
            for i in range(len(self.oderiv)):
                o += dbuf[i]*self.oderiv[i]

            yield o
                
            iml = max(len(self.integ),len(self.ointeg))
            if len(ibuf)<iml:
                ibuf += [0 for i in range(iml-len(ibuf))]
            dml = max(len(self.deriv),len(self.oderiv))
            if len(dbuf)<dml:
                dbuf += [0 for i in range(dml-len(dbuf))]
            p = r
        
    
        

class IIR():
    def __init__(self,iirtaps=[],firtaps=[1]):
        self.iirtaps = iirtaps
        self.firtaps = firtaps
    def __call__(self,g):
        buf = [0 for i in range(max(len(self.iirtaps)+1,len(self.firtaps)))]
        #lower in buf is older
        ind = 0
        for v in g:
            #                     b0
            # -- + -------- . --> * -> + ---> o        buf[ind]
            #    ^          |          ^           
            #    |   a1     V     b1   | 
            #    + <- * <- z-1 -> * -> +               buf[ind-1]
            #    ^          |          ^
            #    |   a2     V     b2   |
            #    + <- * <- z-1 -> * -> +               buf[ind-2]
            #              ...                    ...  buf[ind+1]
            # so a[n] = self.iirtaps[n-1]
            #    b[n] = self.firtaps[n]
            buf[ind] = v+sum((buf[(ind-i-1)%len(buf)]*self.iirtaps[i] for i in range(len(self.iirtaps))))
            yield sum((buf[(ind-i)%len(buf)]*self.firtaps[i] for i in range(len(self.firtaps))))
            if len(buf)<max(len(self.iirtaps)+1,len(self.firtaps)):
                #adjust buffer length to fit
                buf = buf[:ind]+[0 for i in range(max(len(self.iirtaps)+1,len(self.firtaps))-len(buf))]+buf[ind:]
            ind = (ind + 1)%len(buf)
    def times(self,other):
        #return new filter self times other
        #z domain is ∑(b[n]*z^-n) / (1-∑(a[n]*z^-n))
        #assuming I just have to do polynomial multiplication...
        resI = [0 for i in range(len(self.iirtaps)+len(other.iirtaps)+1)]
        resF = [0 for i in range(len(self.firtaps)+len(other.firtaps)-1)]
        #perform multiply
        for i in range(len(self.iirtaps)+1):
            for j in range(len(other.iirtaps)+1):
                resI[i+j] += -(-([-1]+self.iirtaps)[i]*-([-1]+other.iirtaps)[j])
        for i in range(len(self.firtaps)):
            for j in range(len(other.firtaps)):
                resF[i+j] += self.firtaps[i]*other.firtaps[j]
        resI = resI[1:]
        return IIR(resI,resF)

    def set(self,other):
        self.iirtaps = other.iirtaps
        self.firtaps = other.firtaps
        return self
    
        
            
def iir(g,taps = [0]):
    buf = [0 for i in taps]
    ind = 0
    while 1:
        buf[ind] = next(g)
        buf[ind] = sum((taps[i]*buf[(i+ind)%len(buf)] for i in range(len(buf))))
        yield buf[ind]
        ind = (ind + 1)%len(buf)


        
def siir(g,taps = [(0,1)]):
    offs = min((i[0] for i in taps))
    buf = [0 for i in range(offs+1+max((i[0] for i in taps)))]
    ind = 0
    while 1:
        buf[ind] = next(g)
        buf[ind] = sum((i[1]*buf[(offs+i[0]+ind)%len(buf)] for i in taps))
        yield buf[ind]
        ind = (ind + 1)%len(buf)

def prerun(g,n):
    for i in n:
        next(g)
    while 1:
        yield next(g)


def pulse(f,d=.1):
    p = 48000/f
    i = 0
    while 1:
        yield i<d*p
        i = (i+1)%p

def pulseGate(g,f,d=.1):
    p = 48000/f
    i = 0
    while 1:
        if i<d*p:
            yield next(g)
        else:
            yield 0
        i = (i+1)%p



import random
def noise(r=1):
    while 1:
        v = random.random()
        for i in range(r):
            yield v
            

        
def a_o(f,m=1):
    return siir(f,[(0,.1),(int(48000/f/3),.6),(int(48000/f/4/5),.2),(int(48000/f/4/7))])

def aah(g,f):
    return siir(g,[(0,.1),(int(48000/f/3),.6),(int(48000/f/4/5),.2),(int(48000/f/4/7),.1)])

vowels = {"aah":[[0,.1],[1/3,.6],[1/4/5,.2],[1/4/7,.1]]}

def vowel(f,which="aah"):
    return siir(pulse(f),[(int(48000/f*i[0]),i[1]) for i in vowels[which]])

m = [0, 0, 12, -48, 7, -48, -48, 6, -48, 5, -48, 3, -48, 0, 3, 5, -2, -2, 12, -48, 7, -48, -48, 6, -48, 5, -48, 3, -48, 0, 3, 5, -3, -3, 12, -48, 7, -48, -48, 6, -48, 5, -48, 3, -48, 0, 3, 5, -4, -4, 12, -48, 7, -48, -48, 6, -48, 5, -48, 3, -48, 0, 3, 5]+\
    [-12, 0, 12, -12, 7, -12, -12, 6, -12, 5, -12, 3, -12, 0, 3, 5, -14, -2, 12, -14, 7, -14, -14, 6, -14, 5, -14, 3, -14, 0, 3, 5, -15, -3, 12, -15, 7, -15, -15, 6, -15, 5, -15, 3, -15, 0, 3, 5, -16, -4, 12, -16, 7, -16, -16, 6, -14, 5, -14, 3, -14, 0, 3, 5]+\
    [0, 0, 12, -12, 7, -12, -12, 6, -12, 5, -12, 3, -12, 0, 3, 5, -2, -2, 12, -14, 7, -14, -14, 6, -14, 5, -14, 3, -14, 0, 3, 5, -3, -3, 12, -15, 7, -15, -15, 6, -15, 5, -15, 3, -15, 0, 3, 5, -4, -4, 12, -16, 7, -16, -16, 6, -14, 5, -14, 3, -14, 0, 3, 5]*2+\
    [3  ,-12,  3,  3,-12,  3,  0,  3,-12,  0,  0,  0,-12,-12,  0,  0, 3 ,-14,  3,  3,-14,  5, -2,  6,-14,  5,  3,  0,  3,  5, -2, -2, 3 ,-15,  3,  3,-15,  5, -3,  6,-15,  7, -3, 10,-15,  7, -3, -3, 12,-16, 12,-16, 12,  7, 12, 10,-15,-15, -3,-15, 10,-15, -3, -3]+\
    [7  ,-12,  7,  7,-12,  7,  0,  7,-12,  5,  0,  5,-12,-12,  0,  0, 7 ,-14,  7,  7,-14,  7, -2,  6,-14,  7, -2, 12,-15,  7,  6, -2, 12,  6,  7,  2,  6,  0,  3,  0, 10,  0,  6,  0,  4, -6,  2, -6, -4,-16, -2,  0,-16,  3, -4, 10,-15,-15, -3,-15,  2,-15, -3, -3] 

def limitLen(g,l):
    for i in range(l):
        yield next(g)
    while 1:
        yield 0
        
def beepg(p,w):
    return limitLen(Osc(p),int(w*48000/p))

def pulse(f,d=.1):
    p = 48000/f
    i = 0
    while 1:
        yield i<d*p
        i = (i+1)%p

def stitch(gens,times,rate=48000):
    gg = (g for g in gens)
    for t in times:
        g = next(gg)
        for i in range(int(rate*t)):
            yield next(g)
    while 1:
        yield 0

def play(w,s=.1,note=beepg):
    gens = []
    times = []
    for i in m:
        gens.append(note(440*2**(i/12),w))
        times.append(s)
    return stitch(gens,times)
        
def stitch_de(gens,times,rate=48000):
    gg = (g for g in gens)
    for t in times:
        g = next(gg)()
        for i in range(int(rate*t)):
            yield next(g)
    while 1:
        yield 0
    
def play_de(w,s=.1,note=beepg):
    gens = []
    times = []
    for i in m:
        def f(n=i):
            return note(440*2**(n/12),w)
        gens.append(f)
        times.append(s)
    return stitch_de(gens,times)


def clamp(g):
    while 1:
        v = next(g)
        yield [v,(v>0)*2-1][abs(v)>1]
                


def sinKern(f,a = 1,sr = 48000):
    fact = math.e**(2j*math.pi*f/48000)
    while 1:
        yield a
        a *= fact

def iconv(f,k,alpha = .9):
    a = 0
    for i in f:
        kv = next(k)
        a = a*alpha + (1-alpha)*i*kv
        yield (a,kv)



        
def cb(g,f=lambda :None):
    f()
    for i in g:
        yield i
        f()
