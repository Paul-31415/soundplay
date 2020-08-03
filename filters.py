import math
eone = math.exp(2*math.pi)

import brailleG as gr


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
    def __init__(self,iir=[1],fir=[1],pos=0):
        self.pos = pos
        self.iir = iir
        self.fir = fir
    def incPos(self):
        self.iir = [0] + self.iir
        self.fir = [0] + self.fir
        self.pos += 1
    def __call__(self,g):
        buf = [0 for i in range(max(len(self.iir),len(self.fir)))]
        p = 0
        for v in g:
            p = buf[self.pos]
            buf[self.pos] = v
            buf[self.pos] = sum((buf[i]*self.iir[i] for i in range(len(self.iir))))
            for i in range(self.pos-1,-1,-1):#derivitive chain
                p,buf[i] = buf[i],buf[i+1]-p
            for i in range(self.pos+1,len(buf)):#integral chain
                buf[i] += buf[i-1]
            yield sum((buf[i]*self.fir[i] for i in range(len(self.fir))))
            ml = max(len(self.iir),len(self.fir))
            if len(buf)<ml:
                buf += [0 for i in range(ml-len(buf))]
    
def polymult(p1,p2):
    if len(p1) == 0 or len(p2) == 0:
        return []
    res = [0 for i in range(len(p1)+len(p2)-1)]
    for i in range(len(p1)):
        for j in range(len(p2)):
            res[i+j] += p1[i]*p2[j]
    return res

def polyadd(p1,p2):
    res = [0 for i in range(max(len(p1),len(p2)))]
    for i in range(len(p1)):
        res[i] += p1[i]
    for j in range(len(p2)):
        res[j] += p2[j]
    return res

def polyconj(p1):
    return [p.conjugate() for p in p1]

class Polynomial:
    def __init__(self,c,doTrim=True,doCopy=True):
        if doCopy:
            try:
                self.constants = list(c)
            except TypeError:
                self.constants = [c]
        else:
            self.constants = c
        if doTrim:
            self.trim()
    def termOp(self,o,op=lambda a,b: a+b,ext=0):
        if type(o) == Polynomial:
            a = self.constants+[ext]*max(0,len(o)-len(self))
            b = o.constants+[ext]*max(0,len(self)-len(o))
            return Polynomial([op(a[i],b[i]) for i in range(len(a))])
        else:
            return self.termOp(Polynomial(o),op,ext)
    def __len__(self):
        return len(self.constants)
    def __getitem__(self,i):
        return self.constants.__getitem__(i)
    def __setitem__(self,i,v):
        self.constants.__setitem__(i,v)
    def __iter__(self):
        return self.constants.__iter__()
    def order(self):
        return len(self)-1
    def trim(self):
        for i in range(len(self)):
            if self.constants[-i-1] != 0:
                break
        else:
            self.constants = self.constants[:0]
            return
        self.constants = self.constants[:len(self)-i]
    def trimmed(self):
        return Polynomial(self,True)
    def __coerce__(self,o):
        return (self,Polynomial(o))
    def __add__(self,o):
        return self.termOp(o,lambda a,b: a+b)
    def __radd__(self,o):
        return self.termOp(o,lambda a,b: b+a)
    def __sub__(self,o):
        return self.termOp(o,lambda a,b: a-b)
    def __rsub__(self,o):
        return self.termOp(o,lambda a,b: b-a)
    def vtimes(self,o,e=1):
        return self.termOp(o,lambda a,b: a*b,e)
    def vdiv(self,o,e=1):
        return self.termOp(o,lambda a,b: a/b,e)
    def __neg__(self):
        return Polynomial([-e for e in self.constants])
    def conjugate(self):
        return Polynomial([e.conjugate() for e in self.constants])
    def real(self):
        return Polynomial([e.real for e in self.constants])
    def imag(self):
        return Polynomial([e.imag for e in self.constants])
    def __mul__(self,o):
        return Polynomial(polymult(self.constants,o.constants))
    def __divmod__(self,o):
        if type(o) != Polynomial:
            o = Polynomial(o)
        #polynomial division self/o
        # self   1   + x   + x^2 + x^3
        #                    1   + x shift:2 = 4-2
        o = o.trimmed()
        if o.order()<0:
            raise ZeroDivisionError()
        shift = self.order()-o.order()
        rem = Polynomial(self,False,True).constants
        res = [0] * max(0,shift+1)
        while shift>=0:
            f = rem.pop()/o[o.order()]
            res[shift] = f
            for i in range(o.order()):
                rem[-i-1] -= f*o[-i-2]
            shift -= 1
        return Polynomial(res,True,False),Polynomial(rem,True,False)
    def __div__(self,o):
        return divmod(self,o)[0]
    def __floordiv__(self,o):
        return divmod(self,o)[0]
    def __mod__(self,o):
        return divmod(self,o)[1]
    def __lshift__(self,n):
        return Polynomial([0]*n+self.constants)
    def __rshift__(self,n):
        return Polynomial(self.constants[n:])
    def __call__(self,x):
        r = 0
        t = 1
        for c in self.constants:
            r += t*c
            t *= x
        return r
    def __repr__(self):
        return "Polynomial("+repr(self.constants)+")"
    def __str__(self,var='x'):
        res = ""
        term = ""
        exp = ""
        ei = 2
        for c in self.constants:
            res += str(c) + term+exp + " + "
            if term == "":
                term = var
            else:
                exp = "^"+str(ei)
                ei += 1
        return res[:-3]
            
class IIR():
    def __init__(self,iirtaps=[1],firtaps=[1]):
        self.iirtaps = iirtaps
        self.firtaps = firtaps
    def __repr__(self):
        return "IIR("+repr(self.iirtaps)+","+repr(self.firtaps)+")"
    def __call__(self,g,mode = 0,chunk=480):
        buf = [0 for i in range(max(len(self.iirtaps),len(self.firtaps)))]
        #lower in buf is older
        ind = 0
        if mode == 0:
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
                
                buf[ind] = v*self.iirtaps[0]+sum((buf[(ind-i)%len(buf)]*self.iirtaps[i] for i in range(1,len(self.iirtaps))))
                yield sum((buf[(ind-i)%len(buf)]*self.firtaps[i] for i in range(len(self.firtaps))))
                if len(buf)<max(len(self.iirtaps),len(self.firtaps)):
                    #adjust buffer length to fit
                    buf = buf[:ind]+[0 for i in range(max(len(self.iirtaps),len(self.firtaps))-len(buf))]+buf[ind:]
                ind = (ind + 1)%len(buf)
        elif mode == 1:
            for v in g:
                r = 0
                f = v*self.iirtaps[0]
                buf[ind] = f
                for i in range(len(buf)):
                    if 0 < (i - ind)%len(buf) < len(self.iirtaps):
                        f -= buf[i]*self.iirtaps[(i-ind)%len(buf)]
                    if (i - ind)%len(buf) < len(self.firtaps):
                        r += buf[i]*self.firtaps[(i-ind)%len(buf)]
                yield r
                if len(buf)<max(len(self.iirtaps),len(self.firtaps)):
                    #adjust buffer length to fit
                    buf = buf[:ind]+[0 for i in range(max(len(self.iirtaps),len(self.firtaps))-len(buf))]+buf[ind:]
                buf[ind] = f
                ind = (ind + 1)%len(buf)
        elif mode == 2:
            import scipy.signal as ss
            dbuf = [0 for i in range(chunk)]
            buf = buf[:-1]
            i = 0
            res = []
            for v in g:
                dbuf[i] = v
                i += 1
                if i == len(dbuf):
                    i = 0
                    if len(buf)!=max(len(self.iirtaps),len(self.firtaps))-1:
                        buf = ([i for i in buf]+[0 for i in range(max(0,max(len(self.iirtaps),len(self.firtaps))-1-len(buf)))])[:max(len(self.iirtaps),len(self.firtaps))-1]
                    res, buf = ss.lfilter(self.firtaps,self.iirtaps,dbuf,-1,buf)
                if len(res):
                    yield res[i]
            dbuf = dbuf[:i+1]
            if len(buf)!=max(len(self.iirtaps),len(self.firtaps))-1:
                buf = ([i for i in buf]+[0 for i in range(max(0,max(len(self.iirtaps),len(self.firtaps))-1-len(buf)))])[:max(len(self.iirtaps),len(self.firtaps))-1]

            res, buf = ss.lfilter(self.firtaps,self.iirtaps,dbuf,-1,buf)
            for v in res:
                yield v 
    def times(self,other):
        #return new filter self times other
        #z domain is ∑(b[n]*z^-n) / (1-∑(a[n]*z^-n))
        #assuming I just have to do polynomial multiplication...
        resI = [0 for i in range(len(self.iirtaps)+len(other.iirtaps)+1)]
        resF = [0 for i in range(len(self.firtaps)+len(other.firtaps)-1)]
        #perform multiply
        for i in range(len(self.iirtaps)+1):
            for j in range(len(other.iirtaps)+1):
                resI[i+j] += self.iirtaps[i]*other.iirtaps[j]
        for i in range(len(self.firtaps)):
            for j in range(len(other.firtaps)):
                resF[i+j] += self.firtaps[i]*other.firtaps[j]
        resI = resI[1:]
        return IIR(resI,resF)
    def reciprocal(self):
        return IIR(self.firtaps,self.iirtaps)

    def add(self,other):
        #z-space fraction add
        n1 = self.getBPoly()
        n2 = other.getBPoly()

        d1 = self.getAPoly()
        d2 = other.getAPoly()
        res = IIR()
        n = polyadd(polymult(n1,d2),polymult(n2,d1))
        d = polymult(d1,d2)
        res.setPolys(n,d)
        return res
    
    
    def set(self,other):
        self.iirtaps = other.iirtaps
        self.firtaps = other.firtaps
        return self

    def getBPoly(self):
        return self.firtaps
    def getAPoly(self):
        return self.iirtaps #[1] + [-i for i in self.iirtaps]
    
    def setBPoly(self,p):
        self.firtaps = p
    def setAPoly(self,p):
        self.iirtaps = p
        #d = p[0]
        #for i in range(len(self.firtaps)):
        #    self.firtaps[i] /= d
        #self.iirtaps = [-i/d for i in p[1:]]
    def setPolys(self,n,d):
        self.setBPoly(n)
        self.setAPoly(d)
    def getGain(self,freq,sr=48000):
        return self.evalPoly(eone**(-1j*freq/sr))
    def evalPoly(self,f):
        n = 0
        d = 0
        fe = 1
        for i in self.iirtaps:
            d += fe*i
            fe *= f
        fe = 1
        for i in self.firtaps:
            n += fe*i
            fe *= f
        return n/d
    
    def graphZ(self,w=80,h=40):
        gr.graphI(lambda x,y: (abs((self.evalPoly(x+1j*y)))%1)>.5,-2,2,-2,2,w,h)
    def graphM(self,w=80,h=40):
        gr.graph(lambda x: math.log(abs(self.getGain(24000*(2**x)))),-10,0,-10,10,w,h)
    def gresponse(self,w=80,ml=-4,mh=4,fl=-10,fh=0):
        return gr.lgraph(lambda x: math.log(abs(self.getGain(24000*(2**x)))),fl,fh,ml,mh,w)
    def root(self,freq,mag=.9,sr=48000):
        #returns a polynomial with a zero at that freq
        r = eone**(1j*freq/sr)*mag
        return [r,-1]
    def rootPair(self,freq,mag=.9,sr=48000):
        return [i.real for i in polymult(self.root(freq,mag,sr),self.root(-freq,mag,sr))]

    def setFromPoleZero(self,poles=[],zeros=[]):
        a = [1]
        b = [1]
        for p in poles:
            a = polymult(a,p)
        for z in zeros:
            b = polymult(b,z)
        self.setPolys(b,a)
        
    def setFromPZPolar(self,poles=[],zeros=[],sr=48000):
        self.setFromPoleZero([self.rootPair(p[0],p[1],sr) if type(p) == type([]) else self.root(p[0],p[1],sr) for p in poles],
                             [self.rootPair(z[0],z[1],sr) if type(z) == type([]) else self.root(z[0],z[1],sr) for z in zeros])

    
            
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


def iir1(g,a1=0,b0=1,b1=0,acc=0):
    #                     b0
    # -- + -------- . --> * -> + ---> o        
    #    ^          |          ^           
    #    |   a1     V     b1   |
    #    + <- * <- z-1 -> * -> +               
    for v in g:
        r = v-a1*acc
        yield b0*r+b1*acc
        acc = r
def iir1l(a1=0,b0=1,b1=0,acc=0):
    def do(v,a=[a1],b=[b0,b1],s=[acc]):
        p = s[0]
        s[0] = v-a[0]*p
        return b[0]*s[0]+b[1]*p
    return do

import math
def iir1zpl(zero=[math.inf],pole=[math.inf],gain=[1],acc=[0]):
    def do(v,z=zero,p=pole,g=gain,s=acc):
        r = s[0]
        s[0] /= p[0]
        s[0] += v #pole means a = 1-x/pole = pole-x
        return g[0]*(s[0]-r/z[0]) #b = (1-x/zero) * gain
    return do
def iir1zprl(zero_recip=[0],pole_recip=[0],gain=[1],acc=[0]):
    def do(v,z=zero_recip,p=pole_recip,g=gain,s=acc):
        r = s[0]
        s[0] *= p[0]
        s[0] += v 
        return g[0]*(s[0]-r*z[0]) 
    return do


def iir2(g,a1=0,a2=0,b0=1,b1=0,b2=0,s1=0,s2=0):
    #                     b0
    # -- + -------- . --> * -> + ---> o        
    #    ^          |          ^           
    #    |   a1     V     b1   | 
    #    + <- * <- z-1 -> * -> +               s1
    #    ^          |          ^
    #    |   a2     V     b2   |
    #    + <- * <- z-1 -> * -> +               s2
    for v in g:
        r = v-s1*a1-s2*a2
        yield b0*r+b1*s1+b2*s2
        s2,s1 = s1,r

def iir2l(a1=0,a2=0,b0=1,b1=0,b2=0,s1=0,s2=0):
    def do(v,a=[a1,a2],b=[b0,b1,b2],s=[s1,s2]):
        p,s[1],s[0] = s[1],s[0],v-s[0]*a[0]-s[1]*a[1]
        return b[0]*s[0]+b[1]*s[1]+b[2]*p
    return do

def iir2a(g,a=[1,0,0],b=[1,0,0],s1=0,s2=0):    
    for v in g:
        r = v*a[0]-s1*a[1]-s2*a[2]
        yield b[0]*r+b[1]*s1+b[2]*s2
        s2,s1 = s1,r
def iir2al(ap=[0,0],bp=[1,0,0],st=None):
    #the purpose of this is so you can pass the arrays directly
    # so you get to change them while the filter has them
    if st == None:
        st = [0,0]
    if len(ap) == 2: #implicit 1 to start
        def do(v,a=ap,b=bp,s=st):
            p,s[1],s[0] = s[1],s[0],v-s[0]*a[0]-s[1]*a[1]
            return b[0]*s[0]+b[1]*s[1]+b[2]*p
        return do
    def do(v,a=ap,b=bp,s=st):
        p,s[1],s[0] = s[1],s[0],v*a[0]-s[0]*a[1]-s[1]*a[2]
        return b[0]*s[0]+b[1]*s[1]+b[2]*p
    return do

def iir2zpl(zero=[math.inf,math.inf],pole=[math.inf,math.inf],gain=[1],state=[0,0]):
    def do(v,z=zero,p=pole,g=gain,s=state):
        c = p[0]*p[1]
        r,s[1],s[0] = s[1],s[0],v+s[0]*(p[0]+p[1])/c-s[1]/c
        # a = (1-x/p1)(1-x/p2) = 1-x(p2+p1)/(p2p1) + x^2/(p2p1)
        c = z[0]*z[1]
        return g[0]*(s[0]-s[1]*(z[0]+z[1])/c+r/c)
    return do
def iir2zprl(zero_recip=[0,0],pole_recip=[0,0],gain=[1],state=[0,0]):
    def do(v,z=zero_recip,p=pole_recip,g=gain,s=state):
        r,s[1],s[0] = s[1],s[0],v+s[0]*(p[0]+p[1])-s[1]*p[0]*p[1] #(1-xp1)(1-xp2) = 1-x(p1+p2)+x^2(p1p2)
        return g[0]*(s[0]-s[1]*(z[0]+z[1])+r*z[0]*z[1]) 
    return do



def iir3(g,a1=0,a2=0,a3=0,b0=1,b1=0,b2=0,b3=0,s1=0,s2=0,s3=0):
    for v in g:
        r = v-s1*a1-s2*a2-s3*a3
        yield b0*r+b1*s1+b2*s2+b3*s3
        s3,s2,s1 = s1,s2,r


class IIR2:
    #second order iir
    def __init__(self,a = [1,0,0],b=[1,0,0]):
        self.a = a
        self.b = b
    def __call__(self,g):
        return iir2a(self.a,self.b)
    def sever(self):
        self.a = [i for i in self.a]
        self.b = [i for i in self.b]
    def setPole(self,f=.25,m=1.1):
        z = eone**(1j*f)*m
        #[z,-x][-z,-x]
        #[-z^2,zx-zx,+x^2]


def delayl(n=0,start=[]):
    def do(v,i=[n],d=(start+[0]*(n+1))[:n+1]):
        d[i[0]] = v
        i[0] = (i[0]+1)%len(d)
        return d[i[0]]
    return do

def feedbackl(d=lambda x:0,b=lambda x:0,p = lambda x:x,s=0):
    #     .__b________.
    #     v           |
    # --->+-v-d->[s]>-^->+---->
    #       |            ^
    #       '--p---------'
    #s is a single sample memory
    def do(v,s=[s]):
        i = v+b(s[0])
        o = s[0]+p(i)
        s[0] = d(i)
        return o
    return do

def chainl(a=lambda x:x,b=lambda x:x):
    # -->a-->b-->
    def do(v):
        return b(a(v))
    return do

def parrl(a,b,m=.5):
    #   .--a-(m)-.
    # ->|        +-->
    #   '-b-(1-m)'
    def do(v):
        return m*a(v)+(1-m)*b(v)
    return do

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
