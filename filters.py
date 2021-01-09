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
        if i < self.constants.__len__():
            return self.constants.__getitem__(i)
        return 0
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

#complex transform capable of matrix:
# a*x+b*x.conj
# = (ar+br)*xr+(bi-ai)*xi+ i*((ai+bi)*xr+(ar-br)*xi)
# = [ar+br bi-ai] [xr]
#   [ai+bi ar-br] [xi]
def mtrxToCT(m=[[1,0],[0,1]]):
    ar = (m[0][0]+m[1][1])/2
    br = (m[0][0]-m[1][1])/2
    ai = (m[1][0]-m[0][1])/2
    bi = (m[1][0]+m[0][1])/2
    return [ar+ai*1j,br+bi*1j]
def CTdet(*c):
    if len(c) == 1:
        a,b = c[0]
    else:
        a,b = c
    return (a.real+b.real)*(a.real-b.real)-(a.imag+b.imag)*(b.imag-a.imag)
def CTmax(*c):
    if len(c) == 1:
        a,b = c[0]
    else:
        a,b = c
    #returns max magnitude increase factor and phase
    # max_t(|a*e^(i*t)+b*e^(-i*t)|)
    # = max_t( (a*e^(i*t)+b*e^(-i*t)) * (a.c*e^(-i*t)+b.c*e^(i*t)) )
    #  max_t(a*a.c+(a*b.c*e^(2i*t)+b*a.c*e^(-2i*t))+b*b.c)
    r = a*a.conjugate()+b*b.conjugate()
    p = a*b.conjugate()# (a+ib)(c-id) =? (a-ib)(c+id)
    #q = p.conjugate() #ac+i(bc-ad)+bd  ac+i(ad-bc)+bd
    #r+p*n+(p*n).c
    #max v = r+abs(p)^2
    return abs(r)+abs(p)**2

def iirm0al(b=[1,0]):
    def do(v,b=b):
        return v*b[0]+v.conjugate()*b[1]
    return do
def iirm1al(a=[[0,0]],b=[[1,0],[0,0]],s=None):
    if s == None:
        s = [0]
    def do(v,a=a,b=b,s=s):
        v,s[0] = s[0],v-a[0][0]*s[0]-a[0][1]*s[0].conjugate()
        return b[0][0]*s[0]+b[0][1]*s[0].conjugate()+b[1][0]*v+b[1][1]*v.conjugate()
    return do
def iirm2al(a=[[0,0],[0,0]],b=[[1,0],[0,0],[0,0]],s=None):
    if s == None:
        s = [0,0]
    def do(v,a=a,b=b,s=s):
        v,s[1],s[0] = s[1],s[0],v-a[0][0]*s[0]-a[0][1]*s[0].conjugate()-a[1][0]*s[1]-a[1][1]*s[1].conjugate()
        return b[0][0]*s[0]+b[0][1]*s[0].conjugate()+b[1][0]*s[1]+b[1][1]*s[1].conjugate()+b[2][0]*v+b[2][1]*v.conjugate()
    return do



def biquadPeak(f,w,g):
    #freq, width, gain
    #width is the distance of the pole to 1
    #g = (1-abs(z))/(1-abs(p))
    z = (1-w*g)
    p = (1+w)
    #(z*e^(f2πj)-1x) has zero at z
    #z^2, -2zcos, 1
    t = -2*math.cos(f*2*math.pi)
    b = [z*z,z*t,1]
    a = [t/p,1/p/p]#
    return iir2l(*a,*b)
def iir2a(g,a=[1,0,0],b=[1,0,0],s1=0,s2=0):    
    for v in g:
        r = v/a[0]-s1*a[1]-s2*a[2]
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

def polar(mag,freq,sr=48000):
    theta = freq/sr
    return mag*eone**(1j*theta)

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


def iiral(a=[1],b=[1],s=None):
    if s == None:
        s = [0]*max(len(a),len(b))
    def do(v,i=[0],a=a,b=b,s=s):
        #input
        r = v*a[0]
        for o in range(1,len(a)):
            r -= a[o]*s[i[0]-o]
        r /= a[0]
        s[i[0]] = r
        r *= b[0]
        for o in range(1,len(b)):
            r += b[o]*s[i[0]-o]
        i[0] = (i[0]+1)%len(s)
        return r
    return do
        
        
#use show(block=False)
def iirIA(l=3,plotN = 1<<8,ms=.5):#interactive iir filter arrays
    A = [1]+[0]*(l-1)
    B = [1]+[0]*(l-1)
    
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotThings import DraggablePoint
    from matplotlib.widgets import CheckButtons
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set(xlim=(0,5),ylim=(-2.5,2.5))
    plt.subplots_adjust(left=0.2)
    drs = []
    circles = [patches.Circle((5, -1), 0.08, fc='xkcd:blue', alpha=0.75),
               patches.Circle((0, 1), 0.08, fc='xkcd:red', alpha=0.75)]+\
               [i for j in range(l-1) for i in (
                   patches.Circle((10*((j+2)>>1)/l, 2.375*((j&1)*2-1)), 0.05, fc='xkcd:sky blue', alpha=0.75),
                   patches.Circle((10*((j+2)>>1)/l, 0), 0.05, fc='xkcd:light red', alpha=0.75))]
                                                
    for circ in circles:
        ax.add_patch(circ)
        dr = DraggablePoint(circ)
        dr.dat = len(drs)
        drs.append(dr)
        dr.connect()
    refs = [drs]

    cbs = None
    
    s = .5/plotN
    x = [i*s for i in range(1,plotN)]
    def phs(r,xes):
        xes = iter(xes)
        c = 0
        ph = (arg(r(next(xes)))/math.pi)
        yield ph
        for i in xes:
            phn = (arg(r(i))/math.pi)+c
            while abs(ph-phn)>.5:
                a = 1-2*(phn > ph)
                c += a
                phn += a
            yield phn
            ph = phn

    x = [.5*i/plotN for i in range(-plotN,plotN)]
    mdat, = plt.plot([abs(10*i) for i in x], [-1]*plotN+[1]*plotN,"o", ms=ms)
    ppdat, = plt.plot([abs(10*i) for i in x[plotN:]], [0]*plotN,"o", ms=ms)
    pndat, = plt.plot([abs(10*i) for i in x[:plotN]], [0]*plotN,"o", ms=ms)
    r = response(B,A)
    def fmap(x):
        if x<0:
            return -fmap(-x)
        return (x/10)*math.exp(7*(x-5)/5) if cbs.get_status()[1] else x/10
    def drawR():
        mdat.set_ydata([abs(r(fmap(c*10)))*(2*(c>=0)-1) for c in x])
        ppdat.set_ydata([i/2 for i in phs(lambda f:r(fmap(f*10)),x[plotN:])])
        pndat.set_ydata([i/2 for i in phs(lambda f:r(fmap(f*10)),x[plotN-1::-1])][::-1])
        fig.canvas.draw_idle()
    def cmap(x,y):
        return abs(y)*eone**(fmap(x)*1j*(2*(y>0)-1))
    def nrm(a):
        f = math.sqrt(abs(a[0])**2+abs(a[1])**2)
        return [a[0]/f,a[1]/f]
    def amap(x,y):
        g = lambda x: 3*(x+x**11)
        return nrm([1,-cmap(x,(2*(y<0)-1)*math.tanh(g(max(abs(y*2)-.125,0)/(2.5-.25))))])
    def bmap(x,y):
        return nrm([1,-cmap(x,max(0,math.tanh(min(4,4*(1.125-abs(y/2))))/math.tanh(4))*(2*(y<0)-1))])
    def calcTerms():
        a = Polynomial([cmap(*drs[1].point.center)])
        b = Polynomial([cmap(*drs[0].point.center)])
        for i in range(1,l):
            b *= Polynomial(bmap(*drs[i*2].point.center))
            a *= Polynomial(amap(*drs[i*2+1].point.center))
        for i in range(l):
            A[i] = a[i]
            B[i] = b[i]

    rax = plt.axes([0.05, 0.4, 0.1, 0.15])
    cbs = CheckButtons(rax, ["pair","logf","quad"], [True,False,False])
    
    def move(dr):
        if dr.dat > 1:
            if cbs.get_status()[0]:
                oi = ((dr.dat-2) ^ 2)+2
                if oi < len(drs):
                    drs[oi].move(dr.pos()[0],-dr.pos()[1])
                    if cbs.get_status()[2]:
                        ooi = ((oi-2) ^ 1)+2
                        if ooi < len(drs):
                            drs[ooi].move(drs[oi].pos()[0],drs[ooi].pos()[1])
            if cbs.get_status()[2]:
                oi = ((dr.dat-2) ^ 1)+2
                if oi < len(drs):
                    drs[oi].move(dr.pos()[0],drs[oi].pos()[1])
        calcTerms()
        drawR()
    for dr in drs:
        dr.hooks[1]=move
    """dat, = plt.plot(x, [1]*len(x), lw=ms)
    l, = plt.plot([0,.5], [1,1],"k--", lw=ms)

    one, = plt.plot([0,.5], [(L+1)/2]*2,"k-", lw=ms)
    plt.grid(color='xkcd:grey',linestyle='-')
    plt.axis([0,.5,0,L+1])
    plt.subplots_adjust(left=0.25, bottom=0.25)
    
    axdly = plt.axes([0.25, 0.1, 0.65, 0.03])
    axA = plt.axes([0.25, 0.15, 0.65, 0.03])
    dly = Slider(axdly, 'Delay', 0, 1, valinit=1/(L+1))
    alph = Slider(axA, 'Alpha', 0, 1, valinit=a)
    def upd(val):
        a = alph.val
        d = (dly.val*(La[0]+1))
        l.set_ydata([d,d])
        r = response(delayFIRterms(d,La[0],a))
        p = [p for p in phs(r,x)]
        dat.set_ydata([1+p[i]/x[i] for i in range(len(x))])

        one.set_ydata([(La[0]+1)/2]*2)
        mdat.set_ydata([abs(r(i))*(La[0]+1)/2 for i in x])
        
        fig.canvas.draw_idle()
    dly.on_changed(upd)
    alph.on_changed(upd)

    p_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
    m_ax = plt.axes([0.25, 0.025, 0.1, 0.04])
    incL = Button(p_ax, '+')
    decL = Button(m_ax, '-')
    def ul(La=La):
        ax.axis([0,.5,0,La[0]+1])
    def il(ev,La=La):
        La[0] += 1
        ul()
        upd(0)
    incL.on_clicked(il)
    def dl(ev,La=La):
        La[0] = max(La[0]-1,1)
        ul()
        upd(0)
    decL.on_clicked(dl)
    """
    return plt,A,B,refs


def firl(k=[1]):
    def f(v,i=[0],k=k,s=[0]*(len(k)-1)):
        r = v*k[0]
        for o in range(1,len(k)):
            r += s[(i[0]-o)%len(s)]*k[o]
        s[i[0]] = v
        i[0] = (i[0] + 1)%len(s)
        return r
    return f
def lpptl(g=1,n=2):
    if n == 1:
        def lp(v):
            return v*g
    elif n == 2:
        def lp(v,p=[0,g/2]):
            v,p[0] = (v+p[0])*p[1],v
            return v
    elif n == 3:
        def lp(v,p=[0,0,g/4]):
            v,p[0],p[1] = (v+2*p[0]+p[1])*p[2],v,p[0]
            return v
    else:
        k = Polynomial([.5,.5])
        t = Polynomial([1])
        b = n.bit_length()
        for i in range(b-1,0-1,-1):
            t = t*t
            if (1<<i)&n:
                t = t*k
        return firl(t*Polynomial([g]))
    return lp
def hpptl(g=1,n=2):
    if n == 1:
        def hp(v):
            return v*g
    elif n == 2:
        def hp(v,p=[0,g/2]):
            v,p[0] = (v-p[0])*p[1],v
            return v
    elif n == 3:
        def hp(v,p=[0,0,g/4]):
            v,p[0],p[1] = (v-2*p[0]+p[1])*p[2],v,p[0]
            return v
    else:
        k = Polynomial([.5,-.5])
        t = Polynomial([1])
        b = n.bit_length()
        for i in range(b-1,0-1,-1):
            t = t*t
            if (1<<i)&n:
                t = t*k
        return firl(t*Polynomial([g]))
    return hp


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


#todo: Lagrange interpolation delay
# why: it has ≤ 1 gain at all frequencies
#       so it's safe for feedback
#todo: windowed sinc fractional delay
# why: easy? good? idk.
delayl_P_inverses = dict()
def delayl(n=0,prec=3,band=.5,start=[]):
    assert n >= 0
    if n%1 == 0 or prec == 0:
        n = int(round(n))
        def do(v,i=[n],d=(start+[0]*(n+1))[:n+1]):
            d[i[0]] = v
            i[0] = (i[0]+1)%len(d)
            return d[i[0]]
        return do
    elif prec == 1 and 0:
        #linear interp
        r = n%1
        n = int(n)+1
        def do(v,i=[n,r],d=(start+[0]*(n+1))[:n+1]):
            d[i[0]] = v
            i[0] = (i[0]+1)%len(d)
            return d[i[0]]*i[1]+d[(i[0]+1)%len(d)]*(1-i[1])
        return do
    else:
        #predelay with digital delay line
        #want to make d be from floor(L/2+.5) to ceil(L/2 + .5)
        L = prec+1
        d = min(int(L/2+.5)+(n%1),n+1)
        #print(d)
        kern = delayFIRterms(d,L,band) #prec+1 to make it linear
        #print(kern)
        pad = max(0,1+int(n)-int(L/2+.5))
        #print(pad)
        line = (start + [0]*(int(n)+1+len(kern)))[:pad+len(kern)]
        #print(line)
        def do(v,i = [len(line)-1],d = line,k = kern[::-1]):
            d[i[0]] = v
            i[0] = (i[0]+1)%len(d)
            r = 0
            for o in range(len(k)):
                r += d[(i[0]+o)%len(d)]*k[o]
            return r
        return do
        


def delayFIRterms(D,L=3,alpha=.5):
    from signals import sinc
    #http://users.spa.aalto.fi/vpv/publications/vesan_vaitos/ch3_pt1_fir.pdf
    #h = (P^-1 p)
    #note that if n > prec, we must pad out with an integer delay
    #alpha = band
    #D = n adjusted for padding
    #L = prec
    # P_k,l = alpha*sinc(alpha*(k-l))  k,l = 1,2,...,L
    # p_l,k = alpha*sinc(alpha*(k-D))    k = 1,2,...,L
    #
    #P depends only on a and L, so we'll lazy eval and store the inverses
    #so each P^-1 Matrix needs to be made once
    key = (alpha,L)
    if key in delayl_P_inverses:
        P_inv = delayl_P_inverses[key]
    else:
        #matrix is m_row,col
        import scipy.linalg
        P = [[0]*L for i in range(L)]
        for k_l in range(1-L,L):
            r = alpha*sinc(alpha*(k_l))
            for k in range(max(0,k_l),min(1+k_l-(1-L),L)):
                l = k-k_l
                P[k][l] = r
        P_inv = scipy.linalg.inv(P)
        delayl_P_inverses[key] = P_inv
    # p_l,k = alpha*sinc(alpha*(k-D))    k = 1,2,...,L        
    p = [alpha*sinc(alpha*(k-D)) for k in range(1,L+1)]
    from numpy import matmul
    return matmul(P_inv,p)
def plotdelaysSlider(L=3,a=.5,plotN = 1<<8,ms=1):
    d = 1
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider,Button
    fig, ax = plt.subplots()
    La = [L]
    s = .5/plotN
    x = [i*s for i in range(1,plotN)]
    def phs(r,xes):
        xes = iter(xes)
        c = 0
        ph = (arg(r(next(xes)))/2/math.pi)
        yield ph
        for i in xes:
            phn = (arg(r(i))/2/math.pi)+c
            if abs(ph-phn)>.5:
                a = 1-2*(phn > ph)
                c += a
                phn += a
            yield phn
            ph = phn
    dat, = plt.plot(x, [1]*len(x), lw=ms)
    l, = plt.plot([0,.5], [1,1],"k--", lw=ms)
    mdat, = plt.plot(x, [(L+1)/2]*len(x), lw=ms)
    one, = plt.plot([0,.5], [(L+1)/2]*2,"k-", lw=ms)
    plt.grid(color='xkcd:grey',linestyle='-')
    plt.axis([0,.5,0,L+1])
    plt.subplots_adjust(left=0.25, bottom=0.25)
    
    axdly = plt.axes([0.25, 0.1, 0.65, 0.03])
    axA = plt.axes([0.25, 0.15, 0.65, 0.03])
    dly = Slider(axdly, 'Delay', 0, 1, valinit=1/(L+1))
    alph = Slider(axA, 'Alpha', 0, 1, valinit=a)
    def upd(val):
        a = alph.val
        d = (dly.val*(La[0]+1))
        l.set_ydata([d,d])
        r = response(delayFIRterms(d,La[0],a))
        p = [p for p in phs(r,x)]
        dat.set_ydata([1+p[i]/x[i] for i in range(len(x))])

        one.set_ydata([(La[0]+1)/2]*2)
        mdat.set_ydata([abs(r(i))*(La[0]+1)/2 for i in x])
        
        fig.canvas.draw_idle()
    dly.on_changed(upd)
    alph.on_changed(upd)

    p_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
    m_ax = plt.axes([0.25, 0.025, 0.1, 0.04])
    incL = Button(p_ax, '+')
    decL = Button(m_ax, '-')
    def ul(La=La):
        ax.axis([0,.5,0,La[0]+1])
    def il(ev,La=La):
        La[0] += 1
        ul()
        upd(0)
    incL.on_clicked(il)
    def dl(ev,La=La):
        La[0] = max(La[0]-1,1)
        ul()
        upd(0)
    decL.on_clicked(dl)
    return plt
    
def plotdelays(L=3,a=.5,dstep=.25,drange=None,plotN=1<<8,ms=.5):
    import matplotlib.pyplot as plt
    if drange == None:
        drange = [1,L]

    d = drange[0]
    while d < drange[1]+dstep:
        s = .5/(plotN)
        x = [i*s for i in range(1,plotN)]
        r = response(delayFIRterms(d,L,a))
        ph = [(arg(r(i))/2/math.pi)%1 for i in x]
        c = 0
        for i in range(len(ph)-1):
            ph[i+1] += c
            if abs(ph[i]-ph[i+1])>.5:
                a = 1-2*(ph[i+1] > ph[i])
                c += a
                ph[i+1] += a
        
        plt.plot([0,1],[d,d],"k--")
        plt.plot(x,[1+ph[i]/x[i] for i in range(len(x))],'o',ms=ms)
        d += dstep
    plt.axis([0,.5]+[drange[0]-dstep,drange[1]+dstep])
    plt.show()


def response(polyb=[1],polya=[1]):
    def resp(f,r=False,pb = polyb,pa = polya):
        #f in nyquist units
        denom = 0
        num = 0
        x = 1
        factor = eone**(1j*f)
        for i in range(min(len(pb),len(pa))):
            denom += x*pa[i]
            num += x*pb[i]
            x *= factor
        for i in range(len(pb),len(pa)):
            denom += x*pa[i]
            x *= factor
        for i in range(len(pa),len(pb)): #only one of these loops will run
            num += x*pb[i]
            x *= factor
        if r:
            return num,denom
        return num/denom
    return resp

def arg(x):
    return math.atan2(x.imag,x.real)

def safe0Log(x):
    if x <= 0:
        return -1e400
    return math.log(x)

def plotFilt(b=[1],a=[1],frange=[-1,1],res = 1<<12,mrange=[-4,4],ms=.5):
    import matplotlib.pyplot as plt
    r = response(b,a)
    def ran(l,h,s):
        while l < h:
            yield l
            l += s
    funcm = lambda n,d: safe0Log(abs(n))-safe0Log(abs(d))
    funcp = lambda n,d: ((arg(n)-arg(d))/(2*math.pi))
    plt.plot([f for f in ran(frange[0],frange[1],1/res)],[funcm(*r(f,1)) for f in ran(frange[0],frange[1],1/res)],'o',[f for f in ran(frange[0],frange[1],1/res)],[funcp(*r(f,1)) for f in ran(frange[0],frange[1],1/res)],'o',ms=ms)
    plt.axis(frange+mrange)
    plt.show()


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


def ffloor(f):
    if type(f) == float and abs(f) == 1e400:
        return f
    return f-(f%1)

class L_gain:
    def __init__(self,gain=1):
        self.gain = [gain]
        self._keepmeinscope = None
    def plot(self,gm=0,gM=2,gd=1,gf = lambda x:x, glf = lambda x,ox: "Gain:"+\
             str(ffloor(safe0Log(x)*10*100)/100)+" dB"):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
        plt.figure()
        axm = plt.axes([0.15, 0.2, .2, .7])
        axp = plt.axes([0.65, 0.2, .2, .7])
        sm = Slider(axm, 'Gain', gm, gM, valinit=gd,orientation = 'vertical')
        sp = Slider(axp, 'Phase', -math.pi, math.pi, valinit=0,orientation = 'vertical')
        def update(val,g=self.gain):
            m = gf(sm.val)
            sm.label.set_text(glf(m,sm.val))
            g[0] = (math.e**(1j*sp.val))*m
        sm.on_changed(update)
        sp.on_changed(update)
        self._keepmeinscope = update
        return plt
    def __call__(self):
        def do(v,g=self.gain,keepmeinscope=self):
            return v*g[0]
        return do
        
class L_idelay:
    def __init__(self,delay=0,maxD=4800):
        self.delay = [delay]
        self.maxD = maxD
        self._keepmeinscope = None
    def plot(self,df=None):
        if df == None:
            df = 10
        if type(df) in [float,int]:
            b = df
            df = lambda x: self.maxD*(x*b**(x-1))
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
        plt.figure()
        axm = plt.axes([0.1, 0.3, .8, .4])
        sm = Slider(axm, 'delay', 0, 1, valinit=0)
        def update(val,d=self.delay):
            v = df(sm.val)
            d[0] = int(v)
            sm.label.set_text("delay:"+str(d[0]))
        sm.on_changed(update)
        self._keepmeinscope = update
        return plt
    def __call__(self,start=[]):
        def do(v,i=[self.delay[0]],d=self.delay,b=(start+[0]*(self.maxD+1))[:self.maxD+1],keepmeinscope=self):
            b[i[0]] = v
            i[0] = (i[0]+1)%len(b)
            return b[(i[0]-d[0]-1)%len(b)]
        return do

class L_delay:
    def __init__(self,delay=0,maxD=480,prec=3,band=.5):
        self.delay = [delay]
        self.pad = [0]
        self.maxD = maxD+prec+1
        self.band = band
        self.prec = prec
        self.kern = [None]
        self._keepmeinscope = None
        self.update()
    def update(self):
        n = self.delay[0]
        L = self.prec+1
        pad = min(max(0,int(n-L/2+1)),self.maxD-L)
        d = 1 + n-pad
        self.kern[0] = delayFIRterms(d,L,self.band) 
        self.pad[0] = pad
    def __call__(self,start=[]):
        line = (start + [0]*self.maxD)[:self.maxD]
        def do(v,i = [self.pad[0]],p=self.pad,d = line,k = self.kern):
            d[i[0]] = v
            r = 0
            for o in range(len(k[0])):
                r += d[(i[0]-p[0]-o)%len(d)]*k[0][o]
            i[0] = (i[0]+1)%len(d)
            return r
        return do
    def plot(self,df=None):
        if df == None:
            df = 10
        if type(df) in [float,int]:
            b = df
            df = lambda x: self.maxD*(x*b**(x-1))
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
        #plt.figure()
        fig, ax = plt.subplots()
        ax.set_ylim((min(self.kern[0]+[-1]),max(self.kern[0]+[1])))
        l, = ax.plot([i for i in range(self.prec+1)], self.kern[0],"o", ms=2)
        plt.subplots_adjust(left=.80,right=.95)
        
        axm = plt.axes([0.15, 0.5, .6, .4])
        sm = Slider(axm, 'delay', 0, 1, valinit=0)
        axb = plt.axes([0.15, 0.1, .6, .2])
        sb = Slider(axb, 'band', 0, 1, valinit=self.band)
        
        def update(val,d=self.delay):
            v = df(sm.val)
            d[0] = v
            self.band = sb.val
            self.update()
            l.set_ydata(self.kern[0])
            ax.set_ylim((min(self.kern[0]+[-1]),max(self.kern[0]+[1])))
            sm.label.set_text("delay:"+str(math.floor(d[0]*100)/100))
            fig.canvas.draw_idle()
            
        sm.on_changed(update)
        sb.on_changed(update)
        
        self._keepmeinscope = update
        return plt

def allpassl(r,f):
    c = -2*r*math.cos(2*math.pi*f)
    return iir2l(c,r*r,r*r,c,1)

import numpy as np
def fi(n):
    for i in range(n//2):
        yield i/n
    for i in range(n//2-n,0):
        yield i/n
        
def fourierFilter(kern = np.array([1.]),window = None):
    if window == None:
        window = np.array([.5*(1-math.cos(2*math.pi*i/len(kern))) for i in range(len(kern))])
    def do(v,i = [0,len(kern)//2],outb = [np.array([0.j]*len(kern)),np.array([0.j]*len(kern))],k=kern,w=window):
        v,outb[0][i[0]],outb[1][i[1]] = outb[0][i[0]]+outb[1][i[1]],v,v
        i[0] = (i[0]+1)%len(k)
        if i[0] == 0:
            outb[0] = np.fft.ifft(np.fft.fft(outb[0]*w)*k)
        i[1] = (i[1]+1)%len(k)
        if i[1] == 0:
            outb[1] = np.fft.ifft(np.fft.fft(outb[1]*w)*k)
        return v
    return do
    

def fourierStereoify(ramt=.6,amt=.3,fade=.9,l=1<<14):
    if type(l) == int:
        baseKern = np.array([1+0j]*l)
    else:
        baseKern = l
        l = len(l)
    window = np.array([.5*(1-math.cos(2*math.pi*i/l)) for i in range(l)])
    kernf = lambda : (np.random.random(l)-.5)*ramt+(np.random.random(l)-.5)*amt*1j
    def do(v,i = [0,l//2],outb = [np.array([0.j]*l),np.array([0.j]*l)],k=[baseKern,kernf()],w=window,l=l):
        v,outb[0][i[0]],outb[1][i[1]] = outb[0][i[0]]+outb[1][i[1]],v,v
        i[0] = (i[0]+1)%l
        if i[0] == 0:
            outb[0] = np.fft.ifft(np.fft.fft(outb[0]*w)*(k[0]+k[1]))
            k[1] *= fade
            k[1] += kernf()
        i[1] = (i[1]+1)%l
        if i[1] == 0:
            outb[1] = np.fft.ifft(np.fft.fft(outb[1]*w)*(k[0]+k[1]))
            k[1] *= fade
            k[1] += kernf()
        return v
    return do
    


def fourierFuncFilter(filt = lambda a:a,kerl=256,window = None):
    if window == None:
        window = np.array([.5*(1-math.cos(2*math.pi*i/kerl)) for i in range(kerl)])
    def do(v,i = [0,kerl//2],outb = [np.array([0.j]*kerl),np.array([0.j]*kerl)],w=window):
        v,outb[0][i[0]],outb[1][i[1]] = outb[0][i[0]]+outb[1][i[1]],v,v
        i[0] = (i[0]+1)%len(outb[0])
        if i[0] == 0:
            outb[0] = np.fft.ifft(filt(np.fft.fft(outb[0]*w)))
        i[1] = (i[1]+1)%len(outb[1])
        if i[1] == 0:
            outb[1] = np.fft.ifft(filt(np.fft.fft(outb[1]*w)))
        return v
    return do

def npFIR(b=np.array([1.+0j])):
    def do(v,i=[0],s=np.array([0.j]*len(b)),k=b):
        s[i[0]] = v
        r = np.dot(np.concatenate((s[i[0]:],s[:i[0]])),k)
        i[0] = (i[0]-1)%len(s)
        return r
    return do

"""def sincOctaveStack(octaves=10,kernLen=6,window = lambda x: math.pow(-(x*x)*16)):
    kern = np.array([math.sin(math.pi*(i+.5))/(math.pi*(i+.5)) * window((i+.5)/(kernLen+1)) for i in range(kernLen)])
    def do(v,i=[0],s=[np.array(


\"""
class L_sinc_lowpass:
    def __init__(self,delay=0,maxD=4800):
        self.delay = [delay]
        self.maxD = maxD
        self._keepmeinscope = None
    def plot(self,df=None):
        if df == None:
            df = 10
        if type(df) in [float,int]:
            b = df
            df = lambda x: self.maxD*(x*b**(x-1))
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
        plt.figure()
        axm = plt.axes([0.1, 0.3, .8, .4])
        sm = Slider(axm, 'delay', 0, 1, valinit=0)
        def update(val,d=self.delay):
            v = df(sm.val)
            d[0] = int(v)
            sm.label.set_text("delay:"+str(d[0]))
        sm.on_changed(update)
        self._keepmeinscope = update
        return plt
    def __call__(self,start=[]):
        def do(v,i=[self.delay[0]],d=self.delay,b=(start+[0]*(self.maxD+1))[:self.maxD+1],keepmeinscope=self):
            b[i[0]] = v
            i[0] = (i[0]+1)%len(b)
            return b[(i[0]-d[0]-1)%len(b)]
        return do



    
    
class L_rbiquad:#real biquad
    def __init__(self):
        self.gzp = [1,0,0] #gain,zero,pole
                           #zero and pole are complex
    def __call__(self,state=[]):
        state = (state+[0,0])[:2]
        def f(v,s=state,d=self.gzp):
            v -= s[0]*a[0]+s[1]*a[1]
            v,s[0],s[1] = v*b[0]+s[0]*b[1]+s[1]*b[2],v,s[0]
            return v*d[0]
        return f
    def plot(self,plotN=1<<6):
        pass
    
#http://sites.music.columbia.edu/cmc/MusicAndComputers/chapter4/04_09.php
#def Karplus_Strong(l,d=.999,s=None):
#    if s == None:
#        s = [(random.random()-.5)*(1+1j) for i in range(l)]
#    def do(v):
"""





    

def capMagl(cap=1):
    def do(v,c=[cap]):
        if abs(v) > c[0]:
            return c[0]*v/abs(v)
        return v
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


def deriv(g):
    p = 0
    for v in g:
        yield v-p
        p=v
def integ(g,a=1):
    v = 0
    for i in g:
        v *= a
        v += i
        yield v
