
from itools import lmbdWr,lm

import itertools

from bisect import bisect_right

import brailleG as gr

def abs2(n):
    return (n*n.conjugate()).real
    

def fsample(buf,m=1,b=0):
    index = 0
    y = 0
    while 1:
        index = (index+b+m*y)%len(buf)
        y = yield buf[(int(index)+1)%len(buf)]*(index%1)+buf[int(index)]*(1-(index%1))

def fsine(a=1,m=1/48000,b=0):
    s = 0
    c = a
    y = 0
    while 1:
        amt = b+m*y
        s += c*amt
        c -= s*amt
        y = yield s

import math
pi = math.pi
eone = math.exp(2*pi)
buffer_size = 8192

sinBuffer = [math.sin(i*2*math.pi/4/buffer_size) for i in range(buffer_size+1)]
        
def nsin(a):
    a = 4*buffer_size*(a%1)
    if a<=buffer_size:
        return sinBuffer[math.floor(a)]
    elif a<=buffer_size*2:
        return sinBuffer[math.floor(buffer_size-a)-1]
    elif a<=buffer_size*3:
        return -sinBuffer[math.floor(a-buffer_size*2)]
    else:
        return -sinBuffer[math.floor(buffer_size*3-a)-1]
def nsaw(a):
    return (a%1)*2-1
def ntri(a):
    return abs((a%1)-.5)*4-1
def nsquare(a,p=.5):
    return ((a%1)<p)*2-1

lsin = lm(nsin)
lsaw = lm(nsaw)
ltri = lm(ntri)
lsqr = lm(nsquare)

_reserved = []
def polyvars(varstr):
    return [Polynomial(i) for i in varstr]
class Polynomial:
    def __init__(self,coef,var="x"):
        if type(coef) == str:
            var = coef
            coef = [0,1]
        self.a = coef
        self.var = var
        self.trim()
    def trim(self):
        while len(self.a):
            if self.a[-1] == 0:
                self.a = self.a[:-1]
            else:
                break
    def simplified(self,tv=None):
        if tv == None:
            tv = self.var
        if tv == self.var:
            r = Polynomial([],tv)
            x = Polynomial(tv)
            xa = 1
            for t in self.a:
                if type(t) == Polynomial:
                    r += t.simplified(tv)*xa
                else:
                    r += t*xa
                xa *= x
        else:
            r = Polynomial([],tv)
            x = Polynomial(self.var)
            xa = 1
            for t in self.a:
                if type(t) == Polynomial:
                    r += t.simplified(tv).__mul__(xa)
                else:
                    r += t*xa
                xa *= x
        return r
    def __call__(self,vrs):
        if type(vrs) != dict:
            vrs = {self.var:vrs}
        if self.var in vrs:
            x = vrs[self.var]
            v = 0
            xa = 1
            for t in self.a:
                if type(t) == Polynomial:
                    t = t(vrs)
                v += xa*t
                xa *= x
            return v
        return Polynomial([t(vrs) if type(t) == Polynomial else t for t in self.a],self.var)
    def __getitem__(self,i):
        if i>=len(self):
            return 0
        return self.a[i]
    def __setitem__(self,i,v):
        if i>=len(self):
            self.a += [0]*(i-len(self))+[v]
        else:
            self.a[i] = v
        self.trim()
    def __neg__(self):
        return Polynomial([-i for i in self.a],self.var)
    def __radd__(self,o):
        return self.__add__(o)
    def __add__(self,o):
        if type(o) == Polynomial and o.var == self.var:
            return self.padd(o)
        return self.npadd(o)
    def padd(self,o):
        return Polynomial(sumPolyn(self.a,o.a),self.var)
    def npadd(self,o):
        if len(self.a):
            return Polynomial([self.a[0]+o]+self.a[1:],self.var)
        return Polynomial([o],self.var)
    def __rsub__(self,o):
        return -self.__sub__(o)
    def __sub__(self,o):
        if type(o) == Polynomial and o.var == self.var:
            return self.psub(o)
        return self.npsub(o)
    def psub(self,o):
        return self.padd(-o)
    def npsub(self,o):
        if len(self.a):
            return Polynomial([self.a[0]-o]+self.a[1:],self.var)
        return Polynomial([-o],self.var)
    def __rmul__(self,o):
        return self.__mul__(o)
    def __mul__(self,o):
        if type(o) == Polynomial and o.var == self.var:
            return self.pmul(o)
        return self.npmul(o)
    def pmul(self,o):
        return Polynomial(prodPolyn(self.a,o.a),self.var)
    def npmul(self,o):
        if len(self.a):
            return Polynomial([e*o for e in self.a],self.var)
        return Polynomial([],self.var)
    #def __divmod__(self,o):
    #def __repr__(self,var=None):
    #    if var == None:
    #        var = self.var
    #    return f"polyn({var}) = "+" + ".join((f"({self.a[i]})"+["",f"{var}"][i>0]+["",f"**{i}"][i>1] for i in range(len(self.a))))
    def __repr__(self,var=None):
        if var == None:
            var = self.var
        return f"p({var})="*0+" + ".join(((f"({self.a[i]})" if self.a[i] != 1 else ["1",""][i>0])+["",f"{var}"][i>0]+["",f"**{i}"][i>1] for i in range(len(self.a)) if self.a[i] != 0))
    def deriv(self):
        return Polynomial([self.a[i+1]*(i+1) for i in range(len(self.a)-1)],self.var)
    def integ(self,k=0):
        return Polynomial([k]+[self.a[i]*(1/(i+1)) for i in range(len(self.a))],self.var)
    def convolve(self,o):
        #integ of self(t-x)o(t) dt
        #want first arg to be x, second to be bounds
        #so, 
        x = Polynomial('x')
        t = Polynomial('t')
        integrand = Polynomial([self(t-x)*o(t)],'t')
        return integrand.simplified('t').integ()
    def __len__(self):
        return len(self.a)
    def __eq__(self,o):
        if type(o) == Polynomial:
            if len(o) != len(self) or o.var != self.var:
                return False
            for i in range(len(self)):
                if self.a[i] != o.a[i]:
                    return False
            return True
        if type(o) == float or type(o) == int or type(o) == complex:
            return len(self) <= 1 and (self.a+[0])[0] == o
    def __matmul__(self,o):
        return self.convolve(o)
    def plot(self,*args):
        plotPoly(self.a,*args)
    
def evalPolyn(polyn,x):
    v = 0
    xa = 1
    for t in polyn:
        v += xa*t
        xa *= x
    return v
def sumPolyn(p1,p2):
    res = [0 for i in range(max(len(p1),len(p2)))]
    for i in range(len(res)):
        if i < len(p1):
            if i < len(p2):
                res[i] = p1[i] + p2[i]
            else:
                res[i] = p1[i]
        else:
            res[i] = p2[i]
    return res
def prodPolyn(p1,p2):
    if len(p1) == 0 or len(p2) == 0:
        return []
    res = [0 for i in range(len(p1)+len(p2)-1)]
    for i in range(len(p1)):
        for j in range(len(p2)):
            res[i+j] += p1[i]*p2[j]
    return res
def composePolyn(p1,p2): #retuns p1(p2(x))
    px = [1]
    pr = []
    for i in p1:
        pr = sumPolyn(pr,prodPolyn(px,[i]))
        px = prodPolyn(px,p2)
    return pr

def fourierPolyn(p,freq):
    factor = 1/(2j*math.pi*freq)
    mask = [factor]
    result = [0 for i in p]
    for i in range(len(p)):
        facacc = factor
        for j in range(i,-1,-1):
            result[j] += facacc*p[i]
            facacc *= -factor*j
    return result
def evalFourierPolyn(p,freq,phase,low,high):
    l = evalPolyn(p,low)
    h = evalPolyn(p,high)
    return h*(eone**(1j*(freq*high+phase)))-l*(eone**(1j*(freq*low+phase)))
    

def convolvePolyn(p1,p2):
    pass



def softGCD(a,b,f=.01):
    if abs(b)<=f:
        return a
    return softGCD(b,a%b,f)

def convPolyFrags(p0,p1,l0,l1):
    #convolves 2 polynomial fragments
    if l0 > l1:
        return convPolyFrags(p1,p0,l1,l0)
    #l0 ≤ l1
    times = [-l0,0,l1-l0]
    #moving = composePolyn(p0,[t-x])

    def xify(v):
        return Polynomial([v],'x').simplified('x')
    
    p_0 = Polynomial(p0,'x')
    p_1 = Polynomial(p1,'x')
    x = Polynomial('x')
    conv = p_0@p_1
    a,b,c = conv(l0+x)-conv(0),conv(l0+x)-conv(x),conv(l1)-conv(x)
    a,b,c = [xify(xify((a,b,c)[i])(x+times[i])) for i in range(3)]
    if l1 != l0:
        return PiecewizePoly([[],a.a,b.a,c.a,[]],[-math.inf]+times+[l1],0)
    return PiecewizePoly([[],a.a,c.a,[]],[-math.inf]+times[:2]+[l1],0)
    
    

def plotPoly(p,t0=0,t1=1,res=50):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=1, ncols=1)
    st = p[0]
    end = evalPolyn(p,t1-t0)
    mid = []
    ts = []
    if len(p) > 2:
        for j in range(1,res):
            t = (t1-t0)*j/res
            ts += [t+t0]
            mid += [evalPolyn(p,t)]
    ts = [t0]+ts+[t1]
    ys = [st]+mid+[end]
    plt.plot(ts,[i.real for i in ys],linestyle='-',color=(.3,.3,1), linewidth=2)
    plt.plot(ts,[i.imag for i in ys],linestyle='-',color=(1,.3,.3), linewidth=2)
    plt.show(block=0)

#todo: closed form convolution
#   perhaps use @ (__matmul__)
#todo: closed form composition? possible? not always: requires root finding
class PiecewizePoly:
    def __init__(self,polys = [[]],times=[0],mod=1):
        self.times = times
        self.polys = polys
        self.mod = mod
    def __call__(self,x):
        if self.mod != 0:
            x %= self.mod
        #binary search for correct polyn
        l = bisect_right(self.times,x)-1
        #eval polyn
        return evalPolyn(self.polys[l],x-self.times[l])
    def deriv(self):
        #do derivitive on self
        res_t = []
        res_p = []
        for p in range(len(self.polys)):
            res_t += [self.times[p]]
            res_p += [[]]
            for i in range(len(self.polys[p])-1):
                res_p[-1] += [self.polys[p][i+1]*(i+1)]
        return PiecewizePoly(res_p,res_t,self.mod)
    def integ(self,start=0,scale=1):
        #do integral on self
        res_t = []
        res_p = []
        for p in range(len(self.polys)):
            res_t += [self.times[p]]
            res_p += [[start]]
            for i in range(len(self.polys[p])):
                res_p[-1] += [self.polys[p][i]/(i+1)*scale]
        #continuize segments after first
        for i in range(1,len(res_t)):
            val = evalPolyn(res_p[i-1],res_t[i]-res_t[i-1])
            res_p[i][0] = val#-evalPolyn(res[i][1],res[i][0]) #not needed with new def
        return PiecewizePoly(res_p,res_t,self.mod)
    def timeshift(self,s):
        assert self.mod==0
        for i in range(len(self.times)):
            self.times[i] -= s
        return self
    def timescale(self,s):
        self.mod *= s
        for i in range(len(self.times)):
            self.times[i] *= s
            self.polys[i] = composePolyn(self.polys[i],[0,1/s])
        return self
    def convolve(self,o,fudgefactor = .001):
        ts = self.times + [self.mod if self.mod else math.inf]
        to = o.times + [o.mod if o.mod else math.inf]
        result = PiecewizePoly([[]],[-math.inf],0)
        for i in range(len(self.polys)):
            for j in range(len(o.polys)):
                pc = convPolyFrags(self.polys[i],o.polys[j],ts[i+1]-ts[i],to[j+1]-to[j])
                result += pc.timeshift(ts[i]-to[j])
        #now do moddy stuff
        return result
    
    
    def __matmul__(self,o,fudgefactor = .001):
        return self.convolve(o,fudgefactor)
    def __lmbdWr__(self):
        return lmbdWr(self)
    def __iterWr__(self):
        return iterWr(iter(lmbdWr(self)))
    def bias(self):
        intg = self.integ()
        return (intg.end()-intg(0))/self.mod
    def unbiased(self):
        #self shifted to have 0 dc bias
        bias = self.bias()
        res_t = []
        res_p = []
        for p in range(len(self.polys)):
            res_t += [self.times[p]]
            res_p += [sumPolyn([-bias],self.polys[p])]
        return PiecewizePoly(res_p,res_t,self.mod)
    def graph(self,w=40,h=20,lo=-2,hi=2):
        gr.graph(self,0,self.mod,lo,hi,w,h)
    def plot(self,res=50):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(nrows=1, ncols=1)
        dash = 0
        for i in range(len(self.polys)):
            dash = 1-dash
            t0 = self.times[i]
            if t0 == -math.inf:
                t0 = self.times[i+1]-1
            t1 = (self.times+[self.mod if self.mod != 0 else self.times[-2]+1])[i+1]
            p = self.polys[i]
            st = (p+[0])[0]
            end = evalPolyn(p,t1-t0)
            mid = []
            ts = []
            if len(p) > 2:
                for j in range(1,res):
                    t = (t1-t0)*j/res
                    ts += [t+t0]
                    mid += [evalPolyn(p,t)]
            ts = [t0]+ts+[t1]
            ys = [st]+mid+[end]
            plt.plot(ts,[i.real for i in ys],linestyle='-',color=(.3*dash,.3*dash,1), linewidth=2)
            plt.plot(ts,[i.imag for i in ys],linestyle='-',color=(1,.3*dash,.3*dash), linewidth=2)
        plt.show(block=0)
        
    def mag2(self):
        sqd = PiecewizePoly([prodPolyn(p,p) for p in self.polys],[t for t in self.times],self.mod+1).integ()
        return (sqd(self.mod)-sqd(0))/self.mod
    def norm(self,v=.5):
        #normalizes it so that integ(0,mod, of self^2) = v*mod
        target = v
        factor = target/self.mag2()**.5
        return PiecewizePoly([[i*factor for i in p] for p in self.polys],[t for t in self.times],self.mod)
    def __add__(self,o,fudgefactor = .001):
        if type(o) == PiecewizePoly:
            if self.mod == 0:
                assert o.mod == 0
                res_t = [-math.inf]
                res_p = [sumPolyn(self.polys[0],o.polys[0])]
                si = 0
                oi = 0
                sts = self.times + [math.inf]
                ots = o.times + [math.inf]
                sp = self.polys + [[]]
                op = o.polys + [[]]
                while si < len(self.times) and oi < len(o.times):
                    st,ot = sts[si+1],ots[oi+1]
                    if st < ot:
                        si += 1
                        res_t += [st]
                        res_p += [sumPolyn(sp[si],
                                           composePolyn(op[oi],[st-ot,1]))]
                    elif st > ot:
                        oi += 1
                        res_t += [ot]
                        res_p += [sumPolyn(composePolyn(sp[si],[ot-st,1]),
                                           op[oi])]
                    else:
                        si += 1
                        oi += 1
                        res_t += [st]
                        res_p += [sumPolyn(sp[si],op[oi])]
                return PiecewizePoly(res_p,res_t,0)
                    
            else:
                assert o.mod != 0
            gcd = softGCD(self.mod,o.mod,fudgefactor*(self.mod*o.mod)**.5) 
            lcm = self.mod*o.mod/gcd 
            t = 0 
            res_t = []
            res_p = []
            sto = 0
            oto = 0
            si = 0
            oi = 0
            while t < lcm:
                res_t += [t]
                res_p += [sumPolyn(composePolyn(self.polys[si],[t-(self.times[si]+sto),1]),
                                   composePolyn(o.polys[oi],[t-(o.times[oi]+oto),1]))]
                
                st = sto+(self.times+[self.times[0]+self.mod])[si+1]
                ot = oto+(o.times+[o.times[0]+o.mod])[oi+1]
                t = min(st,ot)
                if st <= t:
                    si += 1
                    if si >= len(self.polys):
                        si = 0
                        sto += self.mod
                if ot <= t:
                    oi += 1
                    if oi >= len(o.polys):
                        oi = 0
                        oto += o.mod
            return PiecewizePoly(res_p,res_t,lcm)
        else:
            return PiecewizePoly([sumPolyn(p,[o]) for p in self.polys],[t for t in self.times],self.mod)
    def __mul__(self,o,fudgefactor = .001):
        if type(o) == PiecewizePoly:
            gcd = softGCD(self.mod,o.mod,fudgefactor*(self.mod*o.mod)**.5) 
            lcm = self.mod*o.mod/gcd 
            t = 0 
            res_t = []
            res_p = []
            sto = 0
            oto = 0
            si = 0
            oi = 0
            while t < lcm:
                res_t += [t]
                res_p += [prodPolyn(composePolyn(self.polys[si],[t-(self.times[si]+sto),1]),
                                   composePolyn(o.polys[oi],[t-(o.times[oi]+oto),1]))]
                
                st = sto+(self.times+[self.times[0]+self.mod])[si+1]
                ot = oto+(o.times+[o.times[0]+o.mod])[oi+1]
                t = min(st,ot)
                if st <= t:
                    si += 1
                    if si >= len(self.polys):
                        si = 0
                        sto += self.mod
                if ot <= t:
                    oi += 1
                    if oi >= len(o.polys):
                        oi = 0
                        oto += o.mod
            return PiecewizePoly(res_p,res_t,lcm)
        else:
            return PiecewizePoly([prodPolyn(p,[o]) for p in self.polys],[t for t in self.times],self.mod)
        
    def __radd__(self,o):
        return self.__add__(o)
    def __rmul__(self,o):
        return self.__mul__(o)
    def __sub__(self,o):
        return self.__add__(o.__mul__(-1))
    def __rsub__(self,o):
        return self.__mul__(-1).__add__(o)
    
    def t(self,v=1):
        return PiecewizePoly([[p[i]/(v**i) for i in range(len(p))] for p in self.polys],[t*v for t in self.times],self.mod*v)
    def isZero(self):
        for i in self.polys:
            for j in i:
                if j != 0:
                    return False
        return True
    def end(self):
        x = self.mod
        l = -1
        #eval polyn
        return evalPolyn(self.polys[l],x-self.times[l])
        
    def freqComponent(self,f):
        if f == 0:
            return self.bias()
        result = 0
        f /= self.mod
        for i in range(len(self.polys)):
            p = fourierPolyn(self.polys[i],f)
            result += evalFourierPolyn(p,f,f*self.times[i],0,(self.times+[self.mod])[i+1]-self.times[i])
        return result
    def graphSpectrum(self,w=20,h=10,both=True):
        gr.graphLT(lambda x:abs(self.freqComponent(x)),both-h*2*both,h*(4-2*both)+both,0,1,w,h)
    def graphSpectrumLog(self,w=20,h=10,both = True,low=-10,hi=1):
        gr.graphLT(lambda x: (lambda v: (math.log(v) if v!=0 else -1e300))(abs(self.freqComponent(x))),both-h*2*both,h*(4-2*both)+both,low,hi,w,h)

    def bandLimit(self,t,bl=5,neg=False):
        tot = 0
        for i in range(neg*(1-bl),bl):
            tot += eone**(1j*i*t)*self.freqComponent(i)
        return tot

    def getBandlimitedBuffer(self,denominator,numerator = 1,ff=0,fnd=2,neg=False):
        #f_nyquist = .5
        # f_n = n*(num/den) < f_nyquist
        # n < .5*den/num
        d = softGCD(numerator,denominator,ff)
        numerator=int(round(numerator/d))
        denominator=int(round(denominator/d))
        return [self.bandLimit(numerator*i*self.mod/denominator,int(denominator/numerator/fnd),neg) for i in range(numerator*denominator)]

    def bandConj(self,t,bl=5):
        tot = 0
        re = self.real()
        im = self.imag()
        for i in range(0,bl):
            f = eone**(1j*i*t)
            tot += (f*re.freqComponent(i)).imag+(f*im.freqComponent(i)).imag*1j
        return tot
    
    
    
    def real(self):
        return PiecewizePoly([[i.real for i in j]for j in self.polys],[t for t in self.times],self.mod)
    def imag(self):
        return PiecewizePoly([[i.imag for i in j]for j in self.polys],[t for t in self.times],self.mod)
    def oscope(self,w=40,h=20,s=.5+.5j,m=.5,n=256):
        scrn = gr.brailleScreen(w*2,h*4)
        for i in range(n):
            t = i*self.mod/n
            v = self(t).conjugate()*m+s
            if 0<=int(v.real*w*2)<w*2 and 0<=int(v.imag*h*4) < h*4:
                gr.brailleScreenSet(scrn,int(v.real*w*2),int(v.imag*h*4))
        gr.brailleScreenPrint(scrn)
        
def ditherPoly(p,rate=440/48000,dd=1):
    from random import random
    t = 0
    while 1:
        t += rate
        yield p(t+dd*rate*random())

        
def gaussApprox(mean=0,spread=1,iters=3):
    s = spread/iters
    blip = PiecewizePoly([[],[1/s],[]],[-math.inf,0,s],0)
    acc = blip
    for b in bin(iters)[3:]:
        acc.plot()
        acc @= acc
        if b == '1':
            acc @= blip
    return acc.timeshift(mean)
        

def plinConnectDots(dat,speed=1):
    polys = []
    times = []
    t = 0
    for i in range(len(dat)):
        leng = abs(dat[i-1]-dat[i])
        polys += [[dat[i-1],(dat[i]-dat[i-1])/leng]]
        times += [t]
        t += leng
    return PiecewizePoly(polys,times,t)
def pnlinConnectDots(dat,speed=1):
    r = plinConnectDots(dat,speed)
    return r.t(1/r.mod)
    
    
def papprox(dat,integ=2):
    #derivitive the freqs integ times
    dcs = []
    for intg in range(integ):
        dcs += [dat[-1]/(intg+1)]
        ddat = [(-dat[i-1]+dat[i])/(intg+1) for i in range(len(dat))]
        dat = ddat
    res = PiecewizePoly([[i] for i in dat],[i for i in range(len(dat))],len(dat))
    for i in range(integ):
        res = res.integ(dcs[-i-1])
    return res
    """bl = len(dat)//2
    guess1 = PiecewizePoly([[i] for i in dat],[i/len(dat) for i in range(len(dat))],1)
    freqs = [guess1.freqComponent(i) for i in range(1-bl,bl)]
    dc = guess1.bias()
    #derivitive the freqs integ times
    for i in range(integ):
        for f in range(len(freqs)):
            freqs[f] *= (f+1-bl//2)*1j
    #come up with new samples to integrate repeatedly
    samps = []
    for t in range(len(dat)):
        tot = 0
        for i in range(1-bl,bl):
            tot += eone**(1j*i*t/len(dat))*freqs[i]
        samps += [tot]
    res = PiecewizePoly([[i] for i in samps],[i/len(samps) for i in range(len(samps))],1)
    for i in range(integ):
        res = res.unbiased().integ(0,1).unbiased()
    return res + dc
    """
    

def ppulse(width=.5,amplitude=1):
    return PiecewizePoly([[0,[-1]],[width,[1]]]).unbiased()


psqr = PiecewizePoly([[-1],[1]],[0,.5])

#.5 -> 2
ptri = psqr.integ(0,4).unbiased()

#.25*.5=1/8
ppar = ptri.integ(0,8)


psaw = PiecewizePoly([[1,-2]],[0])


cf = pnlinConnectDots([-.75+1.5j,-.5+1j,.5+1j,.75+1.5j,1+1j,1-1j,-1j-1,1j-1])*.5

cfi = plinConnectDots([-.75+1.5j,-.5+1j,.5+1j,.75+1.5j,1+1j,1-1j,-1j-1,1j-1])
cfi.polys += [[-1/3+.5j,-1j],[1/3+.5j,-1j]]
cfi.times += [cfi.mod,cfi.mod+.75]
cfi.mod += 1.5
cfi = cfi.t(1/cfi.mod)*.5



def reorderTimes(times,order,top):
    newTimes = []
    t = 0
    for i in order:
        if i == len(times)-1:
            l = top-times[i]
        else:
            l = times[i+1]-times[i]
        newTimes += [t]
        t += l
    return newTimes

def reorder(wave,goal,fs=20,wfd = lambda f,a,b: abs(abs2(a)-abs2(b))):
    l = [i for i in range(len(wave.polys))]
    goalF = [goal.freqComponent(i) for i in range(1-fs,fs)]
    best = wave
    bestD = 1e300
    for p in itertools.permutations(l):
        guess = PiecewizePoly([wave.polys[i] for i in p],reorderTimes(wave.times,p,wave.mod),wave.mod)
        guessF = [guess.freqComponent(i) for i in range(1-fs,fs)]
        d = 0
        for i in range(len(goalF)):
            d += wfd(1-fs+i,goalF[i],guessF[i])
        if d < bestD:
            best = guess
            bestD = d
    return best


def quickStar(n,s=2):
    return pnlinConnectDots([eone**(1j*i*s/n) for i in range(n)])*.5

def prettyStar(n,rl=.5):
    return pnlinConnectDots([eone**(1j*(i+.5*j)/n)*[1,rl][j] for i in range(n) for j in range(2)])*.5



def getPSineApprox(sects=2,integs=12):
    offs = integs%4
    guess = PiecewizePoly([[math.sin(((i+.5)/sects+offs/4)*2*math.pi)] for i in range(sects)],[i/sects for i in range(sects)]).unbiased()
    for i in range(integs):
        guess = guess.integ(0,1).unbiased().norm()
    return guess





def c(f,g):
    for i in g:
        yield f(i)

def x(n,g):
    for i in g:
        yield n*i
def p(n,g):
    for i in g:
        yield n+i
def const(n):
    while 1:
        yield n
def integ(g,a=0):
    for i in g:
        a += i
        yield a
def deriv(g):
    p = next(g)
    for i in g:
        yield i-p
        p = i
def clamp(n,v=1):
    return min(max(n,-v),v)
def bderiv(g,b=1):
    p = next(g)
    d = 0
    for i in g:
        d += i-p
        p = i
        v = clamp(d,b)
        yield v
        d -= v




        
        
def send(g1,g2):
    next(g1)
    while 1:
        yield g1.send(next(g2))
        
class passFilter:
    def __init__(self):
        self.value = 0
    def send(self,val,time=1):
        self.value = val
        return val
class contRAvgFilt(passFilter):
    def __init__(self,a):
        self.alpha = math.log(a)
        self.value = 0
    def send(self,val,time=1):
        self.value = val+(self.value-val)*math.exp(self.alpha*time)
        return self.value

def getPerfSquareBuff(n,d=1):
    w = 1
    outbuf = [0 for i in range(n)]
    while w < n/d/2:
        for i in range(n):
            outbuf[i] += math.sin(i*2*pi*d/n*w)/w
        w += 2
    return outbuf


def nearestDownSample(g,r=1):
    a = 0
    for i in g:
        while a < 1:
            yield i
            a += r
        a -= 1
        
def linearDownSample(g,r=1):
    p = 0
    a = 0
    for i in g:
        while a < 1:
            yield a*i+(1-a)*p
            a += r
        p = i
        a -= 1
    
def fsamp(f,s=[(-1,.5),(1,.5)],filt=None,r=48000):
    if filt == None:
        filt = contRAvgFilt(1/r)
    a = 0
    i = 0
    if type(f)==int or type(f)==float:
        def g(v):
            while 1:
                yield v
        f = g(f)
    filtered = 0
    while 1:
        t = next(f)/r
        while t > 0:
            dt = min(t,s[i][1]-a)
            
            a += dt
            t -= dt
            filt.send(s[i][0],dt)

            if a>=s[i][1]:
                a -= s[i][1]
                i = (i+1)%len(s)

        
        yield filt.value
        
    





#actual fm stuff
from filters import IIR
import numpy as np


def phaseModulate(g,d=.1,f=10000,sr=48000):
    t = 0
    for i in g:
        t += f/sr
        yield nsin(t+i.real*d)+1j*(nsin(t+.25+i.imag*d))
def modulate(g,d=0.01,f=10000,sr=48000):
    t = .25
    for i in g:
        t += (d*i+1+1j)*f/sr
        yield (nsin(t.real)+1j*nsin(t.imag))

"""def stereoEncode(g,c=10000,sr=48000):
    t = 0
    flt = IIR()
    flt.setPolys([1],
    for i in g:
        t += (c+i.imag)/sr
        yield nsin(t)+i.real


def stereoDecode(g,c=15000,sr=48000):
    for i in g:
        r = 
"""
