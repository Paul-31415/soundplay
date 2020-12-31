#Polynomials stuff
from bisect import bisect_right
import math 

brailleBase = ord("⠀")

def xorfold(n,b=8):
    r = 0
    assert n >= 0
    while n:
        r ^= n&((1<<b)-1)
        n >>= b
    return r
class Unique:
    def __init__(self,n):
        self.name = n
    def __repr__(self):
        return f"{self.name}{chr(brailleBase+xorfold(hash(self)))}"
    def __eq__(self,o):
        return self is o
    def __hash__(self):
        return hash(id(self))
def polyvars(varstr):
    return [Polynomial(i) for i in varstr]
class Polynomial:
    def __init__(self,coef,var="x"):
        if type(coef) == str or type(coef) == Unique:
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
            v = x*0
            xa = 1
            for t in self.a:
                if type(t) == Polynomial:
                    t = t(vrs)
                v += xa*t
                xa *= x
            return v
        return Polynomial([t(vrs) if type(t) == Polynomial else t for t in self.a],self.var)
    def __getitem__(self,i):
        if type(i) == slice:
            start, stop, step = i.indices(len(self))
            res = Polynomial([],self.var)
            assert step != 0
            if step > 0:
                while start < stop:
                    res[start] = self[start]
                    start += step
            else:
                while start > stop:
                    res[len(self)-start-1] = self[start]
                    start += step
            return res
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
    def padd(self,o,oshift=0):
        res = []+self.a
        for j in range(len(o.a)):
            while j+oshift > len(res):
                res += [0]
            if j+oshift == len(res):
                res += [o.a[j]]
            else:
                res[j+oshift] += o.a[j]
        return Polynomial(res,self.var)
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
        res = []
        for i in range(len(self.a)):
            for j in range(len(o.a)):
                if i+j >= len(res):
                    res += [self.a[i]*o.a[j]]
                else:
                    res[i+j] += self.a[i]*o.a[j]
        return Polynomial(res,self.var)
    def npmul(self,o):
        return Polynomial([e*o for e in self.a],self.var)
    def __floordiv__(self,o):
        return divmod(self,o)[0]
    def __rfloordiv__(self,o):
        if type(o) == Polynomial:
            return o.__floordiv__(self)
        return Polynomial([o],self.var).__floordiv__(self)
    def __truediv__(self,o):
        if type(o) == Polynomial:
            return NotImplemented
        return self.npdiv(o)
    def npdiv(self,o):
        return Polynomial([e/o for e in self.a],self.var)
    def __mod__(self,o):
        return divmod(self,o)[1]
    def pfloordiv(self,o):
        return self.pdivmod(o)[0]
    def pmod(self,o):
        return self.pdivmod(o)[1]
    def __rdivmod__(self,o):
        if type(o) == Polynomial:
            return o.__divmod__(self)
        return Polynomial([o],self.var).__divmod__(self)
    def __divmod__(self,o):
        if type(o) == Polynomial:
            return self.simplified(o.var).pdivmod(o)
        return self.simplified(self.var).pdivmod(Polynomial([o],self.var))
    def pdivmod(self,o):
        quot = Polynomial([],self.var)
        rem = self+0
        for i in range(len(self)-len(o),-1,-1):
            dig = rem[-1]/o[-1]
            quot[i] = dig
            rem = rem.padd(o.npmul(-dig),i)
        return quot,rem
    def npfloordiv(self,o):
        return Polynomial([e//o for e in self.a],self.var)
    def nptruediv(self,o):
        return Polynomial([e/o for e in self.a],self.var)
    def __lshift__(self,n):
        if type(n) != int:
            return NotImplemented
        return Polynomial([0]*n+self.a,self.var)
    def __rshift__(self,n):
        if type(n) != int:
            return NotImplemented
        return Polynomial(self.a[n:],self.var)
    def __pow__(self,n):
        if type(n) != int or n < 0 or (self == 0 and n == 0):
            return NotImplemented
        acc = Polynomial([1],self.var)
        for d in bin(n)[2:]:
            acc *= acc
            if d == '1':
                acc *= self
        return acc
    #def __repr__(self,var=None):
    #    if var == None:
    #        var = self.var
    #    return f"polyn({var}) = "+" + ".join((f"({self.a[i]})"+["",f"{var}"][i>0]+["",f"**{i}"][i>1] for i in range(len(self.a))))
    def __repr__(self,var=None):
        if var == None:
            var = self.var
        if len(self) == 0:
            return "(0)"
        return "("+f"p({var})="*0+" + ".join(((f"{self.a[i]}" if self.a[i] != 1 else ["1",""][i>0])+["",f"{var}"][i>0]+["",f"**{i}"][i>1] for i in range(len(self.a)) if self.a[i] != 0))+")"
    def deriv(self):
        return Polynomial([self.a[i+1]*(i+1) for i in range(len(self.a)-1)],self.var)
    def integ(self,k=0):
        return Polynomial([k]+[self.a[i]*(1/(i+1)) for i in range(len(self.a))],self.var)
    def rconvolve(self,o):
        #integ of self(t-x)o(t) dt
        #so, 
        x = Polynomial(self.var)
        t = Polynomial(Unique('t'))
        integrand = self(t-x)*o.simplified(self.var)(t)
        return integrand.simplified(t.var).integ()
    def convolve(self,o):
        #integ of self(t)o(t-x) dt
        x = Polynomial(self.var)
        t = Polynomial(Unique('t'))
        integrand = self(t)*o.simplified(self.var)(t-x)
        return integrand.simplified(t.var).integ()
    def __len__(self):
        return len(self.a)
    def __hash__(self):
        if len(self)>1:
            return hash(self.var)^hash(self(1))
        return hash(self(1))
    def __eq__(self,o):
        if type(o) == Polynomial:
            if len(o) != len(self):
                return False
            if len(self) > 1 and o.var != self.var:
                return False
            for i in range(len(self)):
                if self.a[i] != o.a[i]:
                    return False
            return True
        if type(o) == float or type(o) == int or type(o) == complex:
            return len(self) <= 1 and (self.a+[0])[0] == o
    def __matmul__(self,o):
        return self.convolve(o)
    def __rmatmul__(self,o):
        return self.rconvolve(o)

    #convienence methods
    def s(self,o):
        if type(o) == Polynomial:
            return self.simplified(o.var)
        return self.simplified(o)
    def v(self):
        return Polynomial(self.var)
    

    
    def plot(self,t0=-1,t1=1,res=50):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(nrows=1, ncols=1)
        st = self(t0)
        end = self(t1)
        mid = []
        ts = []
        if len(self) > 2:
            for j in range(1,res):
                t = t0+((t1-t0)*j/res)
                ts += [t]
                mid += [self(t)]
        ts = [t0]+ts+[t1]
        ys = [st]+mid+[end]
        plt.plot(ts,[i.real for i in ys],linestyle='-',color=(.3,.3,1), linewidth=2)
        plt.plot(ts,[i.imag for i in ys],linestyle='-',color=(1,.3,.3), linewidth=2)
        plt.show(block=0)
        return plt


def makePeicewizePolynomial(o,var="x"):
    if type(o) == PeicewizePolynomial:
        return o
    if type(o) == Polynomial:
        return PeicewizePolynomial([o],[])
    return PeicewizePolynomial([Polynomial([o],var)],[])
class PeicewizePolynomial:
    #defined as 0 outside defined regions
    def __init__(self,polys,times,var=None):
        #each polynomial is defined with x starting at the middle of its region (or the finite end)
        if var == None:
            var = polys[0].var
        self.polys = [p.s(var) for p in polys]
        self.times = times
        self.var = var
    def polyIndex(self,t):
        return bisect_right(self.times,t)
    def __call__(self,t):
        if type(t) == Polynomial or type(t) == PeicewizePolynomial:
            return self.compose(makePeicewizePolynomial(t))
        l = self.polyIndex(t)
        if len(self.times) == 0:
            return self.polys[l](t)
        if l == 0:
            return self.polys[l](t-self.times[0])
        if l == len(self.times):
            return self.polys[l](t-self.times[-1])
        return self.polys[l](t-(self.times[l]+self.times[l-1])/2)
    def plot(self,lo=None,hi=None,res=50):
        if lo == None:
            lo = -1
            if len(self.times):
                lo = self.times[0]-1
        if hi == None:
            hi = 1
            if len(self.times):
                hi = self.times[-1]+1
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(nrows=1, ncols=1)
        dash = 0
        times = self.times + [hi]
        rels = self.centers()
        for i in range(len(self.polys)):
            dash = 1-dash
            if times[i] < lo:
                continue
            t0 = lo
            t1 = times[i]
            if t1 > hi:
                t1 = hi
            lo = t1
            z = rels[i]
            p = self.polys[i]
            rv = 1 if len(p) <= 2 else res
            ys = []
            ts = []
            for j in range(rv+1):
                t = (t1-t0)*j/rv + t0
                ts += [t]
                ys += [p(t-z)]
            plt.plot(ts,[i.real for i in ys],linestyle='-',color=(.5*dash,.5*dash,1), linewidth=2)
            plt.plot(ts,[i.imag for i in ys],linestyle='-',color=(1,.5*dash,.5*dash), linewidth=2)
            if t1 == hi:
                break
        plt.show(block=0)
    def __lshift__(self,t):
        return PeicewizePolynomial(self.polys+[],[t+time for time in self.times],self.var)
    def __rshift__(self,t):
        return PeicewizePolynomial(self.polys+[],[time-t for time in self.times],self.var)
    def __invert__(self):
        return PeicewizePolynomial([p(-p.v()) for p in self.polys[::-1]],[-time for time in self.times[::-1]],self.var)
    def __neg__(self):
        return PeicewizePolynomial([-p for p in self.polys],self.times+[],self.var)
    def centers(self,times=None):
        if times == None:
            times = self.times
        if len(times):
            return [times[0]] + [(times[i]+times[i+1])/2 for i in range(len(times)-1)] + [times[-1]]
        else:
            return [0]
    def regions(self,times=None):
        if times == None:
            times = self.times
        if len(times):
            return [(-math.inf,times[0])] + [(times[i],times[i+1]) for i in range(len(times)-1)] + [(times[-1],math.inf)]
        else:
            return [(-math.inf,math.inf)]
    def relregions(self,times=None):
        if times == None:
            times = self.times
        if len(times):
            return [(-math.inf,0)] + [(lambda x: (-x/2,x/2))(times[i+1]-times[i]) for i in range(len(times)-1)] + [(0,math.inf)]
        else:
            return [(-math.inf,math.inf)]
    def pairs(self,o):
        if len(self.times) == 0:
            res_t = o.times+[]
            rc = self.centers(res_t)
            rp = []
            ply = self.polys[0]
            for i in range(len(rc)):
                rp += [ply(ply.v()+rc[i])]
            return rp,o.polys+[],res_t
        if len(o.times) == 0:
            return o.pairs(self)
        si = 0
        oi = 0
        st = self.times + [math.inf]
        ot = o.times + [math.inf]
        res_t = []
        sc = self.centers()
        oc = o.centers()
        sp = []
        op = []
        while oi < len(ot) and si < len(st):
            stc,otc = st[si],ot[oi]
            spl,opl = self.polys[si],o.polys[oi]
            center = min(stc,otc) if len(res_t) == 0 else (min(stc,otc)+res_t[-1])/2
            sp += [spl(spl.v()-sc[si]+center)]
            op += [opl(opl.v()-oc[oi]+center)]
            if stc < otc:
                res_t += [stc]
                si += 1
            elif stc > otc:
                res_t += [otc]
                oi += 1
            else:
                res_t += [stc]
                si += 1
                oi += 1
        sp[-1] = spl(spl.v()-sc[-2]+res_t[-2])
        op[-1] = opl(opl.v()-oc[-2]+res_t[-2])
        return sp,op,res_t[:-1]
    def __add__(self,o):
        s,o,t = self.pairs(makePeicewizePolynomial(o,self.var))
        return PeicewizePolynomial([s[i]+o[i] for i in range(len(s))],t,self.var)
    def __radd__(self,o):
        s,o,t = self.pairs(makePeicewizePolynomial(o,self.var))
        return PeicewizePolynomial([o[i]+s[i] for i in range(len(s))],t,self.var)
    def __sub__(self,o):
        s,o,t = self.pairs(makePeicewizePolynomial(o,self.var))
        return PeicewizePolynomial([s[i]-o[i] for i in range(len(s))],t,self.var)
    def __rsub__(self,o):
        s,o,t = self.pairs(makePeicewizePolynomial(o,self.var))
        return PeicewizePolynomial([o[i]-s[i] for i in range(len(s))],t,self.var)
    def __mul__(self,o):
        s,o,t = self.pairs(makePeicewizePolynomial(o,self.var))
        return PeicewizePolynomial([s[i]*o[i] for i in range(len(s))],t,self.var)
    def __rmul__(self,o):
        s,o,t = self.pairs(makePeicewizePolynomial(o,self.var))
        return PeicewizePolynomial([o[i]*s[i] for i in range(len(s))],t,self.var)

    def deriv(self):
        return PeicewizePolynomial([p.deriv() for p in self.polys],self.times+[],self.var)
    def integ(self,yint=0):
        rp = [p.integ() for p in self.polys]
        #adjust down and up
        centers = self.centers()
        zi = self.polyIndex(0)
        rp[zi] += yint-rp[zi](0-centers[zi])
        for i in range(zi+1,len(self.polys)):
            end = rp[i-1](self.times[i-1]-centers[i-1])
            start = rp[i](self.times[i-1]-centers[i])
            rp[i] += end-start
        for i in range(zi-1,-1,-1):
            end = rp[i+1](self.times[i]-centers[i+1])
            start = rp[i](self.times[i]-centers[i])
            rp[i] += end-start
        return PeicewizePolynomial(rp,self.times+[],self.var)


    def compose(self,o):
        assert False
        res_t = []
        res_p = []
        #for 
    


    
    def __matmul__(self,o):
        return self.convolve(makePeicewizePolynomial(o))
        
    def convolve(self,o):
        ts = self.centers()
        to = o.centers()
        rs = self.relregions()
        ro = o.relregions()
        result = PeicewizePolynomial([self.polys[0]*0],[],self.var)
        for i in range(len(self.polys)):
            for j in range(len(o.polys)):
                pc = convPolyFrags(self.polys[i],*rs[i],o.polys[j],*ro[j])
                result += pc<<(ts[i]-to[j])
        return result

def denan(v,d=0):
    return d if math.isnan(v) else v
def PeicewizePolynomialAbout0(polys,times,var=None):
    while len(times) and times[0] == -math.inf:
        times = times[1:]
        polys = polys[1:]
    while len(times) and times[-1] == math.inf:
        times = times[:-1]
        polys = polys[:-1]
    r = PeicewizePolynomial(polys,times,var)
    c = r.centers()
    for i in range(len(c)):
        r.polys[i] = r.polys[i](r.polys[i].v()+denan(c[i]))
    return r
def PeicewizePolynomialCullNans(polys,times,var=None):
    while len(times) and times[0] == -math.inf:
        times = times[1:]
        polys = polys[1:]
    while len(times) and times[-1] == math.inf:
        times = times[:-1]
        polys = polys[:-1]
    r = PeicewizePolynomial(polys,times,var)
    return r
def convPolyFrags(p0,l0,h0,p1,l1,h1):
    if p0 == 0 or p1 == 0:
        return PeicewizePolynomial([p0*0],[],p0.var)
    c = p0@p1
    #p1 is the moving one
    x = p0.v()
    # start: where h1-x > l0   (p1 enters p0)
    #   mid: where h1-x > h0   (p0 in p1)
    #   mid: where l1-x > l0   (p1 in p0)
    #   end: where l1-x > h0   (p1 is out of p0)
    p = [x*0]
    xes = []
    
    xes += [h1-l0]
    p += [(c(h1-x-l1)-c(l0))(x)] #entering
    
    if h0-l0 > h1-l1:#in
        #   l1-l0>h1-h0
        
        #p1 can be entirely in p0
        #l0 < l1-x < h1-x < h0
        #l0-l1 < -x    -x < h0-h1
        #l1-l0 > x > h1-h0
        #region: ƒ from l1-x to h1-x
        xes += [l1-l0]
        p += [(c(h1-x)-c(l1-x))(x)]
        xes += [h1-h0]
    elif h0-l0 < h1-l1:
        #p0 can be entirely in p1
        xes += [h1-h0]
        p += [(c(h0)-c(l0)+(0*x))(x)]
        xes += [l1-l0]
    else:
        xes += [l1-l0]
    p += [(c(h0)-c(l1-x-h1))(x+l1)] #exiting    
    xes += [l1-h0]
    p += [x*0]
    
    return ~PeicewizePolynomialAbout0(p[::-1],xes[::-1],x.var)


def gaussianApprox(mean=0,variance=1,n=3,var='x'):
    x = Polynomial(var)
    kern = PeicewizePolynomial([x*0,x*0+(1/variance),x*0],[-variance/2,variance/2],var)
    acc = kern+0
    for d in bin(n)[3:]:
        acc @= acc
        if d == '1':
            acc @= kern
    return acc << mean

def sampleAndHoldApprox(dat,step=1,var='x'):
    return PeicewizePolynomial([Polynomial([v],var) for v in [0]+dat+[0]],[i*step for i in range(len(dat)+1)],var)
