#file for pitch estimation, pitch changing, other vocoding stuff


#idea: take a simple rc-relaxation pll
# then run it on the audio mixed up to a higher frequency band where
# the span is much less than an octave
# then pll to the zero crossings and get the pitch??
#


import math
import numpy as np
import scipy.signal
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
        self.kern = np.array([(lambda x: [x,x])(sp.special.sinc(i/kf-l/kf/2)*gaussian((i/(l-1)-.5)*wl)) for i in range(l)],dtype=float)/kf
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

    def implot(self,g,s=1):
        im = []
        i = 0
        a = np.copy(self.cb)
        for v in g:
            self(v)
            i += s
            if i >= 1:
                i -= 1
                im += [a]
                a = np.copy(self.cb)
            else:
                a += self.cb
        from matplotlib import pyplot as plt
        p1 = plt.subplot(2,1,1)
        p2 = plt.subplot(2,1,2)
        il = p1.imshow(np.array([i[:,0] for i in im]).T,aspect='auto')
        ir = p2.imshow(np.array([i[:,1] for i in im]).T,aspect='auto')
        plt.show(block=0)
        return 
    

        
class autocor_cascade:
    def __init__(self,num=8,*cargs):
        self.cors = [autocor(*cargs) for i in range(num)]
    def __call__(self,v):
        i = 0
        while v is not None and i < len(self.cors):
            v = self.cors[i](v)
            i += 1
        return v
    def pa(self):
        for i in range(len(self.cors)):
            yield self.cors[i].cb[(i!=0)*(len(self.cors[i].cb)//2):]
    def implot(self,g,s=1):
        im = []
        i = 0
        a = np.concatenate([c for c in self.pa()])
        for v in g:
            self(v)
            i += s
            if i >= 1:
                i -= 1
                im += [a]
                a = np.concatenate([c for c in self.pa()])
            else:
                a += np.concatenate([c for c in self.pa()])
        from matplotlib import pyplot as plt
        p1 = plt.subplot(2,1,1)
        p2 = plt.subplot(2,1,2)
        il = p1.imshow(np.array([i[:,0] for i in im]).T,aspect='auto')
        ir = p2.imshow(np.array([i[:,1] for i in im]).T,aspect='auto')
        plt.show(block=0)
        return



    

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

class window_resampler:
    def __init__(self,thing,size=4096,outsize=8192,window=lambda x:math.cos(x*math.pi)/2+1,mult=2):
        self.times=mult
        self.window=np.array([window((i/(size-1))*2-1) for i in range(size)],dtype=complex)
        self.buf=np.zeros(size,dtype=complex)
        self.windowed=np.zeros(size,dtype=complex)
        self.outbuf=np.zeros(outsize,dtype=complex)
        self.i=0
        self.oi = 0
        self.cb = thing
    def __call__(self,v):
        #put into buf
        l = len(self.buf)
        if ((self.i*self.times)%l)+self.times >= l:
            #transform
            self.windowed[:l-self.i] = self.buf[self.i:]*self.window[:l-self.i]
            self.windowed[l-self.i:] = self.buf[:self.i]*self.window[l-self.i:]
            n = (self.i*self.times)//l

            ol = len(self.outbuf)
            oi = (self.oi*ol)//l
            self.outbuf[:ol-oi] = self.outbuf[oi:]
            self.outbuf[ol-oi:] = 0
            self.oi = 0
            self.outbuf += self.cb(self.windowed)
            self.buf[self.i] = v
            self.i = (self.i+1)%len(self.buf)
            self.oi += 1
            return self.outbuf[:oi]
        #put into buf
        self.buf[self.i] = v
        self.i = (self.i+1)%len(self.buf)
        self.oi += 1
        return None
    def gen(self,g):
        for v in g:
            yield self(v)
    def fgen(self,g):
        yield from flat(self.gen(g))
class window_func:
    def __init__(self,thing,size=256,window=lambda x:math.cos(x*math.pi)/2+1,mult=2):
        self.times=mult
        self.window=np.array([window((i/(size-1))*2-1) for i in range(size)],dtype=complex)
        self.buf=np.zeros(size,dtype=complex)
        self.windowed=np.zeros(size,dtype=complex)
        self.i=0
        self.cb = thing
    def __call__(self,v):
        #put into buf
        l = len(self.buf)
        if ((self.i*self.times)%l)+self.times >= l:
            #transform
            self.windowed[:l-self.i] = self.buf[self.i:]*self.window[:l-self.i]
            self.windowed[l-self.i:] = self.buf[:self.i]*self.window[l-self.i:]

            self.buf[self.i] = v
            self.i = (self.i+1)%len(self.buf)
            return self.cb(self.windowed)
        #put into buf
        self.buf[self.i] = v
        self.i = (self.i+1)%len(self.buf)
    def gen(self,g):
        for v in g:
            r = self(v)
            if r is not None:
                yield r
    
class fourier_max_note_estimator:
    def __init__(self,size=1<<12,*args):
        self.w = window_func(self.cb,size,*args)
    def cb(self,b):
        f = sp.fft.fft(b)
        return ftft(np.argmax(abs(f)),len(f))
    def __call__(self,g):
        yield from self.w.gen(g)
fmne = fourier_max_note_estimator
    
def halfshift(a):
    return np.concatenate((a[len(a)//2:],a[:len(a)//2]))
class ft_rspace:
    def __init__(self,thing,size=256,outsize=512,*args):
        self.windower = window_resampler(self.cbm,size,outsize,*args)
        self.cb = thing
    def cbm(self,b):
        return halfshift(sp.fft.ifft(self.cb(sp.fft.fft(halfshift(b)))))
    def __call__(self,v):
        return self.windower(v)
class ft_space_func:
    def __init__(self,thing,size=256,*args):
        self.windower = window_func(self.cbm,size,*args)
        self.cb = thing
    def cbm(self,b):
        return self.cb(sp.fft.fft(halfshift(b)))
    def __call__(self,v):
        return self.windower(v)
    def gen(self,g):
        yield from self.windower.gen(g)
class ft_re2:
    def __init__(self,p=16):
        self.p = p
    def __call__(self,f):
        r = np.concatenate((f,f))
        s = r.reshape((self.p*2,len(f)//self.p))
        s[:self.p,:] = f.reshape((self.p,len(f)//self.p))
        s[self.p:,:] = f.reshape((self.p,len(f)//self.p))
        return r
class ft_ree:
    def __init__(self,p=.05,sf=2):
        self.p = p
        self.sf = sf
    def __call__(self,f):
        l = len(f)
        r = np.zeros(int(self.sf*l),dtype=complex) 
        r[0] = f[0]
        cs = 1
        fi = 1
        while cs < l//2:
            c = min(int(cs),l//2-fi)
            ri = int(fi*self.sf)
            r[ri:ri+c] = f[fi:fi+c]
            r[-ri-c:-ri] = f[-fi-c:-fi]
            fi += c
            cs += cs*self.p
        return r

def ftft(f,s):
    return ((f+s//2)%s)-(s//2)

def expround(a,r=1/12,o=0):
    return np.sign(a)*np.exp2((np.round(np.log2(np.abs(a)+.001)/r-o)+o)*r)

class ft_rep:
    def __init__(self,sf=2,ffunc = lambda f,sf,l,ff: f*sf):
        self.sf = sf
        self.ff = ffunc
    def __call__(self,f,ffov=None):
        ff = self.ff if ffov is None else ffov
        l = len(f)
        p = int(self.sf*l)
        r = np.zeros(p,dtype=complex)
        a = np.abs(f)
        pk = np.concatenate((np.array([0]),sp.signal.find_peaks(a)[0],np.array([l])))
        adj = np.diff(pk)
        rpk = ff(pk,self.sf,l,f).astype(int)%p
        inds = np.ones(l,dtype=int)
        inds[0] = 0
        np.add.at(inds,pk[:-1]+adj//2,np.diff(rpk)-adj)
        np.add.at(r,np.cumsum(inds,out=inds)%p,f)
        return r
        #old loop version:
        pk = np.concatenate((np.array([0]),sp.signal.find_peaks(a)[0],np.array([l])))
        for i in range(1,len(pk)-1):
            hi = (pk[i+1]-pk[i])//2
            lo = (pk[i-1]-pk[i])//2
            ce = int(pk[i]*self.sf)
            r[lo+ce:hi+ce] = f[lo+pk[i]:hi+pk[i]]
        return r
class ft_rep_p:
    def __init__(self,filt=lambda f: np.abs(f),*a):
        self.f = filt
        self.a = a
    def __call__(self,f):
        l = len(f)
        a = self.f(f)
        pk = np.concatenate((np.array([0]),sp.signal.find_peaks(a,*self.a)[0],np.array([l])))
        return pk
def regions_stitch(b,d,p1,p2):
    p1 = p1.astype(int)
    p2 = p2.astype(int)
    adj = np.diff(p1)
    inds = np.ones(len(b),dtype=int)
    inds[0] = 0
    np.add.at(inds,p1[:-1]+adj//2,np.diff(p2)-adj)
    np.add.at(d,np.cumsum(inds,out=inds)%len(d),b)
    return d
def regions_stitch_triangular(b,d,p1,p2):
    dists = np.zeros(len(b),dtype=float)
    adj = np.diff(p1)
    dists[p1[:-1]] = np.diff(np.concatenate((np.zeros(1),adj)))
    np.cumsum(dists,out=dists)

    #every_other = np.zeros(len(b),dtype=int)
    #np.add.at(every_other,p1,1)
    #np.cumsum(every_other,out=every_other)
    #every_other = (every_other&1)==1
    
    triangles = 1/(dists+(dists==0)/2)
    np.cumsum(triangles,out=triangles)
    triangles %= 2
    triangles -= 2*(triangles-1)*(triangles>1)
    #p1[::2] is at trophs
    rtriangles = 1-triangles

    #triangles,rtriangles = rtriangles,triangles
    
    inds1 = np.ones(len(b),dtype=int)
    inds1[0] = 0
    np.add.at(inds1,p1[::2][:-1],np.diff(p2[::2])-np.diff(p1[::2]))
    np.add.at(d,np.cumsum(inds1,out=inds1)%len(d),b*triangles)

    inds2 = np.ones(len(b),dtype=int)
    inds2[0] = 0
    np.add.at(inds2,p1[1::2][:-1],np.diff(p2[1::2])-np.diff(p1[1::2]))
    np.add.at(d,np.cumsum(inds2,out=inds2)%len(d),b*rtriangles)
    return d

class fpa:
    def __init__(self,size,wf = lambda x:math.cos(x*math.pi)/2+1):
        self.window = np.array([wf((i/(size-1))*2-1) for i in range(size)],dtype=complex)
    def __call__(self,d):
        a = np.abs(sp.fft.fft(self.window*d))
        return np.concatenate((np.array([0]),sp.signal.find_peaks(a)[0],np.array([len(d)])))
class ft_re:
    def __init__(self,size=1<<12,sf=1,mf = lambda a,sf:a*sf,sp=1):
        if type(sp) is not int and type(sp) is not float:
            self.pks = sp
        else:
            self.pks = fpa(size,lambda x:math.cos(x*sp*math.pi)/2+1 if abs(x*sp) < 1 else 0)
        self.sf = sf
        self.mf = mf
    def __call__(self,d):
        f = sp.fft.fft(d)
        a = self.pks(d)
        rs = int(len(d)*self.sf)
        r = np.zeros(rs,dtype=complex)
        regions_stitch(f,r,a,self.mf(a,self.sf).astype(int))
        return sp.fft.ifft(r)
    def gp(self,d):
        return self.pks(d)


def tf_bufs(src,dst):
    f = sp.fft.fft(src)
    df = sp.fft.fft(dst)
    r = np.zeros(len(dst),dtype=complex)
    a = sp.signal.find_peaks(np.abs(f))[0]
    da = sp.signal.find_peaks(np.abs(df))[0]
    regions_stitch(f,r,a,round_aa(a,da))
    return sp.fft.ifft(r)
def tf_bufs_qs(src,dst,qa=0,qb=0):
    f = sp.fft.fft(src)
    df = sp.fft.fft(dst)
    r = np.zeros(len(dst),dtype=complex)
    a = quantile_lookups(np.abs(f),sp.signal.find_peaks(np.abs(f))[0],qa)
    da = quantile_lookups(np.abs(df),sp.signal.find_peaks(np.abs(df))[0],qb)
    regions_stitch(f,r,a,round_aa(a,da))
    return sp.fft.ifft(r)

    
    
def quantile_lookups(f,p,q):
    if len(p) == 0:
        return p
    vals = f[p]
    return p[vals>np.quantile(vals,q)]
def numile_lookups(f,p,q):
    if len(p) <= q:
        return p
    vals = f[p]
    return p[vals>np.quantile(vals,1-q/len(p))]

def mt_s(to,s1=1<<12,s2=1<<12,sp1=1,sp2=1,alpha=1):
    f = ft_re(s1,sp=sp2)
    ng = window_func(lambda x: f.gp(x),s1).gen(to)
    return window_resampler(ft_re(s1,s2/s1,lambda a,sf:a*(1-alpha)+alpha*round_aa(a,next(ng)),sp1),s1,s2)
def round_aa(v,r,*a):
    return r[np.searchsorted(r[1:]+r[:-1],v*2)]
def fourier_fmult(f,m=1,l=1<<12):
    if m == 1:
        return f
    return ((f-l*(f>=l/2))*m)%l
def round_aa_with_fmults_curry(mults = [1],l=1<<12):
    import sortednp
    def round_aa_mults(v,r,*ar):
        a = np.array([])
        for f in mults:
            a = sortednp.merge(fourier_fmult(r,f,l),a)
        return round_aa(v,a,*ar)
    return round_aa_mults
        
        
def rm__map_sorted(p,pt,a,at):
    sind = np.argsort(a[p[1:-2].astype(int)])+1 #a[p][sind] is sorted
    sindt = np.argsort(at[pt[1:-2].astype(int)])+1 #at[pt][sindt] is sorted
    res = p.copy()
    if len(sindt) < len(sind):
        res[sind] = pt[np.resize(sindt,len(sind))]
    else:
        res[sind] = pt[sindt[:len(sind)]]
    return res

def mt_pqnfm(to,q1=32,q2=32,m=[1],s1=1<<12,s2=1<<12,st=1<<12):
    return mt_pqn(to,q1,q2,s1=s1,s2=s2,st=st,rounding_mode=round_aa_with_fmults_curry(m,s2))
def mt_pqn(to,q1=32,q2=32,m1=1,m2=1,alpha=1,s1=1<<12,s2=1<<12,st=1<<12,rounding_mode=round_aa,q_mode=numile_lookups):
    return mt_pq(to,q1,q2,m1,m2,alpha,s1,s2,st,rounding_mode,q_mode)
def mt_pq(to,q1=.98,q2=.98,m1=1,m2=1,alpha=1,s1=1<<12,s2=1<<12,st=1<<12,rounding_mode=round_aa,q_mode=quantile_lookups):
    return window_resampler(mt_pq_(to,q1,q2,m1,m2,alpha,s1,s2,st,rounding_mode,q_mode),s1,s2)
class mt_pq_:
    def __init__(self,to,q1=1,q2=1,m1=1,m2=1,alpha=1,s1=1<<12,s2=1<<12,st=1<<12,rounding_mode=round_aa,q_mode=quantile_lookups):
        self.to = window_func(lambda x:x,st).gen(to)
        self.s1 = s1
        self.s2 = s2
        self.st = st
        self.q1 = q1
        self.q2 = q2
        self.m1 = m1
        self.m2 = m2
        self.alpha = alpha
        self.rounding_mode = rounding_mode
        self.quant = q_mode
    def __call__(self,d):
        f = sp.fft.fft(d)
        a = abs(f)
        l = int(len(d)*self.s2/self.s1)
        r = np.zeros(l,dtype=complex)
        td = next(self.to)
        ft = sp.fft.fft(td)
        at = abs(ft)
        pt = sp.signal.find_peaks(at)[0]
        p = sp.signal.find_peaks(a)[0]
        qpt = fourier_fmult(self.quant(at,pt,self.q2),self.m2,len(at))
        pt = np.concatenate((np.array([0]),qpt,(np.array([self.st-1]))))*((l-1)/(self.st-1))
        qp = fourier_fmult(self.quant(a,p,self.q1),self.m1,len(a))
        p = np.concatenate((np.array([0]),qp,(np.array([l-1]))))
        rp = self.rounding_mode(p,pt,a,at)*self.alpha+p*(1-self.alpha)
        regions_stitch(f,r,p,rp%l)
        return sp.fft.ifft(r)

def nearest_resamp_aa(a,l):
    return a[(np.arange(l)*((len(a)-1)/(l-1))).astype(int)]

def round_force_resamp(p,pt,*a):
    return nearest_resamp_aa(pt,len(p))

def remap_peaks_ostat(p,pt,a,at):
    op = np.argsort(a[p])[::-1]
    opt = np.argsort(at[pt])[::-1]
    #remap p[op[i]] to pt[opt[i]]
    return pt[opt[op%len(opt)]]
    

    

def temposcale(s1=1<<12,s2=1<<12,spec=0):
    return window_resampler(ft_re(s1,s2/s1,sp=spec),s1,s2)
def pitchscale(p=1,s=1<<12,spec=0):
    p = genify(p)
    return window_resampler(ft_re(s,1,mf=lambda a,sp:next(p)*ftft(a,s),sp=spec),s,s)
class ft_repp:
    def __init__(self,sf=2,ffunc = lambda f,sf,l,ff: f*sf):
        self.sf = sf
        self.ff = ffunc
    def __call__(self,f,ffov=None):
        ff = self.ff if ffov is None else ffov
        l = len(f)
        p = int(self.sf*l)
        r = np.zeros(p,dtype=complex)
        a = np.abs(f)
        pk = sp.signal.find_peaks(a[:l//2]+a[-1:-l//2-1:-1])[0]
        pk = np.concatenate((np.array([0]),pk,l-pk[::-1],np.array([l])))
        adj = np.diff(pk)
        rpk = ff(pk,self.sf,l,f).astype(int)%p
        inds = np.ones(l,dtype=int)
        inds[0] = 0
        np.add.at(inds,pk[:-1]+adj//2,np.diff(rpk)-adj)
        np.add.at(r,np.cumsum(inds,out=inds)%p,f)
        return r
        #old loop version:
        pk = np.concatenate((np.array([0]),sp.signal.find_peaks(a)[0],np.array([l])))
        for i in range(1,len(pk)-1):
            hi = (pk[i+1]-pk[i])//2
            lo = (pk[i-1]-pk[i])//2
            ce = int(pk[i]*self.sf)
            r[lo+ce:hi+ce] = f[lo+pk[i]:hi+pk[i]]
        return r


def music_transfer(to,s1=1<<12,s2=1<<12,n=0,*a):
    f = ft_rep_p(*a)
    ng = ft_space_func(lambda x: f(afn(x,n)),s1).gen(to)
    return ft_rspace(ft_rep(s2/s1,lambda f,s,l,t: round_aa(f,next(ng))),s1,s2)


def af121(a):
    a[1:] += a[:-1]
    a[:-1] += a[1:]
    a /= 4
    return a
def afn(a,n=1):
    for i in range(n):
        af121(a)
    return a







class ft_note_sequencer:
    def __init__(self,seq,d=.9,th0=1000):
        self.g = seq
        self.v = next(self.g)
        self.d = d
        self.p = None
        self.t = th0
        self.t0 = th0
    def __call__(self,f):
        if self.p is None:
            self.p = np.abs(f)
        p = np.abs(f)
        t = np.sum(np.abs(self.p-p))/len(f)
        if t > self.t:
            self.t = self.t0
            self.v = next(self.g)
        self.p = p
        self.t *= self.d
        return self.v
    
def flat(g):
    for v in g:
        if v is None:
            continue
        if type(v) is complex:
            yield v
            continue
        try:
            yield from flat(v)
        except:
            yield v

    
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





def cplx_to_rgb(v):
    L = abs(v)
    a = v.real
    b = v.imag
    return np.array([L+a-b/2,L-a-b/2,L+b]).T

#next autocor thingy:
# have cyclical buffers that get the signal added to them
#  (they track the periodic components as well as the waveforms)
#
class cyclecor:
    def __init__(self,d=0.01,l=128):
        self.i = 0
        if type(d) is float:
            self.d = np.array([d]*l)
        self.d = d
        self.l = l
        self.cb = np.zeros((l*(l+1))//2,dtype=complex)
        self.inds = (lambda x: x*(x+1)//2) (np.arange(l))
        self.mods = np.arange(l)+1
    def __call__(self,v):
        i = (self.i%self.mods)+self.inds
        self.cb[i] *= 1 - self.d
        self.cb[i] += v * self.d * self.mods
        self.i += 1
        return self.cb[i]
    def implot(self,g,s=1):
        im = []
        i = 0
        a = self(next(g))
        for v in g:
            i += s
            if i >= 1:
                i -= 1
                im += [a]
                a = self(v)
            else:
                a += self(v)
        from matplotlib import pyplot as plt
        fig, axs = plt.subplots(nrows=2, ncols=1)
        ima = np.array(im).T
        il = axs[0].imshow(ima.real,aspect='auto')
        ir = axs[1].imshow(ima.imag,aspect='auto')
        plt.show(block=0)
        return 

def bbank_exp(l=240,df=-1/24,fstart=.5,q=10):
    r = np.arange(l)
    freqs = fstart*2**(df*r)
    poles = np.exp(2j*np.pi*freqs) * (1-freqs/q)
    gains = freqs/q
    #(1-po)(1-pq)
    #1-po-pq+popq
    z = np.zeros(l)
    return np.array([-poles.real*2,(poles*poles.conj()).real,gains,z,z]).T
    
class biquad_bank:
    def __init__(self,arr,state=None):
        self.c = arr
        if state is None:
            self.s = np.zeros((len(arr),2),dtype=complex)
    def __call__(self,v):
        res = v-self.s[:,0]*self.c[:,0]-self.s[:,1]*self.c[:,1]
        out = res*self.c[:,2]+self.s[:,0]*self.c[:,3]+self.s[:,1]*self.c[:,4]
        self.s[:,1] = self.s[:,0]
        self.s[:,0] = res
        return out
    def implot(self,g,s=1):
        im = []
        i = 0
        a = self(next(g))
        for v in g:
            i += s
            if i >= 1:
                i -= 1
                im += [cplx_to_rgb(a)]
                a = self(v)
            else:
                a += self(v)
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ima = np.swapaxes(np.array(im),0,1)
        il = ax.imshow(ima,aspect='auto')
        plt.show(block=0)
        return 
    

def expfs(l=240,df=-1/24,fstart=.5):
    r = np.arange(l)
    return fstart*2**(df*r)
    
class oscbank:
    def __init__(self,dts,decays,sines=None,coses=None):
        if sines is None:
            sines = np.zeros(len(dts),dtype=complex)
        if coses is None:
            coses = np.zeros(len(dts),dtype=complex)
        if type(decays) is float:
            decays = 1-dts*decays
        self.dts = dts
        self.s = sines
        self.c = coses
        self.d = decays
    def __call__(self,v):
        self.s += self.dts*self.c 
        self.c += v - self.dts*self.s
        self.s *= self.d
        self.c *= self.d
        return self.c
    def implot(self,g,s=1):
        im = []
        i = 0
        a = np.copy(self(next(g)))
        for v in g:
            i += s
            if i >= 1:
                i -= 1
                im += [cplx_to_rgb(a)]
                a = np.copy(self(v))
            else:
                a += self(v)
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ima = np.swapaxes(np.array(im),0,1)
        il = ax.imshow(ima,aspect='auto')
        plt.show(block=0)
        return

    
class sinbank:
    def __init__(self,dts,decays):
        if type(decays) is float:
            decays = 1-dts*decays
        self.f = np.exp(2j*np.pi*dts)
        self.p = np.ones(len(dts),dtype=complex)
        self.a = np.zeros(len(dts),dtype=complex)
        self.d = decays
    def __call__(self,v):
        self.a *= self.d
        self.a += self.p*v
        self.p *= self.f
        return self.a
    def implot(self,g,s=1):
        im = []
        i = 0
        a = np.copy(self(next(g)))
        for v in g:
            i += s
            if i >= 1:
                i -= 1
                im += [cplx_to_rgb(a)]
                a = np.copy(self(v))
            else:
                a += self(v)
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ima = np.swapaxes(np.array(im),0,1)
        il = ax.imshow(ima,aspect='auto')
        plt.show(block=0)


def iir1_bank_exp(n,f0=-.875,f1=-7.78,q0=-.875,q1=-7.78):
    c = np.exp(np.arange(n)*((f1-f0)/(n-1))+f0)
    g = np.exp(np.arange(n)*((q1-q0)/(n-1))+q0)
    return iir1_bank(np.array([np.concatenate((np.exp(2j*np.pi*c)*(1-g),np.exp(-2j*np.pi*c)*(1-g))),np.concatenate((g,g)),np.zeros(n*2)]))
    
class iir1_bank:
    def __init__(self,arr,state=None):
        self.c = arr
        if state is None:
            self.s = np.zeros(arr.shape[1],dtype=complex)
        else:
            self.s = state
    def __call__(self,v):
        s = self.s*self.c[2]
        self.s *= self.c[0]
        self.s += v
        return self.s*self.c[1]+s
    def implot(self,g,s=1):
        im = []
        i = 0
        a = self(next(g))
        for v in g:
            i += s
            if i >= 1:
                i -= 1
                im += [cplx_to_rgb(a)]
                a = self(v)
            else:
                a += self(v)
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ima = np.swapaxes(np.array(im),0,1)
        il = ax.imshow(ima,aspect='auto')
        plt.show(block=0)
        return 




    
class wavecutter:
    def __init__(self):
        #attempts to find the period of the wave and cut it there
        pass


#https://ieeexplore.ieee.org/abstract/document/8642626
def harms(b,i):
    ind = i
    while ind < len(b):
        yield b[ind]
        ind += i
def mtharms(b,i):
    a = [v for v in harms(b,i)]+[0]
    pv = a[0]
    pi = 0
    for i in range(1,len(a)):
        if a[i] < pv:
            for x in range(pi,i):
                al = (x-pi)/(i-pi)
                yield x,pv*(1-al)+a[i]*al
            pv = a[i]
            pi = i
class fnotes:
    def __init__(self,l=4096,t=.1,i0=16):
        self.b = np.zeros(l,dtype=complex)
        self.magbuf = np.zeros(l//2)
        self.i = 0
        self.window = np.cos((np.arange(l)/(l-1)-.5)*2*np.pi)*.5+1
        self.prevAns = []
        self.thresh = t/l
        self.i0 = i0
    def __call__(self,v):
        self.b[self.i] = v
        self.i = (self.i+1)%len(self.b)
        if self.i == 0:
            self.prevAns = []
            f = sp.fft.fft(self.b*self.window)
            self.magbuf[:] = abs(f[:len(f)//2])+abs(f[-1:-len(f)//2-1:-1])
            a = np.sum(self.magbuf[self.i0:len(f)//8])
            for i in range(self.i0,len(f)//8):
                if self.magbuf[i] > a*self.thresh:
                    ans = []
                    for h,v in mtharms(self.magbuf,i):
                        self.magbuf[h] -= v
                        ans += [v]
                    a = np.sum(self.magbuf[self.i0:len(f)//8])
                    self.prevAns += [(i/len(self.b),ans)]
        return self.prevAns


    
class notes_reconstructor:
    def __init__(self,fscale=1,wfunc=lambda x,h: (1+1j)*h[0]*np.sin(x*2*math.pi)):
        self.phases = [0]
        self.fscale = fscale
        self.wave = wfunc
    def __call__(self,notes):
        i = 0
        out = 0
        for n,h in notes:
            self.phases[i] += n*self.fscale
            out += self.wave(self.phases[i],h)
            i += 1
            if i >= len(self.phases):
                self.phases += [0]*len(self.phases)
        return out


    
class ftnotes:
    def __init__(self,t=.1,i0=16):
        self.i0 = i0
        self.t = t
    def __call__(self,f):
        magbuf = abs(f[:len(f)//2])+abs(f[-1:-len(f)//2-1:-1])
        for i in range(self.i0,len(f)//8):
            if magbuf[i] > self.t:
                ans = []
                for h,v in mtharms(magbuf,i):
                    fact = v/magbuf[h]
                    ans += [[fact*f[h],fact*f[-h]]]
                    magbuf[h] -= v
                yield (i,ans)
    def place(self,notes,f):
        for i,h in notes:
            hi = 0
            for j in range(i,len(f)//2,i):
                f[j] += h[hi][0]
                f[-j] += h[hi][1]
                hi += 1
                if hi >= len(h):
                    break
        return f

    
def fourier_note_transform(note=lambda x: x>.5,res=1024):
    return 2*sp.fft.fft(note(np.arange(res)/res))[1:res//2]/res
class fourier_note_components:
    def __init__(self,note_transform,stereo=True):
        self.nd = note_transform
        self.s = stereo
        #self.np = self.nd[1:len(self.nd)//2]
        #self.nn = self.nd[-1:-len(self.nd)//2:-1]
    def __call__(self,b):
        for i in range(1,len(b)//2):
            h = b[i:len(b)//2:i]
            h[0] /= self.nd[0]
            h[1:len(self.nd)] -= self.nd[1:len(h)]*h[0]
            h = b[-i:-len(b)//2:-i]
            h[0] /= self.nd[0]
            h[1:len(self.nd)] -= self.nd[1:len(h)]*h[0]
        return b
    def u(self,b):
        for i in range(len(b)//2-1,0,-1):
            h = b[i:len(b)//2:i]
            h[1:len(self.nd)] += self.nd[1:len(h)]*h[0]
            h[0] *= self.nd[0]
            h = b[-i:-len(b)//2:-i]
            h[1:len(self.nd)] += self.nd[1:len(h)]*h[0]
            h[0] *= self.nd[0]
        return b
    def matrix(self,blen):
        m = np.identity(blen,dtype=complex)
        if self.s:
            for i in range(1,blen//2):
                h = m[i:blen//2:i,i]
                h[:len(self.nd)] = self.nd[:len(h)]
                h = m[-i:-blen//2:-i,i]
                h[:len(self.nd)] = self.nd[:len(h)]
                h = m[i:blen//2:i,-i]
                h[:len(self.nd)] = -self.nd[:len(h)]
                h = m[-i:-blen//2:-i,-i]
                h[:len(self.nd)] = self.nd[:len(h)]
        else:
            for i in range(1,blen//2):
                h = m[i:blen//2:i,i]
                h[:len(self.nd)] = self.nd[:len(h)]
                h = m[-i:-blen//2:-i,-i]
                h[:len(self.nd)] = self.nd[:len(h)]
        return m
    def mx(self,blen):
        m = sp.sparse.csc_matrix(self.matrix(blen))
        return sp.sparse.linalg.inv(m),m



fnc = fourier_note_components
def genify(g):
    try:
        yield from g
    except:
        while 1:
            yield g

    
def ostat(order=lambda x: abs(x)):
    def cut(b,q=.5,order=order):
        o = order(b)
        b[o<np.quantile(o,q)] = 0
        return b
    return cut



class buftf:
    def __init__(self,l=1<<12,runner=lambda a,b:a):
        self.bl = l
        self.buf = np.zeros(l,dtype=complex)
        self.wbuf = np.zeros(l,dtype=complex)
        self.buf_i = 0
        self.obuf = np.zeros(l,dtype=complex)
        self.window = 1-.5*np.cos(np.arange(l)*(2*np.pi/l))
        self.run = runner
    def __call__(self,v):
        l = self.bl
        hl = l//2
        self.buf[self.buf_i+hl]=v
        self.buf_i += 1
        if self.buf_i >= hl:
            self.buf_i = 0
            self.obuf[:hl] = self.obuf[hl:]
            self.obuf[hl:] = 0
            self.wbuf[:] = self.buf[:]
            self.wbuf *= self.window
            self.run(self.wbuf,self.obuf)
            self.buf[:hl] = self.buf[hl:]
            self.buf[hl:] = 0
            return self.obuf[:hl]
class bufpk:
    def __init__(self,l=1<<12,runner=lambda a,b:a):
        self.bl = l
        self.obuf = np.zeros(l,dtype=complex)
        self.window = 1-.5*np.cos(np.arange(l)*(2*np.pi/l))
        self.run = runner
    def __call__(self,v):
        l = self.bl
        hl = l//2
        self.obuf[:hl] = self.obuf[hl:]
        self.obuf[hl:] = 0
        self.obuf += self.run(v)
        return self.obuf[:hl]
        



def fourier_harm_pick_inds(l,i,h):
    hl = (l//2)
    ll = min(h,(l-hl)//i)
    tl = min(h+1,(hl-1)//i)
    return (np.arange(ll+1+tl)-ll)*i
def fhpi_pos(l,i,h):
    p = fourier_harm_pick_inds(l,i,h)%l
    return np.sort(p)
def fourier_freq_vals(l):
    r = np.arange(l)
    r[r>=l//2] -= l
    return r
    
def harm_mat(l,harms=1<<16,wf = lambda f,a:np.ones(len(a))/len(a)):
    m = np.zeros((l,l//2))
    for i in range(l//2):
        inds = fourier_harm_pick_inds(l,i+1,harms)
        m[inds,i] = wf(i+1,inds)
    return sp.sparse.csc_matrix(m)
def hmg(gf=-1,bh=.9,norm=True):
    def do(f,a):
        r = (a!=0)*(bh**np.abs(a/f))
        if norm:
            r/=np.sum(r)
        return (f**gf)*r
    return do
def mag2(a):
    return a.imag*a.imag+a.real*a.real

def gwave1(t):
    return t<(2**.5)-1
def gwave2(t):
    c = (2**.5)-1
    return (t<c)*t/c+(t>=c)*(1-(t-c)/(1-c))
def gwave0(t):
    return (t==0)*len(t)

#def fourier_resize(f,l):
    
    


class fourier_filt:
    def __init__(self,spec):
        self.spec=spec
    def filt_f(self,ft):
        if len(self.spec) == len(ft):
            return self.spec*ft
        return sp.signal.resample(self.spec,len(ft))*ft
    def filt_a(self,b):
        return sp.fft.ifft(self.filt_f(sp.fft.fft(b)))
    


class vocode:
    def __init__(self,l=1<<12,wave=gwave1,weight=mag2,wf=lambda f,a:np.ones(len(a))/len(a),voices=1,harms=1<<16):
        self.wave = wave
        self.waveBuf = wave(np.arange(l)/l)
        self.waveFT = sp.fft.fft(self.waveBuf)
        self.bl = l
        self.buf = np.zeros(l,dtype=complex)
        self.wbuf = np.zeros(l,dtype=complex)
        self.buf_i = 0
        self.window = 1-.5*np.cos(np.arange(l)*(2*np.pi/l))
        self.harms = harms
        self.mat = harm_mat(l,harms,wf)
        self.weight_func = weight
        self.voices = voices
    def run(self,b):
        res = []
        l = self.bl
        ft = sp.fft.fft(b)
        for i in range(self.voices):
            w = self.weight_func(ft)@self.mat
            bf = np.argmax(w)+1
            band_inds = fhpi_pos(l,bf,self.harms)
            in_band = ft[band_inds]
            tone = self.waveFT[band_inds//bf]
            vfilt = in_band/tone
            ft[band_inds]=0
            res += [(bf,fourier_filt(vfilt))]
        return res
    def nur(self,a):
        l = self.bl
        ft = np.zeros(l,dtype=complex)
        for bf,vfilt in a:
            band_inds = fhpi_pos(l,bf,self.harms)
            tone = self.waveFT[band_inds//bf]
            ft[band_inds] += vfilt.filt_f(tone)
        return sp.fft.ifft(ft)
    def decoder(self):
        return bufpk(self.bl,self.nur)
    def __call__(self,v):
        l = self.bl
        hl = l//2
        self.buf[self.buf_i+hl]=v
        self.buf_i += 1
        if self.buf_i >= hl:
            self.buf_i = 0
            self.wbuf[:] = self.buf[:]
            self.wbuf *= self.window
            self.buf[:hl] = self.buf[hl:]
            self.buf[hl:] = 0
            return self.run(self.wbuf)
    def gen(self,sig):
        for v in sig:
            r = self(v)
            if r is not None:
                yield r
    def rgen(self,vsig):
        dc = self.decoder()
        for v in vsig:
            r = dc(v)
            yield from r

        

def fgaussb(l,sd):
    r = sd*fourier_freq_vals(l)/l
    r *= r
    return np.exp(-r)
class split_noise:
    def __init__(self,l=1<<12,t=4/3,sds = 4,nds = 16,pds=4,margin=0):
        self.bl = l
        self.buf = np.zeros(l,dtype=complex)
        self.wbuf = np.zeros(l,dtype=complex)
        self.buf_i = 0
        self.window = 1-.5*np.cos(np.arange(l)*(2*np.pi/l))
        self.spargs = {"prominence":t,"width":(0,32),"rel_height":3/4}
        self.smooth = fgaussb(l,sds)
        self.nsmooth = fgaussb(l,nds)
        self.psmooth = fgaussb(l,pds)
        self.margin = margin
        self.phase_t = 1
        self.norm_t = .001
    def set_sd(self,sds):
        self.smooth = fgaussb(self.bl,sds)
    def set_nd(self,nds):
        self.nsmooth = fgaussb(self.bl,nds)
    def set_pd(self,pds):
        self.psmooth = fgaussb(self.bl,pds)
    def run(self,b):
        ft = sp.fft.fft(b)
        pk = self.run_get_mask_from_peaks(*self.run_get_peaks(ft))
        sig = ft*pk
        noi = ft-sig
        return sig,noi
    def run_get_peaks(self,ft):
        aft = np.angle(ft)
        aft -= np.roll(aft,1)
        aft %= np.pi
        aft -= np.roll(aft,-1)
        aft += np.pi/2
        aft %= np.pi
        aft -= np.pi/2

        saft = 1+sp.fft.ifft(sp.fft.fft((2*aft/np.pi)**2)*self.psmooth).real*self.phase_t
        
        mft = mag2(ft)
        mft = sp.fft.ifft((mfft:=sp.fft.fft(mft))*self.smooth).real
        mfnf = sp.fft.ifft(mfft*self.nsmooth).real+self.norm_t
        mft /= mfnf+(mfnf==0)
        mft /= saft
        return sp.signal.find_peaks(mft,**self.spargs)
    def run_get_mask_from_peaks(self,pks,pps):
        l = self.bl
        lb = np.clip(np.round(pps['left_ips']-self.margin).astype(int),0,l-1)
        rb = np.clip(np.round(pps['right_ips']+self.margin).astype(int),0,l-1)
        pk = np.zeros(l,dtype=int)
        np.add.at(pk,lb,1)
        np.add.at(pk,rb,-1)
        pk = np.cumsum(pk)>0
        return pk
    
    def nur(self,a):
        l = self.bl
        ft = np.zeros(l,dtype=complex)
        sig,noi = a
        ft += sig
        ft += noi
        return sp.fft.ifft(ft)
    def decoder(self):
        return bufpk(self.bl,self.nur)
    def __call__(self,v):
        l = self.bl
        hl = l//2
        self.buf[self.buf_i+hl]=v
        self.buf_i += 1
        if self.buf_i >= hl:
            self.buf_i = 0
            self.wbuf[:] = self.buf[:]
            self.wbuf *= self.window
            self.buf[:hl] = self.buf[hl:]
            self.buf[hl:] = 0
            return self.run(self.wbuf)
    def gen(self,sig):
        for v in sig:
            r = self(v)
            if r is not None:
                yield r
    def rgen(self,vsig):
        dc = self.decoder()
        for v in vsig:
            r = dc(v)
            yield from r

    def plot_gen(self,g,frames=1<<11):
        g = self.gen(g)
        mn = np.zeros((self.bl,frames),dtype=complex)
        ms = np.zeros((self.bl,frames),dtype=complex)
        for i in range(frames):
            ms[:,i],mn[:,i] = next(g)
            print(f"{i}/{frames}      ",end="\r")
        from matplotThings import plotimgs
        plotimgs(ms,mn)
    

def snn_noise_temposcale(snn,ol=1<<15):
    fn = ol/snn.bl
    il = snn.bl
    def do(buf):
        #buf is noise
        pks,pkk = snn.run_get_peaks(buf)
        #nm = snn.run_get_mask_from_peaks(pks,pkk)
        sig = np.zeros(ol,dtype=complex)
        regions_stitch_triangular(buf,sig,pks,(fn*pks)%len(sig))
        return sig
                
    return window_resampler(do,il,ol)

    
def snn_for_irl():
    sn = split_noise(1<<14)
    sn.set_sd(2400)
    sn.set_nd(4800)
    sn.phase_t = 0
    sn.norm_t = 0
    sn.spargs = {'prominence': 0, 'width': 0, 'rel_height': 1}
    return sn
        

        

def sn_ptscale(sn,ol=1<<12,fs=1,fn=None,snn=None,ns=1,ss=1):
    try:
        fs*2
        fs = lambda x,f=fs: x*f
    except:
        pass
    if fn is None:
        fn = ol/sn.bl
    try:
        fn*2
        fn = lambda x,f=fn: x*f
    except:
        pass
    if snn is None:
        snn = sn
    il = sn.bl
    def do(buf):
        #separate sig and noise
        ft = sp.fft.fft(buf)
        sigmask = sn.run_get_mask_from_peaks(*sn.run_get_peaks(ft))
        noise = sp.fft.ifft(ft*(1-sigmask))
        m2 = (ft*ft.conjugate()).real
        pks,pkk = sp.signal.find_peaks(sigmask)
        sigf = np.zeros(ol,dtype=complex)
        regions_stitch(ft*sigmask,sigf,pks,fs(pks)%len(sigf))
        sig = sp.fft.ifft(sigf)*ss

        pks,pkk = snn.run_get_peaks(noise)
        #nm = snn.run_get_mask_from_peaks(pks,pkk)
        
        regions_stitch_triangular(noise*ns,sig,pks,fn(pks)%len(sig))
        return sig
        
        
    return window_resampler(do,il,ol)

        
        
class fake_sn_all_noise:
    def __init__(self,l=1<<12):
        self.bl = l
    def run_get_mask_from_peaks(self,a,b):
        return np.zeros(self.bl)
    def run_get_peaks(self,ft):
        return np.array([]),{}
class fake_sn_all_signal:
    def __init__(self,l=1<<12):
        self.bl = l
    def run_get_mask_from_peaks(self,a,b):
        return np.ones(self.bl)
    def run_get_peaks(self,ft):
        return np.array([]),{}
    
def sn_for_irl():
    sn = split_noise()
    sn.set_sd(3)
    sn.set_nd(64)
    sn.set_pd(16)
    sn.phase_t = 1
    sn.norm_t = 0
    sn.spargs = {'prominence': 1, 'width': (0, 32), 'rel_height': 0.8}
    return sn
def sn_for_chiptune():
    sn = split_noise()
    sn.set_sd(4)
    sn.set_nd(8)
    sn.set_pd(16)
    sn.phase_t = -.875
    sn.norm_t = 1
    sn.spargs = {'prominence': 1.1, 'width': (0, 12), 'rel_height': 0.875}
    return sn

class split_noise_p:#todo: this one should use phase and hopefully separate overlapping noise and signal
    def __init__(self,l=1<<12):
        pass


                

#todo next
#cepstrum like thingy for vocoding
# peaks(fft(lowPass(mag2(fft))))
# or peaks(fft( lowPass(mag2(fft))/lowerPass(mag2(fft)) )) like in split_noise


class cep_note:
    def __init__(self,l=1<<12,sds = 4,nds = 16,pds=4):
        self.bl = l
        self.buf = np.zeros(l,dtype=complex)
        self.wbuf = np.zeros(l,dtype=complex)
        self.buf_i = 0
        self.window = 1-.5*np.cos(np.arange(l)*(2*np.pi/l))
        self.smooth = fgaussb(l,sds)
        self.nsmooth = fgaussb(l,nds)
        self.psmooth = fgaussb(l,pds)
        self.phase_t = 0
        self.divoffset = .01
        self.hm = harm_mat(l)
    def set_sd(self,sds):
        self.smooth = fgaussb(self.bl,sds)
    def set_nd(self,nds):
        self.nsmooth = fgaussb(self.bl,nds)
    def set_pd(self,pds):
        self.psmooth = fgaussb(self.bl,pds)
    def run(self,b): 
        l = self.bl
        ft = sp.fft.fft(b)

        aft = np.angle(ft)
        aft -= np.roll(aft,1)
        aft %= np.pi
        aft -= np.roll(aft,-1)
        aft += np.pi/2
        aft %= np.pi
        aft -= np.pi/2

        saft = 1+sp.fft.ifft(sp.fft.fft((2*aft/np.pi)**2)*self.psmooth).real*self.phase_t
        
        mft = mag2(ft)
        mft = sp.fft.ifft((mfft:=sp.fft.fft(mft))*self.smooth).real
        mfnf = sp.fft.ifft(mfft*self.nsmooth).real
        mft /= mfnf+self.divoffset
        
        ct = sp.fft.fft(mft)
        ct[0] = 0
        rv = ct.real@self.hm
        
        return rv
    def __call__(self,v):
        l = self.bl
        hl = l//2
        self.buf[self.buf_i+hl]=v
        self.buf_i += 1
        if self.buf_i >= hl:
            self.buf_i = 0
            self.wbuf[:] = self.buf[:]
            self.wbuf *= self.window
            self.buf[:hl] = self.buf[hl:]
            self.buf[hl:] = 0
            return self.run(self.wbuf)
    def gen(self,sig):
        for v in sig:
            r = self(v)
            if r is not None:
                yield r
    def plot_gen(self,g,frames=1<<11,s=1,h=4):
        g = self.gen(g)
        m = np.zeros((self.bl//2,frames))#,dtype=complex)
        for i in range(frames):
            m[:,i] = next(g)
            print(f"{i}/{frames}      ",end="\r")
        from matplotThings import plotimgs
        plotimgs(np.clip(m,0,h))
        #plotimgs(m*s)



#todo next:
#  note picker algorithm
# use find_peaks to make it work in O(notes*harmonics) python steps instead of O(len)
#

#todo next:
#  multi transform compressor
# find fourier peaks and wavelet/chirplet peaks
# use linearity well
# perhaps make it take as args what kinda transforms to use
#
class transform:
    def __init__(self):
        pass
    def __call__(self,b):
        return np.copy(b)
    def inverse(self,b):
        return b
    def flatten(self,b):
        return b
    def unflatten(self,r):
        return r
class fourier(transform):
    def __init__(self):
        pass
    def __call__(self,b):
        return sp.fft.fft(b)
    def inverse(self,b):
        return sp.fft.ifft(b)
class harmier(transform):
    def __init__(self,l=1<<12,harms=[2**-i for i in range (16)]):
        mat = np.eye(l,dtype=complex)
        ha = np.array(harms)
        for i in range(1,l//2):
            ll = len(mat[i,i:min(i*(1+len(ha)),l//2):i])
            mat[i,i:min(i*(1+len(ha)),l//2):i] = ha[:ll]
            mat[-i,-i:-min(i*(1+len(ha)),l//2):-i] = ha[:ll]
        self.mat = sp.sparse.csc_matrix(mat)
        self.inv = sp.sparse.linalg.inv(self.mat)
    def __call__(self,b):
        return sp.fft.fft(b)@self.inv
    def inverse(self,b):
        return sp.fft.ifft(b@self.mat)
def harmier_phase_series(l,harms,step=1j,n=4):
    f = 1
    for i in range(n):
        h = []
        p = f
        for j in range(len(harms)):
            h += [p*harms[j]]
            p *= f
        yield harmier(l,h)
        f *= step

    
class gausslet(transform):
    def __init__(self,l=1<<12,width=16,sds=4):
        mat = np.eye(l,dtype=complex)
        gau = fgaussb(l,width)
        for i in range(l//width):
            
            for j in range(width):
                assert False
class sinclet(transform):
    def __init__(self,l=1<<12,width=16):
        self.width = width
        mat = np.eye(l,dtype=complex)
        basi = np.eye(width,dtype=complex)
        for i in range(width):
            z = np.zeros(width)
            z[i] = 1
            basi[i,:] = sp.fft.ifft(z)
        for i in range(l):
            b = i//width
            w = i%width
            mat[i,b:b+width] = basi[w]
        self.mat = sp.sparse.csc_matrix(mat)
        self.inv = sp.sparse.linalg.inv(self.mat)
    def __call__(self,b):
        return sp.fft.fft(b)@self.inv
    def inverse(self,b):
        return sp.fft.ifft(b@self.mat)
import pywt
class wavelet(transform):
    def __init__(self,l=1<<12,fam="haar"):
        self.wavelet = fam
        bz = np.zeros(l,dtype=complex)

        wt = pywt.wavedec(bz,self.wavelet)
        self.weights = [i*0 for i in wt]
        for i in range(len(wt)):
            for j in range(len(wt[i])):
                wt[i][:] = 0
                wt[i][j] = 1
                self.weights[i][j] = np.sum(mag2(pywt.waverec(wt,self.wavelet)))
        self.l = l
    def __call__(self,b):
        assert len(b) == self.l
        d = pywt.wavedec(b,self.wavelet)
        return [d[i]*self.weights[i] for i in range(len(d))]
    def inverse(self,b):
        if len(b) != len(self.weights):
            raise Exception("length mismatch",self.wavelet,b)
        return pywt.waverec([b[i]/self.weights[i] for i in range(len(b))],self.wavelet)
        return pywt.waverec(b,self.wavelet)
    def flatten(self,b):
        return np.concatenate(b)
    def unflatten(self,r):
        res = []
        i = 0
        for l in self.weights:
            res += [r[i:i+len(l)]]
            i += len(l)
        return res
        
        
    
class mulitbasis_compressor:
    def __init__(self,l=1<<12,bases=[transform()],q=.95,c=1):
        self.bases = bases
        self.bl = l
        self.buf = np.zeros(l,dtype=complex)
        self.wbuf = np.zeros(l,dtype=complex)
        self.buf_i = 0
        self.window = 1-.5*np.cos(np.arange(l)*(2*np.pi/l))

        self.quantile = q
        self.cycles = c
    def run(self,b): 
        l = len(self.bases)
        tfs = [base(b) for base in self.bases]
        accs = [self.bases[i].flatten(tfs[i]) for i in range(l)]
        totals = [b*0 for b in accs]
        for p in range(self.cycles):
            mb = 0
            m = -1
            mv = -1
            mi = -1
            for j in range(l):
                m2 = mag2(accs[j])
                ami = (m2>=np.quantile(m2,self.quantile))
                amv = accs[j][ami]
                if (v:=np.sum(mag2(ami))) > m:
                    mb = j
                    m = v
                    mi = ami
                    mv = amv
            totals[mb][mi] += mv
            accs[mb][mi] = 0
            b = self.bases[mb].inverse(self.bases[mb].unflatten(accs[mb]))
            for i in range(l):
                if i != mb:
                    accs[i] = self.bases[i].flatten(self.bases[i](b))
        return ([self.bases[i].unflatten(totals[i]) for i in range(l)],b)
    
    def nur(self,a):
        l = self.bl
        t,b = a
        for i in range(len(self.bases)):
            b += self.bases[i].inverse(t[i])
        return b
    def decoder(self):
        return bufpk(self.bl,self.nur)
    def __call__(self,v):
        l = self.bl
        hl = l//2
        self.buf[self.buf_i+hl]=v
        self.buf_i += 1
        if self.buf_i >= hl:
            self.buf_i = 0
            self.wbuf[:] = self.buf[:]
            self.wbuf *= self.window
            self.buf[:hl] = self.buf[hl:]
            self.buf[hl:] = 0
            return self.run(self.wbuf)
    def gen(self,sig):
        for v in sig:
            r = self(v)
            if r is not None:
                yield r
    def rgen(self,vsig):
        dc = self.decoder()
        for v in vsig:
            r = dc(v)
            yield from r
    





















#beat detection by low pass of power
class beat_detector:
    def __init__(self,filt):
        self.f = filt
        self.prev = 0
        self.pasc = False
    def __call__(self,v):
        v = self.f((v*v.conjugate()).real)
        r = None
        if v <= self.prev and self.pasc:
            r = self.prev
        self.pasc = v>self.prev
        self.prev = v
        return r





            
