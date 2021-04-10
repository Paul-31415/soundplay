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
        for v in flat(self.gen(g)):
            yield v
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
        for v in self.w.gen(g):
            yield v
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
        for v in self.windower.gen(g):
            yield v
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
    adj = np.diff(p1)
    inds = np.ones(len(b),dtype=int)
    inds[0] = 0
    np.add.at(inds,p1[:-1]+adj//2,np.diff(p2)-adj)
    np.add.at(d,np.cumsum(inds,out=inds)%len(d),b)
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
def rm__map_sorted(p,pt,a,at):
    sind = np.argsort(a[p[1:-2].astype(int)])+1 #a[p][sind] is sorted
    sindt = np.argsort(at[pt[1:-2].astype(int)])+1 #at[pt][sindt] is sorted
    res = p.copy()
    if len(sindt) < len(sind):
        res[sind] = pt[np.resize(sindt,len(sind))]
    else:
        res[sind] = pt[sindt[:len(sind)]]
    return res


def mt_pq(to,q1=.98,q2=.98,alpha=1,s1=1<<12,s2=1<<12,st=1<<12,rounding_mode=round_aa,q_mode=quantile_lookups):
    return window_resampler(mt_pq_(to,q1,q2,alpha,s1,s2,st,rounding_mode,q_mode),s1,s2)
class mt_pq_:
    def __init__(self,to,q1=1,q2=1,alpha=1,s1=1<<12,s2=1<<12,st=1<<12,rounding_mode=round_aa,q_mode=quantile_lookups):
        self.to = window_func(lambda x:x,st).gen(to)
        self.s1 = s1
        self.s2 = s2
        self.st = st
        self.q1 = q1
        self.q2 = q2
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
        qpt = self.quant(at,pt,self.q2)
        pt = np.concatenate((np.array([0]),qpt,(np.array([self.st-1]))))*((l-1)/(self.st-1))
        qp = self.quant(a,p,self.q1)
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
            for r in flat(v):
                yield r
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
        for v in g:
            yield v
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
            for i in r:
                yield i
        


class split_noise:
    def __init__(self,l=1<<12,t=1.75,sds = 4,nds = 16,margin=0):
        self.bl = l
        self.buf = np.zeros(l,dtype=complex)
        self.wbuf = np.zeros(l,dtype=complex)
        self.buf_i = 0
        self.window = 1-.5*np.cos(np.arange(l)*(2*np.pi/l))
        self.spargs = {"prominence":t,"width":(0,32),"rel_height":.9}
        self.smooth = sds*fourier_freq_vals(l)/l
        self.smooth *= self.smooth
        self.smooth = np.exp(-self.smooth)
        self.nsmooth = nds*fourier_freq_vals(l)/l
        self.nsmooth *= self.nsmooth
        self.nsmooth = np.exp(-self.nsmooth)
        self.margin = margin
    def run(self,b):
        l = self.bl
        ft = sp.fft.fft(b)
        mft = mag2(ft)
        mft = sp.fft.ifft((mfft:=sp.fft.fft(mft))*self.smooth).real
        mfnf = sp.fft.ifft(mfft*self.nsmooth).real
        mft /= mfnf+(mfnf==0)
        pks,pps = sp.signal.find_peaks(mft,**self.spargs)
        lb = np.clip(np.round(pps['left_ips']-self.margin).astype(int),0,l-1)
        rb = np.clip(np.round(pps['right_ips']+self.margin).astype(int),0,l-1)
        pk = np.zeros(l,dtype=int)
        np.add.at(pk,lb,1)
        np.add.at(pk,rb,-1)
        pk = np.cumsum(pk)>0
        sig = ft*pk
        noi = ft-sig
        return sig,noi
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
            for i in r:
                yield i
        
