#               

#abstractions for handling signals and sample rate conversion
import math
import itertools

def fracApprox(r,dmax=20):
    lb = (int(r),1)
    ub = (lb[0]+1,1)
    nd = lb[1]+ub[1]
    while nd <= dmax:
        nb = (lb[0]+ub[0],nd)
        v = nb[0]/nb[1]
        if r < v:
            ub = nb
        elif r == v:
            return nb
        else:
            lb = nb
        nd = lb[1]+ub[1]
    return [lb,ub][abs(ub[0]/ub[1]-r)<abs(lb[0]/lb[1]-r)]


def sinc(x,wf=0):
    if wf != 0:
        return (wf+.5)*sinc((wf+.5)*x)-(wf-.5)*sinc((wf-.5)*x)
    if x == 0:
        return 1
    if x%2>1:
        return -math.sin(math.pi*(x%1))/(math.pi*x)
    else:
        return math.sin(math.pi*(x%1))/(math.pi*x)
    
def blackman_window(x):
    if x == .5:
        return 1
    return 0.42 - 0.5*math.cos(2*math.pi*x ) + 0.08* math.cos(4*math.pi*x )

class ringBuffer:
    def __init__(self,a=[]):
        if type(a) == int:
            self.buf = [0 for i in range(a)]
        else:
            self.buf = a
        self.offs = 0
    def expand(self,n):
        if type(n) == int:
            n = [0 for i in range(n)]
        self.buf = self.buf[:self.offs]+n+self.buf[self.offs:]
    def contract(self,n):
        o = self.offs
        l = len(self.buf)
        if n >= l:
            r = self.buf
            self.buf = []
            return r
        r = self.buf[o:o+n]+self.buf[:o+n-l]
        self.offs = max(0,o+n-l)
        self.buf = self.buf[o+n-l:o]+self.buf[o+n:]
        return r
    def __getitem__(self,i):
        return self.buf[(self.offs+i)%len(self.buf)]
    def __setitem__(self,i,v):
        self.buf[(self.offs+i)%len(self.buf)] = v
    def __len__(self):
        return len(self.buf)
    def additem(self,v):
        r = self.buf[self.offs]
        self.buf[self.offs] = v
        self.offs = (self.offs+1)%len(self.buf)
        return r
    def addgen(self,g):
        r = []
        for v in g:
            r += [self.additem(v)]
        return r

def signalFromBands(self,bn,d,bands):
    r = bands[0].deband(bn[0],bn[1],d)
    for i in range(1,len(bn)-1):
        r = r + bands[1].deband(bn[i],bn[i+1],d)
    return r

class upsampler:
    def __init__(self,inPeriod,outPeriod,nyquistRegion = 0,prec=3,precInWaves=0,shift=0,kf = sinc,kw = blackman_window,doNorm=1):
        assert outPeriod <= inPeriod
        self.outPeriod = outPeriod
        self.inPeriod = inPeriod
        if precInWaves:
            prec *= inPeriod
        l = prec*2
        self.buf = ringBuffer(l)
        for i in range(prec):
            self.buf.additem(0)
            l -= 1
        self.preload = l
        self.prec = prec

        n = inPeriod
        

        shift *= n #get shift in high-rate units

        khl = prec*n
        kl = khl*2
        wc = lambda i: (i+1)/kl
        sc = lambda i: (i+1-khl+(shift%1))/inPeriod
        #gain = outPeriod/inPeriod
        self.kern = [kf(sc(i),nyquistRegion)*kw(wc(i)) for i in range(kl)]
        if doNorm:
            self.kern = normArr(self.kern,inPeriod,nyquistRegion/inPeriod)
        self.suboffset = int(shift)
        self.khl = khl
        
    def send(self,v):
        if self.preload > 0:
             self.buf.additem(v)
             self.preload -= 1
        else:
            while self.suboffset < self.inPeriod:
                r = self.buf[0]*self.kern[self.khl-self.prec*self.inPeriod-self.suboffset]
                for i in range(1,len(self.buf)):
                    r += self.buf[i]*self.kern[(i-self.prec)*self.inPeriod+self.khl-self.suboffset]
                yield r
                self.suboffset += self.outPeriod
            #collect a new datum 
            self.suboffset -= self.inPeriod
            self.buf.additem(v)





        
class f_splitter:
    def __init__(self,sampleLength,lowPassPeriod,prec=3,precInWaves=0,shift=0,kf = sinc,kw = blackman_window,doNorm=1,hfNormFreq=.5):
        #sinc filter resampling splitting into low and high signals
        assert sampleLength <= lowPassPeriod
        #on average each input sample results in an output sample
        #so sl/lpp + sl/hpp = 1
        #so sl*hpp+lpp*sl = hpp*lpp
        #so lpp*sl = hpp*lpp-sl*hpp
        #so lpp*sl/(lpp-sl) = hpp
        highPassRate = lowPassPeriod-sampleLength # per lpp*sl
        if (highPassRate > 0):
            denom = math.gcd(highPassRate,lowPassPeriod*sampleLength)
            factor = highPassRate//denom
            lowPassPeriod *= factor
            sampleLength *= factor
            highPassPeriod = lowPassPeriod*sampleLength//highPassRate
            #           d*A/a*(d*A/a-d*B/b) * d*B/b*(d*A/a-d*B/b) / (d*A/a-d*B/b)
            #           d*A/a*(d*A/a-d*B/b) * d*B/b is integer
        else:
            highPassPeriod = float('inf')
        
        
        self.highPassPeriod = highPassPeriod
        self.lowPassPeriod = lowPassPeriod
        self.sampleLength = sampleLength
        n = sampleLength
        if precInWaves:
            prec *= lowPassPeriod

        l = prec*2
        buf = ringBuffer(l)
        for i in range(prec):
            buf.additem(0)
            l -= 1
        

        shift *= n #get shift in high-rate units
        
        #kernal length: l samples long in rate:d
        khl = prec*n
        kl = khl*2
        
        #i ranges from 0 to kl-1, we want 0 at i=-1 and 1 at i=kl-1
        wc = lambda i: (i+1)/kl
        #sinc filter
        sc = lambda i: (i+1-khl+(shift%1))/lowPassPeriod
        gain = self.sampleLength/lowPassPeriod
        self.lokern = [gain*kf(sc(i))*kw(wc(i)) for i in range(kl)]
        shc = lambda i: (i+1-khl)/n + (shift%1)/lowPassPeriod
        bkern = [kf(shc(i))*kw(wc(i)) for i in range(kl)]

        #low = signal•lokern
        #bls = signal•bkern
        #hi  = bls-low
        self.hikern = [bkern[i]-self.lokern[i] for i in range(kl)]

        if doNorm:
            self.lokern = normArr(self.lokern,sampleLength)
            self.hikern = normArr(self.hikern,sampleLength,(hfNormFreq)/sampleLength+(1-hfNormFreq)/lowPassPeriod)
        self.suboffset = int(shift)
        
        self.lf_counter = lowPassPeriod
        self.hf_counter = highPassPeriod
        self.buf = buf
        self.preload = l
        self.khl = khl
        self.prec = prec
    def send(self,v):
        if self.preload > 0:
             self.buf.additem(v)
             self.preload -= 1
        else:
            while self.suboffset < self.sampleLength:
                #output lf result sample
                if self.lf_counter <= self.hf_counter:
                    r = self.buf[0]*self.lokern[self.khl-self.prec*self.sampleLength-self.suboffset]
                    for i in range(1,len(self.buf)):
                        r += self.buf[i]*self.lokern[(i-self.prec)*self.sampleLength+self.khl-self.suboffset]
                    yield (True,r)
                    self.suboffset += self.lf_counter
                    self.hf_counter -= self.lf_counter
                    self.lf_counter = self.lowPassPeriod
                else:
                    #hf
                    r = self.buf[0]*self.hikern[self.khl-self.prec*self.sampleLength-self.suboffset]
                    for i in range(1,len(self.buf)):
                        r += self.buf[i]*self.hikern[(i-self.prec)*self.sampleLength+self.khl-self.suboffset]
                    yield (False,r)
                    self.suboffset += self.hf_counter
                    self.lf_counter -= self.hf_counter
                    self.hf_counter = self.highPassPeriod
            #collect a new datum 
            self.suboffset -= self.sampleLength
            self.buf.additem(v)

class signal:
    def __init__(self,g,r=48000,rd=1):
        self.gen = g
        self.rate_n = None
        self.rate_d = None
        self.rate(r,rd)
    def rate(self,v=None,vd=1):
        if v != None:
            d = math.gcd(v,vd)
            self.rate_n = v//d
            self.rate_d = vd//d
        return self.rate_n/self.rate_d
    def p(self,sr=48000,sd=1):
        return self.resample(sr,sd)
    def __iter__(self):
        return self.gen
    def __next__(self):
        return next(self.gen)
    def zeroPad_g(self,n=0):
        for i in range(n):
            yield 0
        for v in self.gen:
            yield v
        for i in range(n):
            yield 0
    def resample_g(self,n,d,prec=3,precInWaves=0,shift=0,kf = sinc,kw = blackman_window,doNorm=1):
        #sinc filter resampling
        lowPassPeriod = max(n,d)
        if precInWaves:
            prec *= lowPassPeriod
            
        g = self.zeroPad_g(prec)
        l = prec*2
        buf = ringBuffer(l)
        for i in range(l):
            buf.additem(next(g))

        shift *= n #get shift in high-rate units
        
        #kernal length: l samples long in rate:d
        khl = prec*n
        kl = khl*2
        
        #i ranges from 0 to kl-1, we want 0 at i=-1 and 1 at i=kl-1
        wc = lambda i: (i+1)/kl
        #sinc filter
        gain = n/lowPassPeriod
        # sc(khl-1-(shift%1)) = 0
        sc = lambda i: (i+1-khl+(shift%1))/lowPassPeriod
        kern = [gain*kf(sc(i))*kw(wc(i)) for i in range(kl)]
        if doNorm:
            kern = normArr(kern,n)
        suboffset = int(shift)
        #for n=3,d=2
        #        n:.--.
        # in gen : |  |  |  |  |  |  |
        # high   : |||||||||||||||||||
        # kern   :    __-¯¯|¯¯-__-
        # out gen: | | | | | | | | | |
        #        d:^-^
        #for n=2,d=3
        #        n:.-.
        # in gen : | | | | | | | | | |
        # high   : |||||||||||||||||||
        # kern   :     __-¯¯|¯¯-__-
        # out gen: |  |  |  |  |  |  |
        #        d:^--^
        #
        if lowPassPeriod == n and (shift%1) == 0: 
            #peephole optimization for when alligned
            for v in g:
                while suboffset < n:
                    #output result sample
                    # kern[(khl-1)%n::n] = [...0,0,0,1,0,0,0,...]
                    # kern[khl-1] = 1
                    # khl = prec*n, (khl-1)%n = n-1
                    if suboffset == n-1:
                        #buf[i] with kern[(i-prec)*n+khl-suboffset]
                        #(i-prec)*n+khl-suboffset = khl-1
                        #(i-prec)*n-suboffset = -1
                        #(i-prec)*n-n = 0
                        #i-prec = 1
                        #i = prec+1
                        yield kern[khl-1]*buf[prec+1]
                    else:
                        r = buf[0]*kern[khl-prec*n-suboffset]
                        for i in range(1,len(buf)):
                            r += buf[i]*kern[(i-prec)*n+khl-suboffset]
                        yield r
                    suboffset += d
                #collect a new datum 
                suboffset -= n
                buf.additem(v)
            
        else:
            for v in g:
                while suboffset < n:
                    #output result sample
                    r = buf[0]*kern[khl-prec*n-suboffset]
                    for i in range(1,len(buf)):
                        r += buf[i]*kern[(i-prec)*n+khl-suboffset]
                    yield r
                    suboffset += d
                #collect a new datum 
                suboffset -= n
                buf.additem(v)
    def upsample(self,*args):
        return signal(self.upsample_g(*args))
    def upsample_g(self,n,d=1,band=0,prec=3,precInWaves=0,shift=0,kf = sinc,kw = blackman_window):
        #sinc filter upsampling
        assert n>=d
        lowPassPeriod = n
        if precInWaves:
            prec *= lowPassPeriod
            
        g = self.zeroPad_g(prec)
        l = prec*2
        buf = ringBuffer(l)
        for i in range(l):
            buf.additem(next(g))

        shift *= n #get shift in high-rate units
        
        #kernal length: l samples long in rate:d
        khl = prec*n
        kl = khl*2
        
        #i ranges from 0 to kl-1, we want 0 at i=-1 and 1 at i=kl-1
        wc = lambda i: (i+1)/kl
        #sinc filter
        # sc(khl-1-(shift%1)) = 0
        sc = lambda i: (i+1-khl+(shift%1))/lowPassPeriod
        kern = [kf(sc(i),band)*kw(wc(i)) for i in range(kl)]
        suboffset = int(shift)
        if (shift%1) == 0: 
            #peephole optimization for when alligned
            for v in g:
                while suboffset < n:
                    if suboffset == n-1:
                        yield kern[khl-1]*buf[prec+1]
                    else:
                        r = buf[0]*kern[khl-prec*n-suboffset]
                        for i in range(1,len(buf)):
                            r += buf[i]*kern[(i-prec)*n+khl-suboffset]
                        yield r
                    suboffset += d
                #collect a new datum 
                suboffset -= n
                buf.additem(v)
            
        else:
            for v in g:
                while suboffset < n:
                    #output result sample
                    r = buf[0]*kern[khl-prec*n-suboffset]
                    for i in range(1,len(buf)):
                        r += buf[i]*kern[(i-prec)*n+khl-suboffset]
                    yield r
                    suboffset += d
                #collect a new datum 
                suboffset -= n
                buf.additem(v)
    def resample_g_split(self,sampleLength,lowPassPeriod,prec=3,precInWaves=0,shift=0,kf = sinc,kw = blackman_window,doNorm=1,hfNormFreq=.5):
        #sinc filter resampling splitting into low and high signals
        highPassPeriod = lowPassPeriod-sampleLength
        n = sampleLength
        if precInWaves:
            prec *= lowPassPeriod
            
        g = self.zeroPad_g(prec)
        l = prec*2
        buf = ringBuffer(l)
        for i in range(l):
            buf.additem(next(g))

        shift *= n #get shift in high-rate units
        
        #kernal length: l samples long in rate:d
        khl = prec*n
        kl = khl*2
        
        #i ranges from 0 to kl-1, we want 0 at i=-1 and 1 at i=kl-1
        wc = lambda i: (i+1)/kl
        #sinc filter
        sc = lambda i: (i+1-khl+(shift%1))/lowPassPeriod
        lokern = [kf(sc(i))*kw(wc(i)) for i in range(kl)]
        shc = lambda i: (i+1-khl)/n + (shift%1)/lowPassPeriod
        bkern = [kf(shc(i))*kw(wc(i)) for i in range(kl)]

        #low = signal•lokern
        #bls = signal•bkern
        #hi  = bls-low
        hikern = [bkern[i]-lokern[i] for i in range(kl)]


        if doNorm:
            lokern = normArr(lokern,sampleLength)
            hikern = normArr(hikern,sampleLength,(hfNormFreq)/sampleLength+(1-hfNormFreq)/lowPassPeriod)
        suboffset = int(shift)
        
        lf_counter = lowPassPeriod
        hf_counter = highPassPeriod
        
        for v in g:
            while suboffset < n:
                #output lf result sample
                if lf_counter < hf_counter:
                    r = buf[0]*lokern[khl-prec*n-suboffset]
                    for i in range(1,len(buf)):
                        r += buf[i]*lokern[(i-prec)*n+khl-suboffset]
                    yield (True,r)
                    suboffset += lf_counter
                    hf_counter -= lf_counter
                    lf_counter = lowPassPeriod
                else:
                    r = buf[0]*hikern[khl-prec*n-suboffset]
                    for i in range(1,len(buf)):
                        r += buf[i]*hikern[(i-prec)*n+khl-suboffset]
                    yield (False,r)
                    suboffset += hf_counter
                    lf_counter -= hf_counter
                    hf_counter = highPassPeriod
            #collect a new datum 
            suboffset -= n
            buf.additem(v)

    def play(self,rate=1,sr=48000,fracPrec=16,*args):
        r = self.rate_d * sr/self.rate_n/rate
        fa = fracApprox(r,fracPrec)
        #print(fa)
        if fa[0] == fa[1]:
            return signal(self.gen,self.rate_n,self.rate_d)
        return signal(self.resample_g(fa[0],fa[1],*args),self.rate_n*fa[0],self.rate_d*fa[1])
    
    def resample(self,n,d=1,p=3,piw=0,s=0,kf = sinc,kw = blackman_window):
        #r = (n/d)/self.rate
        rn = self.rate_d * n
        rd = self.rate_n * d
        cd = math.gcd(rn,rd)
        if rn == rd:
            return signal(self.gen,n,d)
        return signal(self.resample_g(rn//cd,rd//cd,p,piw,s,kf,kw),n,d)
    def resampleby(self,n,d,p=3,piw=0,s=0,kf = sinc,kw = blackman_window):
        rn = n
        rd = d
        cd = math.gcd(rn,rd)
        if rn == rd:
            signal(self.gen,self.rate_n,self.rate_d)
        return signal(self.resample_g(rn//cd,rd//cd,p,piw,s,kf,kw),n*self.rate_n,d*self.rate_d)

    def shiftf(self,freq):
        if freq == 0:
            return self
        def s(g):
            f = 1
            m = math.exp(1)**(1j*freq)
            for v in g:
                yield f*v
                f *= m
        return signal(s(self.gen),self.rate_n,self.rate_d)
    def tee(self,n):
        return tuple(signal(g,self.rate_n,self.rate_d) for g in itertools.tee(self.gen,n))
    def split(self,n):
        return self.tee(n)

        
    def band(self,ln,hn,d):
        #in nyquist units
        bw = hn-ln
        if hn+ln == 0:
            return self.resampleby(bw,d)
        return self.shiftf(-(hn+ln)/d/2).resampleby(bw,d)
    def deband(self,ln,hn,d):
        #in nyquist units
        bw = hn-ln
        if hn+ln == 0:
            return self.resampleby(d,bw)
        return self.resampleby(d,bw).shiftf((hn+ln)/d/2)

    def bands(self,bs,d):
        gs = self.tee(len(bs)-1)
        return tuple(gs[i].band(bs[i],bs[i+1],d) for i in range(len(bs)-1))
    
    def pad(self,v):
        return signal(itertools.chain(self.gen,itertools.repeat(v)),self.rate_n,self.rate_d)

    def m(self,o):
        if isinstance(o,signal):
            rn = max(o.rate_n*self.rate_d,self.rate_n*o.rate_d)
            rd = self.rate_d*o.rate_d
            return self.resample(rn,rd)
        return self
    
    def binop(self,o,op=lambda a,b:a+b):
        if isinstance(o,signal):
            if o.rate_n == self.rate_n and o.rate_d == self.rate_d:
                return signal((op(v,next(o.gen)) for v in self.gen),self.rate_n,self.rate_d)
            rn = max(o.rate_n*self.rate_d,self.rate_n*o.rate_d)
            rd = self.rate_d*o.rate_d
            return self.resample(rn,rd).binop(o.resample(rn,rd),op)
        try:
            o.__next__
        except:
            return signal((op(i,o) for i in self.gen),self.rate_n,self.rate_d)
        return signal((op(i,next(o)) for i in self.gen),self.rate_n,self.rate_d)
        
    def __add__(self,o):
        return self.binop(o,lambda a,b:a+b)
    def __radd__(self,o):
        return self.binop(o,lambda a,b:b+a)
    def __sub__(self,o):
        return self.binop(o,lambda a,b:a-b)
    def __rsub__(self,o):
        return self.binop(o,lambda a,b:b-a)
    def __mul__(self,o):
        return self.binop(o,lambda a,b:a*b)
    def __rmul__(self,o):
        return self.binop(o,lambda a,b:b*a)
    def __truediv__(self,o):
        return self.binop(o,lambda a,b:a/b)
    def __rtruediv__(self,o):
        return self.binop(o,lambda a,b:b/a)
    def __floordiv__(self,o):
        return self.binop(o,lambda a,b:a//b)
    def __rfloordiv__(self,o):
        return self.binop(o,lambda a,b:b//a)
    def __mod__(self,o):
        return self.binop(o,lambda a,b:a%b)
    def __rmod__(self,o):
        return self.binop(o,lambda a,b:b%a)
    def __pow__(self,o):
        return self.binop(o,lambda a,b:a**b)
    def __rpow__(self,o):
        return self.binop(o,lambda a,b:b**a)
    def __lshift__(self,o):
        return self.binop(o,lambda a,b:a<<b)
    def __rlshift__(self,o):
        return self.binop(o,lambda a,b:b<<a)
    def __rshift__(self,o):
        return self.binop(o,lambda a,b:a>>b)
    def __rrshift__(self,o):
        return self.binop(o,lambda a,b:b>>a)
    def __and__(self,o):
        return self.binop(o,lambda a,b:a&b)
    def __rand__(self,o):
        return self.binop(o,lambda a,b:b&a)
    def __xor__(self,o):
        return self.binop(o,lambda a,b:a^b)
    def __rxor__(self,o):
        return self.binop(o,lambda a,b:b^a)
    def __or__(self,o):
        return self.binop(o,lambda a,b:a|b)
    def __ror__(self,o):
        return self.binop(o,lambda a,b:b|a)
    def __lt__(self,o):
        return self.binop(o,lambda a,b:a<b)
    def __le__(self,o):
        return self.binop(o,lambda a,b:a<=b)
    def __eq__(self,o):
        return self.binop(o,lambda a,b:a==b)
    def __ne__(self,o):
        return self.binop(o,lambda a,b:a!=b)
    def __ge__(self,o):
        return self.binop(o,lambda a,b:a>=b)
    def __gt__(self,o):
        return self.binop(o,lambda a,b:a>b)

    def op(self,op=lambda a:a):
        return signal((op(v) for v in self.gen),self.rate_n,self.rate_d)
    def __neg__(self):
        return self.op(lambda a:-a)
    def __pos__(self):
        return self.op(lambda a:+a)
    def __abs__(self):
        return self.op(abs)
    def __invert__(self):
        return self.op(lambda a:~a)
    def __complex__(self):
        return self.op(complex)
    def __int__(self):
        return self.op(int)
    def __long__(self):
        return self.op(long)
    def __float__(self):
        return self.op(float)
    def __bool__(self):
        return self.op(bool)
    

    
def normArr(arr,v=1,f=0,w = lambda x:math.cos(x*math.pi*2)):
    tot = sum((w(f*(i-len(arr)//2))*arr[i] for i in range(len(arr))))
    factor = v/tot
    for i in range(len(arr)):
        arr[i] *= factor
    return arr
    
class resampledAccessor:
    def __init__(self,src,srcPeriod,lowPassPeriod,prec=3,precInWaves=0,shift=0,kf = sinc,kw = blackman_window,doNorm=1):
        self.src = src
        self.srcPeriod = srcPeriod
        #sinc filter resampling
        if precInWaves:
            prec *= lowPassPeriod
            
        shift *= srcPeriod #get shift in high-rate units
        
        #kernal length: l samples long in rate:d
        khl = prec*srcPeriod
        kl = khl*2
        
        #i ranges from 0 to kl-1, we want 0 at i=-1 and 1 at i=kl-1
        wc = lambda i: (i+1)/kl
        #sinc filter
        sc = lambda i: (i+1-khl+(shift%1))/lowPassPeriod
        gain = srcPeriod/lowPassPeriod
        self.kern = [gain*kf(sc(i))*kw(wc(i)) for i in range(kl)]
        if doNorm:
            self.kern = normArr(self.kern,srcPeriod)
        suboffset = int(shift)
        self.prec = prec
        self.offs = suboffset
        self.kerns = [self.kern[i::self.srcPeriod] for i in range(self.srcPeriod)]
        self.shift = shift%1
    def __getitem__(self,o):
        o += self.offs
        so = self.srcPeriod-(o%self.srcPeriod)-1
        s = (o//self.srcPeriod) - self.prec+1#start
        #return sum((self.src[s+i]*self.kerns[so][i] for i in range(self.prec*2)))
        if so == self.srcPeriod-1 and self.shift == 0:
            return self.src[o//self.srcPeriod]*self.kern[self.prec*self.srcPeriod-1]
        r = self.src[s]*self.kern[so]
        for i in range(1,self.prec*2):
            r += self.src[s+i]*self.kern[so+i*self.srcPeriod]
        return r
    def __len__(self):
        return self.srcPeriod*len(self.src)
