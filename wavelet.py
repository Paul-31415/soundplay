


def convolve(a1,a2):
    res = [0 for i in range(len(a1)+len(a2)-1)]
    for i in range(len(a1)):
        for j in range(len(a2)):
            res[i+j] += a1[i]*a2[j]
    return res
    

def dwt(arr,kern=[-1,1],base=2,end=1):
    amount = len(arr)-len(kern)+1
    while (amount >= end):
        for i in range(int(amount)):
            t = 0
            for j in range(len(kern)):
                t += kern[j]*arr[i+j]
            arr[i] = t
        amount /= base
    return arr

def iwt(arr,kern=[-1,1],base=2,end=1):
    amount = len(arr)-len(kern)+1
    while (amount >= end):
        amount /= base
    amount *= base
    while amount <= len(arr)-len(kern)+1:
        for ii in range(int(amount)):
            i = int(amount)-ii-1
            t = 0
            for j in range(len(kern)-1):
                t += arr[i+j+1]*kern[j+1]
            #t+oarr[i]*kern[0] = arr[i]
            #oarr[i] = (arr[i]-t)/kern[0]
            arr[i] = (arr[i]-t)/kern[0]
        amount *= base
    return arr

def hdwt(arr,end=1):
    amount = (len(arr)>>1)
    while (amount >= end):
        other = [arr[i*2]+arr[i*2+1] for i in range(amount)]+[arr[i*2]-arr[i*2+1] for i in range(amount)]
        arr = other+arr[amount*2:]
        amount >>= 1
    return arr
def hdwt_inverse(arr,end=1):
    amount = (len(arr)>>1)
    o = 0
    while (amount>>o) >= end:
        o += 1
    while o > 0:
        o -= 1
        a = amount>>o
        other = arr[:a*2]
        re = []
        for i in range(a):
            re += [(other[i]+other[a+i])/2,(other[i]-other[a+i])/2]
        arr = re+arr[a*2:]
    return arr

        
def hdwt_sep(arr,end=1):
    amount = (len(arr)>>1)
    res = []
    i = 0
    other = [arr[i*2]+arr[i*2+1] for i in range(amount)]+[arr[i*2]-arr[i*2+1] for i in range(amount)]
    resta = amount
    while (amount >= end):
        arr = other
        i += 1
        res += [(i,arr[amount:])]
        resta = amount
        amount >>= 1
        other = [arr[i*2]+arr[i*2+1] for i in range(amount)]+[arr[i*2]-arr[i*2+1] for i in range(amount)]
    res += [(i+1,arr[:resta])]
    return res



class ArrQueue:
    def __init__(self):
        self.arr = []
        self.i = 0
    def __len__(self):
        return len(self.arr)-self.i
    def push(self,v):
        self.arr += [v]
    def deque(self):
        v = self.arr[self.i]
        self.i += 1
        if (len(self.arr) > 256):
            if (self.i*2 > len(self.arr)):
                self.arr = self.arr[self.i:]
                self.i = 0
        return v
    def has(self):
        return len(self.arr)>self.i
class RingQueue:
    def __init__(self,size=2,fill=0,initialElements=0):
        self.arr = [fill for i in range(size)]
        self.i = initialElements
        self.j = 0
    def __len__(self):
        return (self.i-self.j)%len(self.arr)
    def push(self,v):
        if self.i == ((self.j-1)%len(self.arr)):
            #double arr
            if self.j:
                self.j += len(self.arr)
            self.arr *= 2
        self.arr[self.i] = v
        self.i = (self.i+1)%len(self.arr)
    def deque(self):
        v = self.arr[self.j]
        self.j = (self.j+1)%len(self.arr)
        return v
    def rpush(self,v):
        if self.i == ((self.j-1)%len(self.arr)):
            #double arr
            if self.j:
                self.j += len(self.arr)
            self.arr *= 2
        self.j = (self.j+1)%len(self.arr)
        self.arr[self.j] = v
    def rdeque(self):
        self.i = (self.i-1)%len(self.arr)
        v = self.arr[self.i]
        return v
    def has(self):
        return self.i!=self.j
    def __getitem__(self,i):
        #push pushes to the end (high side)
        #so 0 is about to be dequed
        l = len(self)
        i %= l
        return self.arr[(self.j+i)%len(self.arr)]
    def __setitem__(self,i,v):
        #push pushes to the end (high side)
        #so 0 is about to be dequed
        l = len(self)
        i %= l
        self.arr[(self.j+i)%len(self.arr)] = v
    def items(self):
        if self.i >= self.j:
            return self.arr[self.j:self.i]
        else:
            return self.arr[self.j:]+self.arr[:self.i]
    def __repr__(self):
        return "RingQueue("+repr(self.items())+")"
class LLQueue:
    def __init__(self):
        self.head = None
        self.tail = None
        self.length = 0
    def __len__(self):
        return self.length
    def push(self,v):
        self.length += 1
        if self.head == None:
            self.head = [v,None]
            self.tail = self.head
        else:
            r = [v,None]
            self.tail[1] = r
            self.tail = r
    def deque(self):
        self.length -= 1
        v = self.head[0]
        self.head = self.head[1]
        return v
    def has(self):
        return self.head != None



def hdwt_gen(g,maxt=10):
    buf = [None for i in range(maxt)]
    for v in g:
        for i in range(len(buf)):
            if buf[i] == None:
                buf[i] = v
                break
            d = buf[i]-v
            v += buf[i]
            buf[i] = None
            yield (i,d)
        else:
            yield (maxt,v)

            
def hdwt_gen_inv(g,maxt=10,q = RingQueue):
    streams = [q() for i in range(maxt+1)]
    buf = [None for i in range(maxt)]
    v = None
    d = 0
    def get(i):
        if streams[i].has():
            return streams[i].deque()
        t = next(g)
        while t[0] != i:
            streams[t[0]].push(t[1])
            t = next(g)
        return t[1]
    while 1:
        try:
            for i in range(len(buf)):
                if buf[i] != None:
                    v = buf[i]
                    buf[i] = None
                    break
            else:
                i = maxt
                v = get(i)
            for j in range(i-1,-1,-1):
                d = get(j)
                #v + d = 2buf,  v-d = 2v
                buf[j] = (v-d)/2
                v -= buf[j]
            yield v
        except StopIteration:
            return

#haar wavelets and sinc wavelets (shannon) are fourier duals
# so does that mean that ifft(haar(fft(dat))) = shannon(dat)?
#(no, unfortunately)







import numpy as np
def fft_to_wavelets(f,layers=10):
    if layers <= 0 or len(f) < 4:
        return [(layers,np.fft.ifft(f))]
    l = len(f)
    hi = f[l//4:l*3//4]
    lo = f[:l//4]+f[l*3//4:]
    #want ifft([0]*l//4+hi+[0]*l//4)[::2]
    #freqs            -1  -.75 -.5  -.25  0   .25   .5  .75   1
    #                 |    |    |    |    |    |    |    |    |
    #                       123456789           123456789
    # to:             -2  -1.5 -1   -.5   0    .5   1   1.5   2
    #                 |    |    |    |    |    |    |    |    |
    #                 56789 123456789 123456789 123456789 12345
    
    return fft_to_wavelets(lo,layers-1)+[(layers,np.fft.ifft([0]*(l//4)+hi+[0]*(l//4)).tolist()[::2])]

        
def haarFilt(g,fl=lambda x:x,m1=5,m2=5):
    return w.hdwt_gen_inv(fl(w.hdwt_gen(g,m1)),m2)

def haarFiltLm(g,fl=lambda x:x,m1=5,m2=5):
    return w.hdwt_gen_inv((fl(i) for i in w.hdwt_gen(g,m1)),m2)


from signals import f_splitter,sinc,blackman_window,upsampler,ringBuffer

class sincWavelet:
    def __init__(self,sampleLength,lowPassPeriod,prec=3,precInWaves=0,shift=0,kf = sinc,kw = blackman_window):
        self.sargs = [sampleLength,lowPassPeriod,prec,precInWaves,shift,kf,kw]
    def makeFilter(self):
        return f_splitter(*self.sargs)
    def makeUnFilter(self):
        class merger:
            def __init__(self,f1,f2):
                self.f1 = f1
                self.q = RingQueue()
                self.f2 = f2
                self.whoseq = False
            def send(self,v):
                if v[0]:
                    for r in self.f1.send(v[1]):
                        if self.q.has() and not self.whoseq:
                            yield r+self.q.deque()
                        else:
                            self.whoseq = True
                            self.q.push(r)
                else:
                    for r in self.f2.send(v[1]):
                        if self.q.has() and self.whoseq:
                            yield r+self.q.deque()
                        else:
                            self.whoseq = False
                            self.q.push(r)
        assert self.sargs[1]>self.sargs[0]
        return merger(upsampler(self.sargs[1],self.sargs[0],0,*self.sargs[2:]),
                      upsampler(self.sargs[1],self.sargs[1]-self.sargs[0],self.sargs[0]/(self.sargs[1]-self.sargs[0]),*self.sargs[2:]))
            
        
    
                            

class wavelet_transformer:
    def __init__(self,layers=120,wavelets=[sincWavelet(16,17)],wli=0):
        self.layer = layers
        self.flt = wavelets[wli].makeFilter()
        self.cont = None
        if layers > 1:
            self.cont = wavelet_transformer(layers-1,wavelets,(wli+1)%len(wavelets))
    def send(self,inp):
        for v in self.flt.send(inp):
            if v[0]:
                if self.cont == None:
                    yield (self.layer-1,inp)
                else:
                    for i in self.cont.send(v[1]):
                        yield i
            else:
                yield (self.layer,v[1])

def wavelet_gen(g,layers=120,wavelet=sincWavelet(16,17)):
    tf = wavelet_transformer(layers,[wavelet])
    for v in g:
        for r in tf.send(v):
            yield r

class wavelet_untransformer:
    def __init__(self,layers=120,wavelets=[sincWavelet(16,17)],wli = 0):
        self.layer = layers
        self.flt = wavelets[wli].makeUnFilter()
        self.cont = None
        if layers > 1:
            self.cont = wavelet_untransformer(layers-1,wavelets,(wli+1)%len(wavelets))
    def send(self,inp):
        
        if inp[0] == self.layer:
            for v in self.flt.send((False,inp[1])):
                yield v
        elif self.cont == None:
            for r in self.flt.send((True,inp[1])):
                yield r
        else:
            for v in self.cont.send(inp):
                for r in self.flt.send((True,v)):
                    yield r
def wavelet_gen_inv(g,layers=120,wavelet=sincWavelet(16,17)):
    tf = wavelet_untransformer(layers,[wavelet])
    for v in g:
        for r in tf.send(v):
            yield r
#idea for pyboard:
#dacs through op-amps
# gain controlled through pwm pins that are low-passed (slow dac)

                
            




class unzipper:
    def __init__(self,n,d,i=0,interleaved=True):
        self.i = i
        self.n = n
        self.d = d
        assert n <= d
        self.interleaved = interleaved
    def send(self,v):
        if self.interleaved:
            yield (self.i >= self.d,v)
            self.i = (self.i%self.d)+self.n
        else:
            yield (self.i <= self.d,v)
            self.i = (self.i+1)%self.d
    def gen(self,g):
        i = self.i
        if self.interleaved:
            for v in g:
                yield (i >= self.d,v)
                i = (i%self.d)+self.n
        else:
            for v in g:
                yield (i <= self.d,v)
                i = (i+1)%self.d
class zipper:
    def __init__(self,n,d,i=0,interleaved=True):
        self.i = i 
        self.n = n
        self.d = d
        assert n <= d
        self.interleaved = interleaved
        self.bufs = (RingQueue(),RingQueue())
    def send(self,v):
        self.bufs[v[0]].push(v[1])
        if self.interleaved:
            while self.bufs[self.i >= self.d].has():
                yield self.bufs[self.i >= self.d].deque()
                self.i = (self.i%self.d)+self.n
        else:
            while self.bufs[self.i <= self.d].has():
                yield self.bufs[self.i<=self.d].deque()
                self.i = (self.i+1)%self.d
    def gen(self,g):
        i = self.i
        bufs = (RingQueue(),RingQueue())
        if self.interleaved:
            for v in g:
                bufs[v[0]].push(v[1])
                while bufs[i >= self.d].has():
                    yield bufs[i >= self.d].deque()
                    i = (i%self.d)+self.n
        else:
            for v in g:
                bufs[v[0]].push(v[1])
                while bufs[i <= self.d].has():
                    yield bufs[i<=self.d].deque()
                    i = (i+1)%self.d



class streaming_adder:
    def __init__(self):
        self.q = RingQueue()
        self.w = False
    def send(self,inp):
        if inp[0] != self.w:
            if self.q.has():
                yield self.q.deque()+inp[1]
            else:
                self.w = not self.w
                self.q.push(inp[1])
        else:
            self.q.push(inp[1])

#lifting scheme

class lifting_schemer:
    def __init__(self,z,p,u):
        self.z = z
        self.p = p
        self.u = u
        self.ta = streaming_adder()
        self.fa = streaming_adder()
    #   f•->(+)--•->
    #    |   ^   |
    #->--z   p   u
    #    |   |   v
    #   t•---•->(+)->
    def send(self,inp):
        for t in self.z.send(inp):
            if t[0]:
                for v in self.ta.send(t):
                    yield (True,v)
                for v in self.p.send(t[1]):
                    for r in self.fa.send((True,v)):
                        yield (False,r)
                        for u in self.u.send(r):
                            for o in self.ta.send((False,u)):
                                yield (True,o)
            else:
                for r in self.fa.send(t):
                    yield (False,r)
                    for u in self.u.send(r):
                        for o in self.ta.send((False,u)):
                            yield (True,o)

class general_lifting_schemer:
    def __init__(self,z,p,u):
        self.z = z
        self.p = p
        self.u = u
        self.ta = streaming_adder()
        self.fa = streaming_adder()
    #   f•-->p---•->
    #    |   ^   |
    #->--z   |   |
    #    |   |   v
    #   t•---•-->u->


def buffFiller():
    bf = RingQueue()
    def g(b):
        while 1:
            x = yield
            b.push(x)
    r = g(bf)
    next(r)
    return (bf,r)

def sls(g,*args):
    e = buffFiller()
    o = buffFiller()
    l = sinc_lifting_scheme(e[1],o[1],*args)
    next(e[1])
    next(o[1])
    next(l)
    for v in g:
        l.send(v)
    return (e[0],o[0],l)
def ssg(g,*args):
    e = buffFiller()
    o = buffFiller()
    l = sinc_splitter(e[1],o[1],*args)
    next(e[1])
    next(o[1])
    next(l)
    for v in g:
        l.send(v)
    for v in l:
        pass
    return (e[0],o[0])

def eater():
    while 1:
        yield

class sinc_wavelet_tree:
    def __init__(self,hf,lf,hif,lof,*args):
        self.args = args
        self.hf = hf
        self.lf = lf
        self.fs = (lof,hif)
    def splitter(self):
        hf = self.hf
        lf = self.lf
        if type(hf) == type(self):
            hf = hf.splitter()
        if type(lf) == type(self):
            lf = lf.splitter()
        stage = sinc_splitter(lf,hf,*self.args)
        next(stage)
        return stage
    def merger(self,streams,i=0):
        hf = self.hf
        lf = self.lf
        if type(hf) == type(self):
            hf,i = hf.merger(streams,i)
        else:
            hf = streams[i]
            i += 1
        if type(lf) == type(self):
            lf,i = lf.merger(streams,i)
        else:
            lf = streams[i]
            i += 1
        stage = sinc_merger(lf,hf,*self.args)
        return (stage,i)
            
    def __repr__(self):
        return "swt(flo:"+repr(self.fs[0])+", fhi:"+repr(self.fs[1])+")"

def sinc_tree(maxRatio=16/17,minF=20,hiF=48000,lowF=0,*args):
    if hiF <= minF or lowF/hiF>maxRatio:
        buf, gen = buffFiller()
        return ([buf],gen)
    else:
        lb,lo = sinc_tree(maxRatio,minF,(lowF+hiF)/2,lowF)
        hb,hi = sinc_tree(maxRatio,minF,hiF,(lowF+hiF)/2)
        return (hb+lb,sinc_wavelet_tree(hi,lo,hiF,lowF,*args))
        
    
    

def sinc_splitter(even,odd,prec=3,f=sinc,w=blackman_window):
    buffs = (ringBuffer(prec*2),ringBuffer(prec*2))
    outs = (even,odd)
    phase = False
    kern = [f(i+.5)*w(.5+i/prec) for i in range(prec-1,-1,-1)]
    #     sig: 000000||||||||||
    #   evens: 0 0 0 | | | | |
    #    odds:  0 0 0 | | | | |
    #    kern:  - _ ¯1¯ _ -
    #need prec odds to start
    #2*prec samples
    post = 0
    for i in range(prec*2):
        v = yield
        if (v == None):
            v = 0
        else:
            post += 1
        buffs[phase].additem(v)
        phase = not phase
    v = yield
    while v != None:
        buffs[phase].additem(v)
        phase = not phase
        if outs[not phase]!= None:
            v = buffs[not phase][prec-1]
            for i in range(prec): #need high pass when odd (phase = true)
                v += (buffs[phase][i]+buffs[phase][-i-1])*kern[i]*(phase*2-1)
            outs[not phase].send(v)
        v = yield
    for i in range(post):
        buffs[phase].additem(0)
        phase = not phase
        if outs[not phase]!= None:
            v = buffs[not phase][prec-1]
            for i in range(prec):
                v += (buffs[phase][i]+buffs[phase][-i-1])*kern[i]*(phase*2-1)
            outs[not phase].send(v)

def sinc_merger(even,odd,prec=3,f=sinc,w=blackman_window):
    buffs = (ringBuffer(prec*2),ringBuffer(prec*2))
    ins = (even,odd)
    phase = False
    kern = [f(i+.5)*w(.5+i/prec) for i in range(prec-1,-1,-1)]
    post = 0
    pre = 0
    try:
        for i in range(prec*2):
            v = next(ins[phase])
            post += 1
            pre += 1
            buffs[phase].additem(v)
            phase = not phase
        while 1:
            v = next(ins[phase])
            buffs[phase].additem(v)
            v = buffs[phase][prec-1]
            phase = not phase
            for i in range(prec): 
                v += (buffs[phase][i]+buffs[phase][-i-1])*kern[i]*(1-phase*2)
            yield v
    except StopIteration:
        for i in range(prec*2-pre):
            buffs[phase].additem(0)
            phase = not phase
        for i in range(post):
            buffs[phase].additem(0)
            v = buffs[phase][prec-1]
            phase = not phase
            for i in range(prec): 
                v += (buffs[phase][i]+buffs[phase][-i-1])*kern[i]*(1-phase*2)
            yield v


    
def sinc_splitter_bad(even,odd,prec=3,f=sinc,w=blackman_window):
    evenBuff = ringBuffer(prec*2)
    oddBuff = ringBuffer(prec*2)
    pkern = [f(i+.5)*w(.5+i/prec) for i in range(prec-1,-1,-1)]
    missede = 0
    missedo = 0
    for i in range(prec-1):
        evenBuff.additem(0)
        oddBuff.additem(0)
    for i in range(prec-1):
        ev = yield
        if ev == None:
            if missede < prec-1:
                missede += 1
                ev = 0
            else:
                return
        evenBuff.additem(ev)
        od = yield
        if od == None:
            if missedo < prec:
                missedo += 1
                od = 0
            else:
                return
        oddBuff.additem(od)
    ev = yield
    if ev == None:
        if missede < prec-1:
            missede += 1
            ev = 0
        else:
            return
    evenBuff.additem(ev)
    ev = evenBuff[prec]
    for i in range(prec):
        ev += (oddBuff[i]+oddBuff[-i-1])*pkern[i]
    even.send(ev)
    od = yield
    if od == None:
        if missedo < prec:
            missedo += 1
            od = 0
        else:
            return
    oddBuff.additem(od)
    while 1:
        ev = yield
        if ev == None:
            if missede < prec-1:
                missede += 1
                ev = 0
            else:
                od = yield
                if od == None:
                    if missedo < prec:
                        missedo += 1
                        od = 0
                else:
                    return
                oddBuff.additem(od)
                od = oddBuff[prec]
                for i in range(prec):
                    od -= (evenBuff[i]+evenBuff[-i-1])*pkern[i]
                odd.send(od)
                return
        evenBuff.additem(ev)
        ev = evenBuff[prec]
        for i in range(prec):
            ev += (oddBuff[i]+oddBuff[-i-1])*pkern[i]
        even.send(ev)
        
        od = yield
        if od == None:
            if missedo < prec:
                missedo += 1
                od = 0
            else:
                return
        oddBuff.additem(od)
        od = oddBuff[prec]
        for i in range(prec):
            od -= (evenBuff[i]+evenBuff[-i-1])*pkern[i]
        odd.send(od)




def sinc_merger_bad(even,odd,prec=3,f=sinc,w=blackman_window):
    evenBuff = ringBuffer(prec*2)
    oddBuff = ringBuffer(prec*2)
    pkern = [f(i+.5)*w(.5+i/prec) for i in range(prec-1,-1,-1)]
    for i in range(prec):
        evenBuff.additem(0)
        oddBuff.additem(0)
    try:
        for i in range(prec):
            evenBuff.additem(next(even))
            oddBuff.additem(next(odd))
        while 1:
            #even is lf, odd is hf
            #len(odd) = len(even) or len(even)-1
            evenBuff.additem(next(even))
            ev = evenBuff[prec]
            for i in range(prec):
                ev -= (oddBuff[i]+oddBuff[-i-1])*pkern[i]
            yield ev
            oddBuff.additem(next(odd))
            od = oddBuff[prec]
            for i in range(prec):
                od += (evenBuff[i]+evenBuff[-i-1])*pkern[i]
            yield od
    except StopIteration:
        for i in range(prec//2):
            evenBuff.additem(0)
            ev = evenBuff[prec]
            for i in range(prec):
                ev -= (oddBuff[i]+oddBuff[-i-1])*pkern[i]
            yield ev
            oddBuff.additem(0)
            od = oddBuff[prec]
            for i in range(prec):
                od += (evenBuff[i]+evenBuff[-i-1])*pkern[i]
            yield od
        if prec&1:
            evenBuff.additem(0)
            ev = evenBuff[prec]
            for i in range(prec):
                ev -= (oddBuff[i]+oddBuff[-i-1])*pkern[i]
            yield ev
        
        
def sinc_lifting_scheme(even,odd,prec=3,f=sinc,w=blackman_window):
    #   f•->(-)--•--> odd  (hf)
    #    |   ^a  |b
    #->--z   p   u
    #    |   |   v c
    #   t•---•->(+)-> even (lf)

    #if p is sinc
    #  0_0¯1¯0_0
    #then odd is high pass
    #and for p to be low pass,
    # u is identity function?
    #kern looks like
    #at a : 0_0¯0¯0_0
    #at b : 0¯0_1_0¯0 aka unfilterd minus low : high
    #at c : want low = unfiltered - high
    #so u is negate?
    #     f:    1
    #     t:   1
    #     a: _ ¯ ¯ _
    #     b: ¯ _1_ ¯
    #     c: ¯ _1_ ¯->u (u is hf interpolate)
    #      +   1
    #want:  _ ¯1¯ _
    #u is hf interp is 2sinc(2x)-sinc(x)
    #which  is just -sinc(x) at x = .5k
    evenBuff = ringBuffer(prec*2)
    oddBuff = ringBuffer(prec*2)
    pkern = [f(i+.5)*w(.5+i/prec) for i in range(prec-1,-1,-1)]
    evenBuff.addgen((0 for i in pkern))
    oddBuff.addgen((0 for i in pkern))
    oddQueue = RingQueue(prec+1)
    #evenQueue = RingQueue(prec+ use evenbuff[-1]
    for i in range(prec):
        ev = yield
        evenBuff.additem(ev)
        od = yield
        oddQueue.push(od)
    for i in range(prec):
        ev = yield
        evenBuff.additem(ev)
        prediction = -oddQueue.deque()
        for i in range(prec):
            prediction += (evenBuff[i]+evenBuff[-i-1])*pkern[i]
        odd.send(-prediction)
        oddBuff.additem(-prediction)
        #run through with prec = 1 for finding off by 1 errors
        #   f•->(-)--•--> odd
        #    |   ^a  |b
        #->--z   p   u
        #    |   |   v c
        #   t•---•->(+)-> even
        #
        # in: 9876543210->
        #
        # pkern 2 vals wide, so p produces 
        # t sees 0       2       4       6       8
        # f sees     1       3       5       7       9
        # a sees          p02     p24     p46     p68
        # b sees          1-p02   3-p24   5-p46   7-p68
        # c sees                   0+u1    2+u3    4+u5    6+u7
        od = yield
        oddQueue.push(od)
    while 1:
        ev = yield
        ev_out = evenBuff.additem(ev)
        prediction = -oddQueue.deque()
        for i in range(prec):
            prediction += (evenBuff[i]+evenBuff[-i-1])*pkern[i]
        odd.send(-prediction)
        oddBuff.additem(-prediction)
        for i in range(prec):
            ev_out -= (oddBuff[i]+oddBuff[-i-1])*pkern[i]
        even.send(ev_out)
        od = yield
        oddQueue.push(od)
        
        
def sinc_lifting_scheme_inverse(even,odd,prec=3,f=sinc,w=blackman_window):
    #->-----•--(+)--• odd (high freq)
    #       |a  ^b  |
    #       u   p   m-->
    #       v   |  c|
    #->--->(-)--•---• even (low freq)
    evenBuff = ringBuffer(prec*2)
    oddBuff = ringBuffer(prec*2)
    pkern = [f(i+.5)*w(.5+i/prec) for i in range(prec-1,-1,-1)]
    evenBuff.addgen((0 for i in pkern))
    oddBuff.addgen((0 for i in pkern))
    evenQueue = RingQueue(prec+1)
    for i in range(prec):
        evenQueue.push(next(even))
        oddBuff.additem(next(odd))
    for i in range(prec):
        ev = evenQueue.deque()
        for i in range(prec):
            ev += (oddBuff[i]+oddBuff[-i-1])*pkern[i]
        evenBuff.additem(ev)
        evenQueue.push(next(even))
        oddBuff.additem(next(odd))
    while 1:
        ev = evenQueue.deque()
        for i in range(prec):
            ev += (oddBuff[i]+oddBuff[-i-1])*pkern[i]
        yield evenBuff.additem(ev)
        evenQueue.push(next(even))
        od = oddBuff.additem(next(odd))
        for i in range(prec):
            od += (evenBuff[i]+evenBuff[-i-1])*pkern[i]
        yield od





#def fftWavelets(f,fmin=20,sr=48000,step=2**(1/12)):
#    if sr < fmin:
#        return [np.fft.ifft(f)]
#    l = len(f)
#    ll = int(l/step/2)
#    lo = f[:ll]+f[l-ll:]
#    hi = f[ll:l-ll]
#    return

def padAndFFT(dat):
    l = len(dat)-1
    for i in range(l.bit_length().bit_length()):
        l |= l >> (1<<i)
    l += 1
    return np.fft.fft(dat + [0]*(l-len(dat)))



def fftOctaveWavelets(f,layers=10,transform = True):
    if layers <= 0:
        return [np.fft.ifft(f).tolist() if transform else f]
    if len(f) < 4:
        fftOctaveWavelets(f,layers-1,transform)+[[]]
    l = len(f)
    lo = f[:l//4]+f[3*l//4:]
    hi = f[l//2:3*l//4]+f[l//4:l//2]
    hiSig = np.fft.ifft(hi).tolist() if transform else hi
    return fftOctaveWavelets(lo,layers-1,transform)+[hiSig]

def bintreeOctPartition(ratio=16/17,r_is_min=1,lo=1,hi=2):
    if lo/hi>=ratio and r_is_min:
        return str(lo)+"/"+str(hi)
    mid = hi+lo
    if mid/(hi*2)>=ratio and not r_is_min:
        return str(lo)+"/"+str(hi)
    h = bintreeOctPartition(ratio,r_is_min,mid,hi*2)
    l = bintreeOctPartition(ratio,r_is_min,lo*2,mid)
    if type(l) == tuple:
        return l+(h,)
    return (l,h)

def fftWavelets(f,octaves=10,otree=bintreeOctPartition(.96,0),transform=True):
    if octaves == None:
        if type(otree) != tuple:
            return np.fft.ifft(f).tolist() if transform and len(f) else f
        os = fftOctaveWavelets(f,len(otree)-1,False)
        for i in range(len(otree)):
            os[i] = fftWavelets(os[i],None,otree[i],transform)
        return os
    else:
        return fftWavelets(f,None,('dc',)+(otree,)*octaves,transform)

def wavelet_treeverse(wl,f):
    if len(wl) == 0:
        return wl
    else:
        try:
            len(wl[0])
        except:
            return f(wl)
        return [wavelet_treeverse(w,f) for w in wl]


    
def fftWavelets_inverse(wl,transform = True):
    if len(wl) == 0:
        return wl
    else:
        try:
            len(wl[0])
        except:
            return np.fft.fft(wl).tolist() if transform else wl
    lo = fftWavelets_inverse(wl[0],transform)
    for i in range(1,len(wl)):
        hi = fftWavelets_inverse(wl[i],transform)
        l = len(lo)
        h = len(hi)#in usual cases l == h
        lo = lo[:l//2]+hi[h//2:]+hi[:h//2]+lo[l//2:]
    return lo














