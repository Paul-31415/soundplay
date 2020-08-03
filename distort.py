
#nonlinear stuffs
import math
import random
def l(g,f):
    for i in g:
        yield f(i)
def o(f):
    def r(x):
        return f(x.real)+1j*f(x.imag)
    return r

def stereoDelay(g,d=0):
    if d == 0:
        for i in g:
            yield i
    else:
        b = [0 for i in range(abs(d))]
        i = 0
        if d > 0:
            for s in g:
                yield b[i]+1j*s.imag
                b[i] = s.real
                i = (i+1)%d
        else:
            for s in g:
                yield b[i]*1j+s.real
                b[i] = s.imag
                i = (i+1)%-d

def abs2(n):
    return (n*n.conjugate()).real

def waveDither(wave,signal,samples=4):
    err = 0
    for i in signal:
        goal = i+err
        bestErr = 1e300
        bestSamp = 0
        for j in range(samples):
            samp = wave(random.random())
            d = abs2(samp-goal)
            if d<bestErr:
                bestSamp = samp
                bestErr = d
        yield bestSamp
        err += i-bestSamp
def waveDither2(wave,signal,samples=4):
    err = 0
    for i in signal:
        goal = i+err
        bestErr = 1e300
        bestSamp = 0
        off = random.random()/samples
        for j in range(samples):
            samp = wave(off+j/samples)
            d = abs2(samp-goal)
            if d<bestErr:
                bestSamp = samp
                bestErr = d
        yield bestSamp
        err += i-bestSamp

def waveDither3(wave,signal,samples=4,rang=.05):
    err = 0
    t = 0
    for i in signal:
        goal = i+err
        bestErr = 1e300
        bestSampT = 0
        bestSamp = 0
        for j in range(1-samples,samples):
            testT = t+rang*j
            samp = wave(testT)
            d = abs2(samp-goal)
            if d<bestErr:
                bestSamp = samp
                bestSampT = testT
                bestErr = d
        yield bestSamp
        err += i-bestSamp
        t = bestSampT

def waveDither4Curry(wave,lutSize=64,waveStep=256):
    lut = [[0 for i in range(lutSize)] for j in range(lutSize)]
    lc = lambda v: min(lutSize-1,max(0,int((v/2+.5)*lutSize)))
    cl = lambda c: (c/lutSize-.5)*2
    #initialize lut
    pts = []
    for i in range(waveStep):
        t = i/waveStep
        p = wave(t)
        pts += [[p,t]]
    for x in range(lutSize):
        for y in range(lutSize):
            minD = 1e300
            minT = 0
            minV = 0
            for pt in pts:
                d = abs2(pt[0]-cl(x)-1j*cl(y))
                if d < minD:
                    minD = d
                    minT = pt[1]
                    minV = pt[0]
            lut[x][y] = minV
    #now do the dither
    def f(signal):
        err = 0
        for i in signal:
            r = lut[lc((i+err).real)][lc((i+err).imag)]
            yield r
            err += i-r
    return f
def waveDither4(wave,signal,lutSize=64,waveStep=256):
    for i in waveDither4Curry(wave,lutSize,waveStep)(signal):
        yield i
                

def sign(v):
    return (v>0)-(v<0)

def crinkle(v,a):
    def r(n):
        if n>v:
            return max(n-a,v)
        elif n < -v:
            return min(n+a,v)
        else:
            return n
    return r

def crinkles(v,a):
    d = max(v+a,v)
    s = max(v-a,v)
    def r(n):
        div,rem = divmod(n+v/2,d)
        if rem>v:
            return s*(div+.5)
        else:
            return s*(div-.5)+rem
    return r

def bound(l,h=None):
    if h == None:
        h = -l
    def f(x):
        return max(l,min(x,h))
    return f

def genify(v):
    val = True
    try:
        if v.__next__:
            val = False
    except:
        pass
    if val:
        while 1:
            yield v
    else:
        for i in v:
            yield i


def tanh(x):
    return (lambda a,b: (a-b)/(a+b))(math.exp(x),math.exp(-x))

        


            
def compress(g,low=0,high=1,falloff = .999,lookahead=0,heuristic = lambda x: abs(x.real)+abs(x.imag)):
    v = 0
    arr = []
    for i in range(lookahead+1):
        arr += [next(g)]
        v = v*falloff+heuristic(arr[i])*(1-falloff)
    index = 0
    for i in g:
        v = v*falloff+heuristic(i)*(1-falloff)
        d = max(low,min(high,(v+.00001)))
        yield arr[index]*d/(v+.00001)
        arr[index] = i
        index = (index+1)%len(arr)
        
        
def movingAverageCompress(g,low=1,high=1,length=2400,offset=1200,heuristic = lambda x: abs(x.real)+abs(x.imag)):
    v = 0
    arr = []
    for i in range(length):
        arr += [next(g)]
        v+= heuristic(arr[i])
    index = offset
    for i in g:
        d = max(low,min(high,v/len(arr)))
        yield arr[index]*d/(v/len(arr)+.00001)
        v -= heuristic(arr[index])
        arr[index] = i
        v += heuristic(i)
        index = (index+1)%len(arr)


def maxholdCompress(g,high=.7,low=0,falloff=1):
    m = 1
    for i in g:
        m = max(low,m*falloff,abs(i.real),abs(i.imag))
        yield high*i/m
    

def resamp0(g,r):
    t = 0
    r = genify(r)
    v = next(g)
    while 1:
        yield v
        t += next(r)
        while t > 1:
            t -= 1
            v = next(g)

"""def resamp1(g,r):
    t = 0
    r = genify(r)
    p = 0
    v = next(g)
    while 1:
        yield p*tv
        t += next(r)
        while t > 1:
            t -= 1
            v = next(g)
#"""         

            
def puncture(g,n,d):
    n = genify(n)
    ne = 0
    d = genify(d)
    de = 0
    while 1:
        ne += next(n)
        de += next(d)
        while ne>0:
            yield next(g)
            ne -= 1
        while de>0:
            next(g)
            de -= 1

def section(g,n,p):
    n = genify(n)
    ne = 0
    p = genify(p)
    pe = 0
    arr = [0]
    while 1:
        ne += next(n)
        pe += next(p)
        if ne > 0:
            arr = []
        while ne>0:
            arr += [next(g)]
            ne -= 1
        i = 0
        while pe>0:
            yield arr[i]
            i = (i + 1)%len(arr)
            pe -= 1

def noise():
    while 1:
        yield random.random()*2+random.random()*2j-2-2j
def gaussNoise(mu=0,sigma=1+1j):
    while 1:
        yield random.gauss(mu.real,sigma.real)+random.gauss(mu.imag,sigma.imag)*1j
def normNoise():
    for i in gaussNoise():
        yield i/abs(i)

def phaseNoise(rate = 1,sigma = 1):
    if rate > 1:
        sigma /= rate
        rate = 1
    t = 0
    v = 0
    p = 0
    while 1:
        t += rate
        if t > 1:
            t -= 1
            p = v
            v = random.gauss(0,sigma)
        yield math.e**(1j*(v*t+p*(1-t)))

def hfNoise(f,sigma=1):
    t = 0
    while 1:
        v = random.gauss(-t*f,sigma)
        yield v
        t += v

def resampFilt(g,v=1,a=.01,b=.0001):
    err = 0
    drive = 0
    for i in g:
        r = i-a*err-b*drive
        err += r-v
        drive += err
        yield r
        
def s(i,n,d):
    return i+((i*d)%n)

"""
def movingTaps(i,r,tr,n,f = lambda x:1-abs(x)):
    #evenly spaced(by n) taps traveling at rate tr, look at nearest 2 and interp
    #
   """ 
            


        
def split(g):
    buf = [0,[]]
    def gen(b):
        i = 0
        while 1:
            if len(b[1])+b[0] <= i:
                v = next(g)
                b[1] += [v]
                yield v
            else:
                yield b[1][i-b[0]]
                if (i-b[0])*2>len(b[1]):
                    #chop b[1] down by half
                    c = len(b[1])//2
                    b[1] = b[1][c:]
                    b[0] += c
            i += 1
    return gen(buf),gen(buf)
        

def makeiir(f,order=1):
    #f is in units of sampling frequency
    if order == 1:
        #returns an iir low pass filter for positive frequencies,
        # high pass for negative freqs
        t1 = f
        t2 = 1-f
        def r(g):
            t = 0
            for i in g:
                t *= t1
                t += i*t2
                yield t
        return r

eone = math.exp(math.pi*2)
    
def addFreq(g,f,sr=48000,filtmaker = lambda f: makeiir(f) ):
    fi = eone**(1j*f/sr)
    fter = filtmaker(f/sr)
    def c(g):
        t = 1
        for i in g:
            t *= fi
            yield i*t
    return fter(c(g))
        
        


"""def blnoise(f=24000,sr=48000):
    while 1:
   """     
        
        
"""def pitchTempo(g,p,t,s=9000):
    p = genify(p)
    t = genify(t)
    while 1:
   """






#data compression based distortion

def centerMod(n,m):
    return ((n+m//2)%m)-m//2

import numpy as np

def fourierMaxCompress_c(n=16,w=64):
    def c(g):
        while 1:
            f = np.fft.fft([next(g) for i in range(w)])
            m = f.argsort()[-n:][::-1]
            yield [(centerMod(i,w),f[i]) for i in m]
    return c
def fourierMaxDecompress(g,w=64):
    for a in g:
        f = [0 for i in range(w)]
        for e in a:
            f[e[0]%w] = e[1]
        d = np.fft.ifft(f)
        for s in d:
            yield s






def dpcm_linAcc_c(r):
    return lambda a,d:a+d*r

def dpcm_expAcc_c(r):
    m = math.exp(-r)
    def r(a,d):
        return (a-d)*m+d
    return r

def dpcm_satLinAcc_c(r,lo=-1,hi=1):
    f = o(lambda x: max(lo,min(hi,x)))
    return lambda a,d:f(a+d*r)

def to_dpcm(g,oversample=1,accFunc=dpcm_linAcc_c(.01)):
    a = 0
    for v in g:
        for i in range(oversample):
            d = (2*(a.real<v.real)+2j*(a.imag<v.imag)-1-1j)
            yield d
            a = accFunc(a,d)
            
def from_dpcm(g,oversample=1,accFunc=dpcm_linAcc_c(.01)):
    a = 0
    i = 0 
    for v in g:
        a = accFunc(a,v)
        i = (i+1)%oversample
        if i==0:
            yield a



#some other fft based shenanigans
def explode(arr,lowf=110,highf=14080,ratio=2,sr=48000):
    f = np.fft.fft(arr)
    return (np.fft.ifft(a) for a in explodef(f,lowf,highf,ratio,sr))

def explodef(four,lowf=110,highf=14080,ratio=2,sr=48000):
    fs = (len(four)/sr)*lowf
    fh = (len(four)/sr)*highf
    for i in explodefs(four,fs,fh,ratio):
        yield i
def explodefs(four,fs,fh,ratio=2,pad_to_pow_2=True):
    l = int(fs)*2-1
    la = 1<<((l-1).bit_length())
    yield [four[i] for i in range(int(fs))]+[0 for i in range((la-l)*pad_to_pow_2)]+[four[len(four)-i] for i in range(1,int(fs))]
    fp = fs
    zerosarr = [0 for i in range(1,int(fs))]
    while fs < fh:
        fs *= ratio
        l = (int(fs)-int(fp))*2+1+len(zerosarr)*2
        la = 1<<((l-1).bit_length())
        yield [0] + zerosarr + [four[i] for i in range(int(fp),int(fs))] + \
             [0 for i in range((la-l)*pad_to_pow_2)] + [four[len(four)-i] for i in range(int(fp),int(fs))]+zerosarr
        zerosarr += [0 for i in range(int(fp),int(fs))]
        fp = fs
    l = len(four)
    la = 1<<((l-1).bit_length())
    yield [0] + zerosarr + [four[i] for i in range(int(fp),len(four)//2)]+[0 for i in range((la-l)*pad_to_pow_2)]+[four[i] for i in range(len(four)//2,len(four)-int(fp)+1)]+zerosarr


def sinc(x):
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

def then(*gs):
    for g in gs:
        for v in g:
            yield v
def limit(g,l):
    for i in range(l):
        yield next(g)
def sinclowPass(g,f,ffrange=4,sr=48000,window=blackman_window,filtFunc=sinc):
    f/=sr
    g = then(g,(0 for i in range(ffrange-1)))
    lookahead = [next(g) for i in range(ffrange)]
    buf = [0 for i in range(ffrange-1)]+lookahead
    i = 0
    coef = [filtFunc((i-ffrange+1)*f)*window(i/len(buf)) for i in range(len(buf))]
    for v in g:
        r = 0
        for o in range(len(buf)):
            r += buf[o]*coef[(o-i)%len(buf)]
        yield r
        buf[i] = v
        i = (i+1)%len(buf)
def resample(g,rate,ffrange=3,filtFunc = sinc,windowFunc=blackman_window):
    g = then(g,(0 for i in range(ffrange-1)))
    lookahead = [next(g) for i in range(ffrange)]
    fact = 1 if rate <= 1 else 1/rate
    buf = [0 for i in range(ffrange-1)]+lookahead
    i = 0
    coef = lambda x: filtFunc(x*fact)*windowFunc(.5+x/len(buf))
    suboffs = 0
    for v in g:
        while suboffs < 1:
            r = 0
            for o in range(len(buf)):
                r += buf[o]*coef((o-i)%len(buf)-ffrange+1-suboffs)
            yield r
            suboffs += rate
        suboffs -= 1
        buf[i] = v
        i = (i+1)%len(buf)
def rationalResample(g,num,denom,ffrange=3,shift=0,doGCD=True,filtFunc = sinc,windowFunc=blackman_window):
    #denom is in sr, num is desired sr
    #a positive shift shifts the resulting signal earlier by shift samples in the input
    # ie, shift is - delay
    if doGCD:
        gcd = math.gcd(num,denom)
        num //= gcd
        denom //= gcd

    g = then((0 for i in range(ffrange)),g,(0 for i in range(ffrange)))
    sshift = int(math.floor(shift))
    if sshift < 0:
        #pad zeros onto start of g
        def gf(en,s):
            delayBuf = [0 for i in range(s)]
            i = 0
            for v in en:
                yield delayBuf[i]
                delayBuf[i] = v
                i = (i+1)%s
        g = gf(g,-sshift)
    elif sshift > 0:
        #look at future of g
        def gf(en,s):
            i = 0
            for v in en:
                if i >= s:
                    yield v
                    break
                i += 1
            for v in en:
                yield v
            for v in range(i):
                yield 0
        g = gf(g,sshift)
    
    buf = [next(g) for i in range(ffrange*2)]

    shift = shift%1

    size = max(num,denom)
    fact = min(num,denom)/size
    if num>=denom:
        shift /= fact

    #rate of input is denom
    #upconvert to rate: num*denom
    #then low pass
    #then downconvert to rate num

    #so, with a buffer of size ffrange*2 of rate denom, we need coefficients for the same range on the upconverted area
    # ie we need num coefficients for each buffer cell
    clen = ffrange*2*num
    #now then, the low pass frequency is min(1/num,1/denom)
    lpf = max(denom,num)
    #this is the divisor for the coordinate

    #the shift is in input samples i.e. shift*num samples in the fast space
    fshift = (shift*num)%denom
    coord = lambda n: (fshift+ n -(clen//2)+1)/lpf
    #the midpoint of the kernal is n = (clen//2)-1
    wcoord = lambda n: ((fshift+ n +1)/(clen))
    #calculate the windowed sinc kernal
    coef = [filtFunc(coord(i))*windowFunc(wcoord(i)) for i in range(clen)]
    print([round(c,3) for c in coef])
    #now upconvert by num, run the filter in fast space, and downconvert by 1/denom
    #ie, each sample turns into num fast-samples, and each denom fast-sample is taken
    suboffs = int(shift*num/denom)
    i = 0
    for v in g:
        #calc and output value
        while suboffs < num:
            r = 0
            for o in range(len(buf)):
                # buf[i+ffrange] is the center input
                # the midpoint of the kernal is at (clen//2)-1 in coef
                #   and at (t*num + suboffs) in fast space
                # buf[(i+ffrange)%len(buf)] multiplies with coef[(clen//2)-1] when suboffs == 0
                # buf[(i+ffrange+offs)%len(buf)] with coef[(clen//2)-1+num*offs]
                # buf[(i+o)%len(buf)] with coef[-suboffs+o*num-1]
                #len(coef) is ffrange*2*num
                r += buf[(i+ffrange+o)%len(buf)]*coef[-suboffs+o*num]
            yield r
            suboffs += denom
        #collect into buffer
        buf[i] = v
        i = (i+1)%len(buf)
        suboffs -= num
        
def rationalBand(g,low=-1,high=1,denom=2,prec=3,filtFunc = sinc,windowFunc=blackman_window):#extracts the band of the signal between low and high and downsamples (and shifts frequency)
    bandwidth = high-low #decimation ratio is bandwidth/denom

    middle = high+low
    bandwidth *= 2
    denom *= 2

    def shift(g):
        mul = eone**(1j*-middle/denom)
        f = 1
        for v in g:
            yield v*f
            f *= mul
    shifted = shift(g)
    #print(middle,bandwidth,denom)
    for v in rationalResample(shifted,bandwidth,denom,prec,0,True,filtFunc,windowFunc):
        yield v
def rationalUnband(g,low=-1,high=1,denom=2,p=3,ff=sinc,wf=blackman_window):
    #just "zoom out"
    bandwidth = high-low #decimation ratio is bandwidth/denom
    #(p-m)/bw = (-.5-m)/(bw/d) = (-1-2((h+l)/2d))/(2(bw/d))
    # = (-d-((h+l)))/(bw*2)
    for v in rationalBand(g,(-denom-high-low),denom-high-low,2*bandwidth,p,ff,wf): 
        yield v

def splitToBands(g,bands,p=3):
    pass

        
        
#scipy rational resample
import scipy.signal
def srationalResample(g,n,d=48000,o=8):
    gcd = math.gcd(n,d)
    n //= gcd
    d //= gcd
    n *= o
    d *= o
    buf = [0 for i in range(d)]
    dest = []
    di = 0
    bi = 0
    for v in g:
        buf[bi] = v
        bi += 1
        if bi >= len(buf):
            bi = 0
            while di < len(dest):
                yield dest[di]
                di += 1
            dest = scipy.signal.resample(buf,n)
            di = 0


#some time-stretch things
def sep(g,f,*args):
    gr,gi = split(g)
    gr = (v.real for v in gr)
    gi = (v.imag for v in gi)
    fr = f(gr,*args)
    fi = f(gi,*args)
    for v in fr:
        yield v+1j*next(fi)
    
        
def zcstretch(g,r=.5):
    for v in sep(g,zcstretch_,r):
        yield v
def zcstretch_(g,r=.5):
    buf = [0]
    l = 0
    t = 0
    prev = next(g)
    for v in g:
        buf[l] = v
        l += 1
        t -= 1
        if l >= len(buf):
            buf += buf
        if v > 0 and prev <= 0:
            i = 0
            while t < 0:
                for i in range(l):
                    yield buf[i]
                    t += r
            l = 0
        prev = v

def resizeFFT(a,ds):
    if ds < 0:
        return a[:(len(a)//2)+ds//2]+a[(len(a)//2)-ds+ds//2:]
    else:
        return a[:(len(a)//2)] + [0 for i in range(ds)] + a[(len(a)//2):]
def resizeFFT(a,ds):
    if ds < 0:
        return a[:(len(a)//2)+ds//2]+a[(len(a)//2)-ds+ds//2:]
    else:
        return a[:(len(a)//2)] + [0 for i in range(ds)] + a[(len(a)//2):]
    
#expensive fft-based pitch-scaler idea
def offtps(g,fftsize = 256,resize = -128,winFunc = blackman_window):
    buf = [0 for i in range(fftsize//2)]+[next(g) for i in range(fftsize-(fftsize//2))]
    window = [winFunc(2*(i-fftsize/2)/fftsize) for i in range(fftsize)]
    tmp = [i for i in buf]
    ind = 0
    for v in g:
        for i in range(fftsize):
            tmp[i] = window[i%fftsize]*buf[(i+ind)%fftsize]
        ftmp = np.fft.fft(tmp)
        fres = ftmp[:(fftsize+resize)]
        
        
        buf[ind] = v
        ind = (ind+1)%len(buf)







#moving taps thingy
def movtaps(arr,rate=.5,pos=0,tapsep = 960,kern = None):
    if kern == None:
        kern = [i/tapsep for i in range(tapsep)]+[i/tapsep for i in range(tapsep,0,-1)]
    tapbase = int(pos/tapsep)*tapsep
    taps = [tapbase+tapsep*i for i in range(1+len(kern)//tapsep)]
    tapmod = len(taps)*tapsep
    while 1:
        p = int(pos)
        v = 0
        for i in range(len(taps)):
            if taps[i] <= p-tapsep:
                taps[i] += tapmod
            if p+len(kern) > taps[i] >= p:
                v += kern[taps[i]-p]*arr[taps[i]]
            taps[i] += 1
            if taps[i] >= p+len(kern):
                taps[i] -= tapmod
        yield v
        pos += rate

from wavelet import RingQueue
from random import random

class tapview_even:
    def __init__(self,spacing,offset=0):
        self.spacing = spacing
        self.offset = offset
    def get_taps(self,low,high):
        t = self.spacing*((low-self.offset)//self.spacing)+self.offset
        if t<low:
            t += self.spacing
        while t < high:
            yield (t,1)
            t += self.spacing
    def advance(self,n):
        self.offset = (self.offset+n)%self.spacing

class tapview_rand:
    def __init__(self,spacing):
        self.spacing = spacing
        self.prevRegion = [0,0]
        self.activeTaps = RingQueue()
    def get_taps(self,low,high):
        if low > self.prevRegion[1] or high < self.prevRegion[0]:
            self.activeTaps = RingQueue()
            for i in range(int(low),int(high)):
                if random()*self.spacing < 1:
                    self.activeTaps.push(i)
        else:
            if low < self.prevRegion[0]:
                #going down, taps are pushed in ascending order, so we must rpush
                for i in range(int(self.prevRegion[0])-1,int(low)-1,-1):
                    if random()*self.spacing < 1:
                        self.activeTaps.rpush(i)
            if high > self.prevRegion[1]:
                #going up
                for i in range(int(self.prevRegion[1]),int(high)):
                    if random()*self.spacing < 1:
                        self.activeTaps.push(i)
        while self.activeTaps.has() and self.activeTaps[-1] > high:
            self.activeTaps.rdeque()
        while self.activeTaps.has() and self.activeTaps[0] < low:
            self.activeTaps.deque()
        if len(self.activeTaps) == 1:
            yield (self.activeTaps[0],(high-low)/self.spacing)
        if self.activeTaps.has():
            yield (self.activeTaps[0],(self.activeTaps[0]-low+(self.activeTaps[1]-self.activeTaps[0])/2)/self.spacing)
            yield (self.activeTaps[-1],(high-self.activeTaps[-1]+(self.activeTaps[-1]-self.activeTaps[-2])/2)/self.spacing)
        for i in range(1,len(self.activeTaps)-1):
            yield (self.activeTaps[i],(self.activeTaps[i+1]-self.activeTaps[i-1])/2/self.spacing)
        self.prevRegion = [low,high]
    def advance(self,n):
        self.prevRegion = [r+n for r in self.prevRegion]
        for i in range(len(self.activeTaps)):
            self.activeTaps[i]+=n

from itools import it
def taps(arr,rate=.5,pos=0,tapspeed=1,tapview=tapview_even(1000,0),kern = None):
    rate = it(rate)
    tapspeed = it(tapspeed)
    if kern == None:
        kern = [i/tapview.spacing for i in range(tapview.spacing)]+[i/tapview.spacing for i in range(tapview.spacing,0,-1)]
    while 1:
        p = int(pos)
        v = 0
        for t in tapview.get_taps(pos,pos+len(kern)):
            if 0<=int(t[0]-pos)<len(kern):
                v += kern[int(t[0]-pos)]*arr[int(t[0])]*t[1]
        yield v
        pos += next(rate)
        tapview.advance(next(tapspeed))




def delay(samples=0,wet=1):
    def delayCurry(g,samples=samples,wet=wet):
        if wet == 0 or samples == 0:
            for v in g:
                yield v
            return
        if wet == 1:
            for v in range(samples):
                yield 0
            for v in g:
                yield v
            return
        from wavelet import RingQueue
        buf = RingQueue()
        for i in range(samples):
            v = next(g)
            buf.push(v)
            yield v * (1-wet)
        for v in g:
            yield v*(1-wet)+wet*buf.deque()
            buf.push(v)
    return delayCurry

def schroeder_allpass(gain=.7,t=1051):
    def schroeder_allpassCurry(g):#,gain=gain,t=t):
        buf = [0]*t
        i = 0
        for v in g:
            delayed = buf[i]         #hypothetical for if gain=1
            buf[i] = delayed*gain+v  # buf[i] += v
            yield delayed-buf[i]*gain# -v
            i = (i+1)%t
    return schroeder_allpassCurry

def schroeder_nat_reverb(gain=.7,t=8000,dargs = [(.7,1051),(.7,337),(.7,113)]):
    def schroeder_nat_reverbCurry(g):#,t=t,gain=gain,dargs=dargs):
        totlen = t
        for gn,a in dargs:
            totlen += a+1
        i = 0
        buf = [0]*totlen
        for v in g:
            tap = totlen-1
            for gn,a in dargs: #allpass delays with gain 1
                #time goes down, so above is before is input
                buf[tap-i] += buf[tap-a-i]*gn #input += gn*delayed
                buf[tap-a-i] -= buf[tap-i]*gn #output = delayed-gn*inp
                tap -= a+1
            out = buf[-i]
            buf[-i] = v+out*gain
            i = (i+1)%len(buf)
            yield (1-gain*gain)*out-gain*v
    return schroeder_nat_reverbCurry

def schroeder_nat_reverb_filt(gain=.7,t=8000,fp=lambda x:x,fo=lambda x:x,ff=lambda x:x,dargs = [(.7,1051,lambda x:x,lambda x:x,lambda x:x),(.7,337,lambda x:x,lambda x:x,lambda x:x),(.7,113,lambda x:x,lambda x:x,lambda x:x)]):
    def schroeder_nat_reverbCurry(g):#,t=t,gain=gain,dargs=dargs):
        totlen = t
        for gn,a,p,r,i in dargs:
            totlen += a+1
        i = 0
        buf = [0]*totlen
        for v in g:
            tap = totlen-1
            for gn,a,p,r,i in dargs: #allpass delays with gain 1
                #time goes down, so above is before is input
                buf[tap-i] = i(buf[tap-i]+r(buf[tap-a-i]*gn)) #input += gn*delayed
                buf[tap-a-i] += p(-buf[tap-i]*gn) #output = delayed-gn*inp
                tap -= a+1
            out = buf[-i]
            buf[-i] = v+ff(out*gain)
            i = (i+1)%len(buf)
            yield fo((1-gain*gain)*out)+fp(-gain*v)
    return schroeder_nat_reverbCurry

def schroeder_nat_reverb1(gain=.7,t=8000,dg0 = .7,dd0=1051):
    def schroeder_nat_reverbCurry(g):#,t=t,gain=gain,dargs=dargs):
        totlen = t+dd0+1
        i = 0
        buf = [0]*totlen
        io0 = totlen-1
        io1 = io0-dd0
        for v in g:
            buf[io0-i] += buf[io1-i]*dg0
            buf[io1-i] -= buf[io0-i]*dg0
            out = buf[-i]
            buf[-i] = v+out*gain
            i = (i+1)%len(buf)
            yield (1-gain*gain)*out-gain*v
    return schroeder_nat_reverbCurry

def schroeder_nat_reverb2(gain=.7,t=8000,dg0 = .7,dd0=1051,dg1=.7,dd1=337):
    def schroeder_nat_reverbCurry(g):#,t=t,gain=gain,dargs=dargs):
        totlen = t+dd0+dd1+2
        i = 0
        buf = [0]*totlen
        io0 = totlen-1
        io1 = io0-dd0
        io2 = io1-dd1
        for v in g:
            buf[io0-i] += buf[io1-i]*dg0
            buf[io1-i] -= buf[io0-i]*dg0
            buf[io1-i-1] += buf[io2-i-1]*dg1
            buf[io2-i-1] -= buf[io1-i-1]*dg1
            out = buf[-i]
            buf[-i] = v+out*gain
            i = (i+1)%len(buf)
            yield (1-gain*gain)*out-gain*v
    return schroeder_nat_reverbCurry
def schroeder_nat_reverb3(gain=.7,t=8000,dg0=.7,dd0=1051,dg1=.7,dd1=337,dg2=.7,dd2=113):
    def schroeder_nat_reverbCurry(g):#,t=t,gain=gain,dargs=dargs):
        totlen = t+dd0+dd1+dd2+3
        i = 0
        buf = [0]*totlen
        io0 = totlen-1
        io1 = io0-dd0
        io2 = io1-dd1
        io3 = io2-dd2
        for v in g:
            buf[io0-i] += buf[io1-i]*dg0
            buf[io1-i] -= buf[io0-i]*dg0
            buf[io1-i-1] += buf[io2-i-1]*dg1
            buf[io2-i-1] -= buf[io1-i-1]*dg1
            buf[io2-i-2] += buf[io3-i-2]*dg2
            buf[io3-i-2] -= buf[io2-i-2]*dg2
            out = buf[-i]
            buf[-i] = v+out*gain
            i = (i+1)%len(buf)
            yield (1-gain*gain)*out-gain*v
    return schroeder_nat_reverbCurry

def schroeder_nat_reverb_filt1(gain=.7,t=8000,fb=lambda x:x,dg0 = .7,dd0=1051,dfb=lambda x:x):
    def schroeder_nat_reverbCurry(g):#,t=t,gain=gain,dargs=dargs):
        totlen = t+dd0+1
        i = 0
        buf = [0]*totlen
        io0 = totlen-1
        io1 = io0-dd0
        for v in g:
            buf[io0-i] += dfb(buf[io1-i]*dg0)
            buf[io1-i] -= buf[io0-i]*dg0
            out = buf[-i]
            buf[-i] = v+fb(out*gain)
            i = (i+1)%len(buf)
            yield (1-gain*gain)*out-gain*v
    return schroeder_nat_reverbCurry


import math
def fmeydist(x,f=1,g=1,d=8):
    return g*math.sin(x*f)/math.cosh(x/d) + math.tanh(x)

def greycode(x):
    return x^(x>>1)
def sgreycode(x):
    return greycode(x) if x >= 0 else -greycode(-x)
def fgreycode(x,sf = 1<<32):
    return greycode(int(sf*x))/sf
def fsgreycode(x,sf=1<<32):
    return sgreycode(int(sf*x))/sf

def coshySin(x,f=1,c=1):
    return math.sin(x*f*math.cosh(c*x))

def polarDist(x,mf=lambda x: x,thf = lambda x: x):
    return math.e**(1j*thf(math.atan2(x.imag,x.real)))*mf(abs(x))

def polarSplit(g):
    for v in g:
        yield abs(v)+1j*math.atan2(v.imag,v.real)
def polarMerge(g):
    for v in g:
        yield v.real*math.e**(1j*v.imag)

def cln(x):
    yield math.ln(abs(x))+1j*math.atan2(x.imag,x.real)

def logSplit(g):
    for v in g:
        yield cln(v)
def logMerge(g):
    for v in g:
        yield math.e**v
        
def roundy(x,stepsize=1):
    return round(x/stepsize)*stepsize

def roundyfloat(x,ss=.125):
    return sign(x)*2**(roundy(math.log2(abs(x)),ss)) if x!=0 else 0

def log2spacel(f):
    def do(x):
        return sign(x)*2**(f(math.log2(abs(x))))

def roundish(x,ss=.1,power=3):
    x/=ss
    i = round(x)
    e = x-i
    return ss*(i+sign(e)/2*abs(2*e)**power)
    #maximum gain is when rounding up to 1
    # x < 1
    # i = 1
    # e = x-1 < 0
    # sign(e) = -1
    # gain = max((1-1/2*(2-2x)^pow)/x)
    #            hd = pow*(2-2x)^(pow-1)
    #     d =    (x*pow*(2-2x)^(pow-1)-(1-1/2*(2-2x)^pow))/x^2 = 0
    #            x*pow*(2-2x)^(pow-1) = 1-1/2*(2-2x)^pow
    #            x = ( 1-1/2*(2-2x)^pow )/( pow*(2-2x)^(pow-1) )
    #            x*pow*(2-2x)^(pow-1) + 1/2*(2-2x)^pow = 1
    #            (x*pow + 1-x) * (2-2x)^(pow-1) = 1
    #            (1 + (pow-1)x)* (1-x)^(pow-1) = 2^(1-pow)
    #            u = 1-x
    #            ((1-p)u+p) * u^(p-1) = 2^(1-p)
    #            (1-p)u^p+pu^(p-1) - 2^(1-p) = 0
    #      general root = ???
    


#l section

def compressl(maxMag=1,maxHoldDecay=.9995):
    def do(v,c=[0,maxMag,maxHoldDecay]):
        c[0] = max(abs(v),c[0]*c[2])
        return c[1]*v/c[0] if c[0]>c[1] else v
    return do

def gatel(minMag=.1, magDecay = .999):
    def do(v,c=[0,minMag,magDecay]):
        c[0] = max(abs(v),c[0]*c[2])
        return v if c[0]>c[1] else 0
    return do

def maxholdfuncl(f=lambda v,m:v,decay = .999):
    def do(v,c=[0,decay]):
        c[0] = max(abs(v),c[0]*c[1])
        return f(v,c[0])
    return do

def energy(x):
    return (x*x.conjugate()).real

def maxholdmfuncl(f=lambda v,m:v,decay = .999,m=energy):
    def do(v,c=[0,decay]):
        c[0] = max(m(v),c[0]*c[1])
        return f(v,c[0])
    return do

def phasezcl(f=lambda v,p:v):
    def do(v,c=[0,0]):
        c[1] += abs(sign(c[0].real)-sign(v.real))
        c[1] += 1j*abs(sign(c[0].imag)-sign(v.imag))
        c[0] = v
        return f(v,c[1])
    return do
csign = o(sign)
def freqzcl(f=lambda v,f:v):
    def do(v,c=[0]):
        r = csign(c[0])-csign(v)
        c[0] = v
        return f(v,r)
    return do
def pllzcl(f = lambda v,f:v,ts = .001,tf = .0001):
    def do(v,c=[0,0,0,0,ts,tf]):
        dr = (c[0].real>0)^(v.real>0)
        di = (c[0].imag>0)^(v.imag>0)
        p = 0#phase
        #if dr: #change freq (error expected be 0, if more, then speed up)
        #    c[2] += c[1].real*c[5]
        #if di:
        #    c[3] += c[1].imag*c[5]
        c[1] += dr+1j*di #err
        c[0] = v #prev

        fr = c[2]+1j*c[3]
        c[1] -= fr 

        
        
        if c[1].real < 0:#exponentially slow down freq
            c[2] *= 1-c[4]
        elif c[1].real > 2:
            c[2] += (c[2]+c[1].real)*c[5]
        if c[1].imag < 0:
            c[3] *= 1-c[4]
        elif c[1].imag > 2:
            c[3] += (c[3]+c[1].imag)*c[5]
        return f(v,p+fr)
    return do
        
def pllzclnw(f = lambda v,f:v,tc = .001,hc=3,hf=2,tn=0):
    def hys(x):
        #3 regions: linear, cubic, and flat
        if abs(x) < hf: return 0
        x -= sign(x)*hf
        if abs(x) < hc-hf: return (x/(hc-hf))**3
        x -= sign(x)*(hc-hf)
        return x+sign(x)
    hyst = o(hys)
    def do(v,c=[0,0,0,tc,tn]):
        d = ((c[0].real>0)^(v.real>0))+(1j*((c[0].imag>0)^(v.imag>0)))
        c[1] += d
        c[0] = v
        #have the same number of zero crossings on average
        #cumulative
        h = hyst(c[1])
        c[2] += h*c[3]- c[2]*c[4] #f estimate
        c[1] -= c[2] #error
        return f(v,c[2])
    return do

def erodeml():
    def do(v,p=[100]):
        r = v if abs(v)<abs(p[0]) else p[0]
        p[0] = v
        return r
    return do
def dialateml():
    def do(v,p=[0]):
        r = v if abs(v)>abs(p[0]) else p[0]
        p[0] = v
        return r
    return do

def maxholdl(tc=.999):
    def do(v,p=[0,tc]):
        if abs(v)>abs(p[0]):
            p[0] = v
        else:
            p[0] *= p[1]
        return p[0]
    return do

def minholdl(tc=1.001):
    def do(v,p=[0,tc]):
        if abs(v)<abs(p[0]):
            p[0] = v
        else:
            p[0] *= p[1]
        return p[0]
    return do




def tritapsl(tr=0,bs=4800):
    #tempo will always be 1:1
    #can only change pitch
    #
    #triangle kernal 2-tapped thingy
    def do(v,c=[0,0,tr],b=[0]*bs):
        b[c[0]] = v
        c[0] = (c[0]+1)%len(b)
        #triangle peak is at c[0]-len(b)/2
        #move taps
        c[1] = (c[1]+c[2])%(len(b)/2)
        # c[0] is newest, c[0]-1 is prev newest
        #
        f = 2*abs((c[1]-c[0])%len(b)-(len(b)/2))/len(b)
        return b[int(c[1])]*(1-f)+b[int(c[1]+len(b)/2)]*f
    return do

def tritapsrcl(bs=4800):
    #tempo will always be 1:1
    #can only change pitch
    #
    #triangle kernal 2-tapped thingy
    def do(v,tr,c=[0,0],b=[0]*bs):
        b[c[0]] = v
        c[0] = (c[0]+1)%len(b)
        #triangle peak is at c[0]-len(b)/2
        #move taps
        c[1] = (c[1]+tr)%(len(b)/2)
        # c[0] is newest, c[0]-1 is prev newest
        #
        f = 2*abs((c[1]-c[0])%len(b)-(len(b)/2))/len(b)
        return b[int(c[1])]*(1-f)+b[int(c[1]+len(b)/2)]*f
    return do
                    
