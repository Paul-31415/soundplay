
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


#https://ccrma.stanford.edu/~jos/pasp/Schroeder_Reverberators.html
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

#stereo-ify through reverb: ((.5+.5j)*(i.real+i.imag)+1*(.5-.5j)*(i.real-i.imag) for i in schroeder_nat_reverb_filt1(.2,fb=lambda x: x*(eone**(-.1j)),dg0=.3,dfb=lambda x:x*(.9+.3j))(g))


#comb filter stereoify
class schroeder_comb_matrix4:
    def __init__(self,g1=.773,l1=1687,g2=.802,l2=1601,g3=.753,l3=2053,g4=.733,l4=2251):
        self.lens = l1,l2,l3,l4
        self.gains = g1,g2,g3,g4
    def __call__(self,g):
        cs = [[0]*l for l in self.lens]
        ci = [0]*4
        for v in g:
            for i in range(4):
                cs[i][ci[i]] *= self.gains[i]
                cs[i][ci[i]] += v*(1-self.gains[i])
                ci[i] = (ci[i]+1)%len(cs[i])
            s1 = cs[0][ci[0]]+cs[2][ci[2]]
            s2 = cs[1][ci[1]]+cs[3][ci[3]]
            yield (s1+s2,s1-s2)

def default_schroeder(g,s=1):
    yield from (a*1j+b for a,b in schroeder_comb_matrix4(l1=int(1687*s),
                                                         l2=int(1601*s),
                                                         l3=int(2053*s),
                                                         l4=int(2251*s))
                (schroeder_nat_reverb3(dd0=int(347*s),dd1=int(113*s),dd2=int(37*s))(g)))



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

def pid(p=1,i=0,d=0):
    def c(v,a=[0,0],t=[p,i,d]):
        a[0] += v
        d = v-a[1]
        a[1] = v
        return v*t[0]+a[0]*t[1]+d*t[2]
    return c
def zc(i=0):
    def do(v,p=[i]):
        r = (v<0)^(p[0]<0)
        p[0] = v
        return r
    return do
def fll_zc(ctl = pid(.1,.1,.1),fmax=1/3):
    zcflt = zc()
    zcvco = zc(-.5)
    def do(v,a=[0,0],e=[0]):
        cross = zcflt(v)
        a[0] = (a[0]+a[1])%1
        vcc = zcvco(a[0]-.5)
        e[0] += cross-(vcc)*((a[1]>=0)*2-1)
        a[1] += ctl(e[0])
        if abs(a[1])>fmax:
            a[1] *= fmax/abs(a[1])
        return a
    return do
def pll_zc(ctl = pid(.1,0,0),fmax=1/3):
    zcflt = zc()
    zcvco = zc(-.5)
    def do(v,a=[0,0]):
        cross = zcflt(v)
        a[0] = (a[0]+a[1])%1
        vcc = zcvco(a[0]-.5)
        a[1] += ctl(cross-(vcc)*((a[1]>=0)*2-1))
        if abs(a[1])>fmax:
            a[1] *= fmax/abs(a[1])
        return a
    return do


def sin2π(x):
    return math.sin(2*math.pi*x)

def vco(mod=1,phase=0):
    def do(v,p=[phase,mod]):
        p[0] = (p[0] + v)%p[1]
        return p[0]
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

def tapsl(tr=0,n=2,bs=4800,kf = lambda f:1-abs(f)):
    #tempo will always be 1:1
    #can only change pitch
    if n == 2 and 0:
        def do(v,c=[0,0,tr],b=[0]*bs):
            b[c[0]] = v
            c[0] = (c[0]+1)%len(b)
            #move taps
            c[1] = (c[1]+c[2])%(len(b)/2)
            # c[0] is newest, c[0]-1 is prev newest
            
            f = 2*abs((c[1]-c[0])%len(b)-(len(b)/2))/len(b)
            return b[int(c[1])]*kf(f)+b[int(c[1]+len(b)/2)]*kf(f-1)
    else:
        def do(v,c=[0,0,tr],b=[0]*bs,n=n):
            b[c[0]] = v
            c[0] = (c[0]+1)%len(b)
            #move taps
            c[1] = (c[1]+c[2])%(len(b)/n)
            # c[0] is newest, c[0]-1 is prev newest

            i = c[0]+c[1]
            f = 2*c[1]/len(b)-1
            r = b[int(i)%len(b)]*kf(f)
            for t in range(1,n):
                i += len(b)/n
                f += 2/n
                r += b[int(i)%len(b)]*kf(f)
            return r
    return do
def tapsdl(tr=0,n=2,bs=4800,kf = lambda v,f:v*(1-abs(f))):
    #tempo will always be 1:1
    #can only change pitch
    if n == 2 and 0:
        def do(v,c=[0,0,tr],b=[0]*bs):
            b[c[0]] = v
            c[0] = (c[0]+1)%len(b)
            #move taps
            c[1] = (c[1]+c[2])%(len(b)/2)
            # c[0] is newest, c[0]-1 is prev newest
            
            f = 2*abs((c[1]-c[0])%len(b)-(len(b)/2))/len(b)
            return kf(b[int(c[1])],f)+kf(b[int(c[1]+len(b)/2)],f-1)
    else:
        def do(v,c=[0,0,tr],b=[0]*bs,n=n):
            b[c[0]] = v
            c[0] = (c[0]+1)%len(b)
            #move taps
            c[1] = (c[1]+c[2])%(len(b)/n)
            # c[0] is newest, c[0]-1 is prev newest

            i = c[0]+c[1]
            f = 2*c[1]/len(b)-1
            r = kf(b[int(i)%len(b)],f)
            for t in range(1,n):
                i += len(b)/n
                f += 2/n
                r += kf(b[int(i)%len(b)],f)
            return r
    return do

def tapsmpll(tr=0,n=2,bs=4800,kf = None):
    if kf == None:
        kf = lambda v,f:v*(1-abs(f)) #triangle
        kf = lambda v,f,n=n: v if f == 0 else v*math.sin(math.pi*n/2*f)/(math.pi*n/2*f) #sinc
    #tempo will always be 1:1
    #can only change pitch
    c = [0,0,tr]


    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    plt.figure()
    axm = plt.axes([0.1, 0.3, .8, .4])
    sm = Slider(axm, 'pitch adjust', -5, 3, valinit=tr)
    def update(val,d=c):
        d[2] = sm.val
    sm.on_changed(update)
    plt.show(block=0)
    def do(v,c=c,b=[0]*bs,n=n,s=update):
        b[c[0]] = v
        c[0] = (c[0]+1)%len(b)
        #move taps
        c[1] = (c[1]+c[2])%(len(b)/n)
        # c[0] is newest, c[0]-1 is prev newest
        
        i = c[0]+c[1]
        f = 2*c[1]/len(b)-1
        r = kf(b[int(i)%len(b)],f)
        for t in range(1,n):
            i += len(b)/n
            f += 2/n
            r += kf(b[int(i)%len(b)],f)
        return r    

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

def fmask(m=np.ones(4096)):
    def msk(x,m=m):
        a = abs(x)
        l = len(a)//2
        a[1:l] += a[-1:-l:-1]
        return a[:l]*m
    return msk

def fourier_max_pitch_estimator(bs=8192,mask =None,window = lambda x: np.cos(x*math.pi*2)*.5+.5):
    if mask is None:
        mask = fmask((lambda x: (x*((.5-x)**1)**(1/10)))(np.arange(bs//2)/bs))
    def do(v,b=np.zeros(bs,dtype=complex),a=[0,.0001],w = window(np.arange(bs)/(bs-1)-.5)):
        b[a[0]] = v
        a[0] = (a[0]+1)%len(b)
        if a[0] == 0:
            a[1] = np.argmax(mask(scipy.fft.fft(b*w)))/len(b)
        return a[1]
    return do

fmpe = fourier_max_pitch_estimator

def fourier_max_pitch_estimator_crossfade(bs=8192,th=.5,mask =None,window = lambda x: np.cos(x*math.pi*2)*.5+.5):
    if mask is None:
        mask = fmask((lambda x: (x*((.5-x)**1)**(1/10)))(np.arange(bs//2)/bs))
    def do(v,b=np.zeros(bs,dtype=complex),a=[0,.0001,.0001],w = window(np.arange(bs)/(bs-1)-.5)):
        b[a[0]+(len(b)//2)] = v
        a[0] = (a[0]+1)%(len(b)//2)
        if a[0] == 0:
            v = mask(scipy.fft.fft(b*w))
            i = np.argmax(v)
            a[2],a[1] = i/len(b) if v[i]>th else a[2],a[2]
            b[:(len(b)//2)] = b[(len(b)//2):]
        return a[2]*(2*a[0]/len(b))+a[1]*(1-2*a[0]/len(b))
    return do

fmpec = fourier_max_pitch_estimator_crossfade


def followl(vf=lambda o,d: o*.995+d*(abs(d)-.1)*.01):
    def do(v,p=[0,0],vf=vf):
        p[1] = vf(p[1],v-p[0])
        p[0] += p[1]
        return p[0]
    return do

lerp = lambda a,b,x: a*(1-x)+b*x

sc = followl(lambda o,d: lerp(o,d,max(0.01,1.0-abs(d)*6.0)))
sc2 = followl(lambda o,d: lerp(o,d/(abs(d)+.1),max(0.05,1.0-abs(d))))
ah = followl(lambda o,d: o*.999+d*.01/(abs(d)+1e-10))




def swapperl(sg=[12000],i0=0):
    def do(a,b,i=[0,i0],sg=sg,w=[0]):
        i[1] += 1
        if i[1] > sg[i[0]]:
            i[1] = 0
            i[0] = (i[0]+1)%len(sg)
            w[0] = not w[0]
        if w[0]:
            return b
        return a
    return do
        


from random import random
def embiggen(a,l,n=10,fd = lambda i: .995+random()/100,md = lambda i: (random()*2+random*2j)-(1+1j)):
    r = a*l



def pred_nearest():
    def p(v):
        return v
    return p
def pred_kernel(k=[2,-1]):
    def p(v,h=[0]*len(k),k=k,o=[0]):
        h[o[0]] = v
        v *= k[0]
        for i in range(1,len(k)):
            v += h[(o[0]+i)%len(k)]*k[i]
        o[0] = (o[0]-1)%len(k)
        return v
    return p
    


    
def ms_slowest(invert=False):
    def f(a,b,p=[0],i=invert):
        w = (abs(a-p[0])>abs(b-p[0]))^i
        p[0] = [a,b][w]
        return w
    return f
def arg(n):
    return math.atan2(n.imag,n.real)
def ms_least_spinny(invert=False):
    def f(a,b,p=[0],i=invert,p2=2*math.pi):
        w = (((arg(a)-p[0])%p2)>((arg(b)-p[0])%p2))^i
        p[0] = arg([a,b][w])
        return w
    return f
def ms_predicted(predictor=lambda x: x,invert=False):
    def f(a,b,p=[0],pd=predictor,i=invert):
        pred = pd(p[0])
        w = (abs(a-pred)>abs(b-pred))^i
        p[0] = [a,b][w]
        return w
    return f
def ms_length():
    def f(a,b,p=[0,0]):
        which = p[0]>p[1]
        s = [a,b][which]
        p[which] += abs(s)
        return which
    return f
def ms_dlength():
    def f(a,b,p=[0,0,0,0]):
        which = p[2]>p[3]
        s = [a,b][which]
        p[2+which] += abs(s-p[which])
        p[which] = s
        return which
    return f

def merge_sort(g1,g2,f=ms_slowest(),emit_all=True):
    v2 = None
    v1 = None
    g = None
    while 1:
        if v1 == None:
            try:
                v1 = next(g1)
            except StopIteration:
                v = v2
                g = g2
                break
        if v2 == None:
            try:
                v2 = next(g2)
            except StopIteration:
                v = v1
                g = g1
                break
        if f(v1,v2):
            v = v2
            v2 = None
        else:
            v = v1
            v1 = None
        yield v
    if emit_all:
        if v != None:
            yield v
        for v in g:
            yield v
            
            
def merge_sort_deriv_l(*g):
    lens = [0]*len(g)
    vals = [0]*len(g)
    out = 0
    


            

def exp_dpcm_encode(a=0.01,h=1,l=-1):
    def enc(v,p=[0],a=a,d=h-l,l=l):
        r = l+1j*l+((v.real>p[0].real)+1j*(v.imag>p[0].imag))*d
        p[0] -= (p[0]-r)*a
        return r
    return enc
def exp_dpcm_decode(a=0.01,h=1,l=-1):
    def dec(v,p=[0],a=a):
        p[0] += (v-p[0])*a
        return p[0]
    return dec
        
    



def twang(blen=128,bfill=lambda i: ((i**.5)%.1)*20-1+((i**.33)%.1)*20j-1j,filt=lambda x:x*.9,offset=0):
    buf = [bfill(i) for i in range(blen)]
    i = 0
    while 1:
        yield buf[i]
        buf[(i-offset)%blen] = filt(buf[i])
        i = (i+1)%blen




cround = o(round)
def fpiir1l(a1=0,b0=64,b1=0,point=6,mask=0xff,acc=0):
    def do(v,a=[a1],b=[b0,b1],s=[acc]):
        p = s[0]
        s[0] = ((round(v*(1<<point))&mask)-((a[0]*p)>>point))
        return (((b[0]*s[0])>>point)+((b[1]*p)>>point))&mask
    return do


def btwangf(s=128):
    def adj(a,b):
        if a > b:
            return a-1
        elif a < b:
            return a+1
        return a
    def ca(a,b):
        return adj(a.real,b.real)+1j*adj(a.imag,b.imag)
    def do(v,p=[0],s=s):
        pr = p[0]
        p[0] = cround(v*s)
        return ca(p[0],pr)/s
    return do








def kspr(k=.1,d=.99):
 def do(v,k=k,d=d,x=[0,0]):
  x[1] *= d
  x[1] -= (x[0]-v)*k
  x[0] += x[1]
  return x[0]
 return do

def kine(force):
 def do(v,f=force,x=[0,0]):
     x[1] += f(v,*x)
     x[0] += x[1]
     return x[0]
 return do

def magclip(v,m=1):
    if (a:=abs(v))>m:
        return m*v/a
    return v

def stsd(f=.001,d=.0001):
    return kine(lambda a,b,v: magclip((a-b),f)-magclip(v,d))

def kthing(p=2,d=3):
    kine(lambda a,b,v: magclip((a-b)*(abs(a-b)**p),1)-magclip(v,min(1,abs(a-b)**d)))



def flutter_li(gen,fgen):
    p = next(gen)
    n = next(gen)
    t = 0
    for v in fgen:
        t += v
        while t >= 1:
            p,n = n,next(gen)
            t -= 1
        yield t*n+p*(1-t)

def cflutter_c(ff):
    def cflutter(gen,fgen,*a):
        from itertools import tee
        g1,g2 = tee(gen)
        f1,f2 = tee(fgen)
        gr = ff((i.real for i in g1),(i.real for i in f1),*a)
        gi = ff((i.imag for i in g2),(i.imag for i in f2),*a)
        for v in gr:
            yield v+1j*next(gi)
    return cflutter
cflutter_li = cflutter_c(flutter_li)

def flutter_2i(gen,fgen):
    p = next(gen)
    c = next(gen)
    n = next(gen)
    t = 0
    for v in fgen:
        t += v
        while t >= 1:
            p,c,n = c,n,next(gen)
            t -= 1
        #fit from mathematica with x mat:
        #[1 -1  1]          [0   1   0]
        #[1  0  0]  -inv->  [-.5 0  .5]
        #[1  1  1]          [.5 -1  .5]
        #so, C = c
        #    B = .5(n-p)
        #    A = .5(n+p)-c
        yield t*t*(.5*(n+p)-c)+t*.5*(n-p)+c
cflutter_2i = cflutter_c(flutter_2i)

def flutter_3i(gen,fgen):
    p = next(gen)
    c = next(gen)
    n = next(gen)
    o = next(gen)
    t = 0
    for v in fgen:
        t += v
        while t >= 1:
            p,c,n,o = c,n,o,next(gen)
            t -= 1
        #fit from mathematica with x mat:
        #[1 -1  1 -1]          [0    1    0    0 ]
        #[1  0  0  0]  -inv->  [-1/3 -1/2 1 -1/6 ]
        #[1  1  1  1]          [1/2 -1   1/2   0 ]
        #[1  2  4  8]          [-1/6 1/2 -1/2 1/6]
        #so, D = c
        #    C = n-(2p+3c+o)/6
        #    B = (n+p)/2-c
        #    A = (3c+o-3n-p)/6
        yield ((((c-n)/2+(o-p)/6)*t-c+(n+p)/2)*t+n-c/2-o/6-p/3)*t+c
cflutter_3i = cflutter_c(flutter_3i)

def flutter_4i(gen,fgen):
    r = next(gen)
    p = next(gen)
    c = next(gen)
    n = next(gen)
    o = next(gen)
    t = 0
    for v in fgen:
        t += v
        while t >= 1:
            r,p,c,n,o = p,c,n,o,next(gen)
            t -= 1
        yield ((((c/4-n/6+o/24-p/6+r/24)*t-n/6+o/12+p/6-r/12)*t-5*c/4+2*n/3-o/24+2*p/3-r/24)*t+2*n/3-o/12-2*p/3+r/12)*t+c 
cflutter_4i = cflutter_c(flutter_4i)
def flutter_5i(gen,fgen):
    r = next(gen)
    p = next(gen)
    c = next(gen)
    n = next(gen)
    o = next(gen)
    v = next(gen)
    t = 0
    for v in fgen:
        t += v
        while t >= 1:
            r,p,c,n,o,v = p,c,n,o,v,next(gen)
            t -= 1
        yield c+t*(-c/3+n-o/4-p/2+r/20+v/30+t*(-5*c/4+2*n/3-o/24+2*p/3-r/24+t*(5*c/12-7*n/12+7*o/24-p/24-r/24-v/24+t*(c/4-n/6+o/24-p/6+r/24+t*(-c/12+n/12-o/24+p/24-r/120+v/120)))))
cflutter_5i = cflutter_c(flutter_5i)

def flutter_ni_c(hfa=[]):
    def flutter(gen,fgen,hfa=hfa):
        l = len(hfa)
        a = [0]*l
        t = 0
        ai = 0
        for v in fgen:
            t += v
            while t >= 1:
                a[ai] = next(gen)
                ai = (ai+1)%l
                t -= 1
            r = 0
            for j in range(l):
                r *= t
                for i in range(l):
                    r += a[(ai+i+1)%l]*hfa[j][i]
            yield r

def gaus(x):
    return np.exp(-x*x)

def flutter_sinc(gen,sgen,res=3,sds=3):
    vals = np.zeros(res*sds,dtype=complex)
    times = np.arange(res*sds)-((res*sds)>>1)
    filt = lambda t: np.sinc(t)*gaus(t/res)
    t = 0
    for s in sgen:
        t += s
        while t >= 1:
            vals[:-1] = vals[1:]
            vals[-1] = next(gen)
            t -= 1
        yield np.dot(vals,filt(times-t))  
cflutter_sinc = cflutter_c(flutter_sinc)

"""
def flutter_sinc(gen,sgen,res=36,num=16):
    times = np.zeros(num)
    vals = np.zeros(num,dtype=complex)
    filt = lambda t: np.sinc(t)*np.exp(-(t*t/res))
    for s in sgen:
        
        
        yield np.dot(vals,filt(times))
"""















        
import struct
def remExp(x):
    x = float(x)
    return struct.unpack('<d',((1 .from_bytes(struct.pack('<d',x),'little')&0x800fffffffffffff)|0x3fe0000000000000).to_bytes(8,'little'))[0]



















import numpy as np
import scipy as sp

def make_square_note(p=110,l=8192,e=1):
    pf = math.ceil(p/2)*2
    lf = round(l*pf/p)
    i = np.arange(lf)
    nb = ((i%pf)<=pf/2)*2.-1
    nb *= np.exp(-e*i/lf)*(1-i/lf)
    return sp.signal.resample(nb,l)
def make_square_notes(l=8192,n=128,e=1):
    import numpy as np
    import scipy as sp
    notes = np.zeros((128,l),dtype=float)
    for i in range(128):
        p = 48000/(440*2**((i-64)/12))
        notes[i,:] = make_square_note(p,l,e)
    return notes

class note_seq_thingy:
    def __init__(self,notes=None):
        if notes is None:
            notes = make_square_notes()
        self.notes = notes
        self.buf = np.zeros((notes.shape[1]),dtype=complex)
    def gen(self,g):
        def then(a,b):
            yield from a
            yield from b
        g = then(g,(0 for i in range(len(self.buf))))
        for i in range(len(self.buf)):
            self.buf[i] = next(g)
        for v in g:
            if self.buf[0] == 0:
                yield (0,0)
                self.buf[:-1] = self.buf[1:]
            else:
                res = self.buf-self.notes/np.array([self.notes[:,0]]).T*self.buf[0]
                #energy removal:
                rem = np.sum((res*res.conjugate()).real,axis=1)
                #minimax:
                #rem = np.max(np.array([np.max(np.abs(res.real),axis=1),np.max(np.abs(res.imag),axis=1)]),axis=0)
                im = np.argmin(rem)
                yield (self.buf[0]/self.notes[im,0],im)
                self.buf[:-1] = res[im,1:]
            self.buf[-1] = v
    def ungen(self,g):
        for v in g:
            m,f = v
            self.buf += m*self.notes[f,:]
            yield self.buf[0]
            self.buf[:-1] = self.buf[1:]
            self.buf[-1] = 0
        for i in range(len(self.buf)-1):
            yield self.buf[i]
        self.buf[:] = 0
    





        
#amps
def slew_budget_amp(maxp=1000,recharge_rate=.05):
    def do(v,a=[maxp,0]):
        d = v-a[1]
        dm = abs(d)
        if dm > a[0]:
            dd = d/dm
            v = a[1]+dd*a[0]
            a[0] = 0
        else:
            a[0] -= dm
        a[0] = min(maxp,a[0]+recharge_rate)
        a[1] = v
        return v
    return do

def tanh_fraction_slew_amp(k=1):
    def do(v,a=[0]):
        d = v-a[0]
        dm = abs(d)
        frac = math.tanh(k*dm)
        d *= frac
        v = a[0]+d
        a[0] = v
        return v
    return do

#lowpasses when k*s < 1
#sounds funky and phasey when k*s > 1
def tanh_slew_amp(k=1,s=1):
    def do(v,a=[0]):
        d = v-a[0]
        dm = abs(d)
        amount = math.tanh(k*dm)*s
        if dm != 0:
            d *= amount/dm
        v = a[0]+d
        a[0] = v
        return v
    return do

def stutterPlayback(g,p=0xc000,l=64):
 s = 1
 c = 0
 for v in g:
  if c <= 0:
   c = l
   s = (s>>1)^((s&1)*p)
  yield v
  c -= 1
  if s&1:
   yield v
   c -= 1






def gTaps(gen,tap_speed = 1,progress_speed = 1,buflen = 4096,bufOverLen=4800000,wf = lambda x:.5+.5*math.cos(2*math.pi*x)):
    buf = [0]*buflen
    bufi = 0
    ps = genify(progress_speed)
    ts = genify(tap_speed)
    tapp = 0
    p = 0
    while 1:
        tapp += next(ts)
        dp = next(ps)
        p += dp
        while int(p) > 0:
            if bufi == len(buf):
                buf += [next(gen)]
            else:
                buf[bufi] = next(gen)
            bufi = (bufi+1)%bufOverLen
            tapp -= 1
            p -= 1
        
        tapp %= buflen//2
        vtapp = (int(tapp)-int(p))%(buflen//2)
        weight = wf(tapp/buflen)
        yield buf[(vtapp+bufi+int(p)-buflen)%len(buf)]*(1-weight)+\
            buf[(vtapp+bufi+int(p)-buflen//2)%len(buf)]*weight

class relay:
    def __init__(self,source=0):
        self.source = source
    def __next__(self):
        try:
            return next(self.source)
        except:
            return self.source

def loopback(cont):
    v = [0]
    def loop(v=v):
        while 1:
            yield v[0]
    gen = cont(loop())
    for r in gen:
        v[0] = r
        yield r
#eg: loopback(lambda l,f=filt.iir1l(-.99999,10,-10): gTaps(ll.play(),1,(1-f(abs(v)) for v in l),1<<12))





def squishWave(wave,n=1000,span=lambda w:np.max(w.view(float))-np.min(w.view(float))):
    wave = np.array(wave,dtype=complex)
    ftr = np.fft.rfft(wave.real)
    fti = np.fft.rfft(wave.imag)
    #for stereo, restrict the real and imag components to have the same relative phase
    # and also, the amplitudes must be the same
    best = wave
    bestS = span(best)
    l = len(wave)
    pert = ftr.copy()
    pv = pert.view(float)
    #print(ftr,fti)
    for i in range(n):
        pv[:] = np.random.normal(0,1,pv.shape)
        pert[0] = pert[0].real
        pert[-1] = pert[-1].real
        pert /= np.abs(pert)
        #print(pert,np.abs(pert))
        #pert[-1:l//2:-1] = pert[1:l//2].conjugate()
        guess = np.fft.irfft(ftr*pert)+1j*np.fft.irfft(fti*pert)
        if (s:=span(guess)) <= bestS:
            bestS = s
            best = guess
        #print(s,"    ",end="\r")
    #print("")
    return best
def scrambleWave(wave):
    return squishWave(wave,1,lambda x:1)
def detuneWave(wave,detune=-1,mult=16):
    ft = np.fft.fft(wave)
    rft = np.zeros(len(ft)*mult,dtype=complex)
    d = mult+detune
    hl = len(ft)//2
    rft[:d*hl:d] = ft[:hl]*mult
    rft[1-d*hl::d] = ft[hl:]*mult
    return np.fft.ifft(rft)
def chirp_r_wave(wave,s=0):
    ft = np.fft.rfft(wave)
    r = [ft[0]]
    for i in range(1,len(ft)):
        r += [0]*(i+s)
        r += [ft[i]]
    return np.fft.irfft(r)*len(r)/len(ft)
def rchirp_r_wave(wave,s=0):
    ft = np.fft.rfft(wave)
    r = [ft[0]]
    for i in range(1,len(ft)):
        r += [0]*(len(ft)-1-i+s)
        r += [ft[i]]
    return np.fft.irfft(r)*len(r)/len(ft)
def harm_distort_plot(wave):
    fft = np.fft.rfft(wave)
    linear = np.arange(len(fft))/len(fft)
    quad = linear*linear
    cubic = quad*linear
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    plt.figure()
    axp = plt.axes([0.05, 0.15, .9, .8])
    axl = plt.axes([0.1, 0.025, .8, .025])
    axq = plt.axes([0.1, 0.05, .8, .025])
    axc = plt.axes([0.1, 0.10, .8, .025])
    sl = Slider(axl, 'linear shift', -5, 5, valinit=0)
    sq = Slider(axq, 'quadratic shift', -50, 50, valinit=0)
    sc = Slider(axc, 'cubic shift', -500, 500, valinit=0)
    terms = [0,0,0]
    lineplot, = axp.plot(np.arange(len(wave)),np.array(wave))
    def replot():
        lineplot.set_ydata(np.fft.irfft(np.exp((terms[0]*linear+terms[1]*quad+terms[2]*cubic)*1j)*fft))
    def update(i,s,ngc=[]):
        def do(val,i=i,sld=s):
            terms[i] = sld.val*len(wave)
            replot()
        ngc += [do]
        return do
    sl.on_changed(update(0,sl))
    sq.on_changed(update(1,sq))
    sc.on_changed(update(2,sc))
    plt.show(block=0)
    return [plt,update,replot]



#do Energy transfer filters? https://core.ac.uk/reader/9556830
