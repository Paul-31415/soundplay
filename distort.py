
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

def sign(v):
    return [0,1,-1][(v>0)+2*(v<0)]

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

