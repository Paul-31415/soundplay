


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
buffer_size = 1024

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
        
    
