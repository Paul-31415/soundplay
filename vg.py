import math
import random
e = math.exp(1)
eone = math.exp(2*math.pi)
phi = (1+5**.5)/2
ephi = math.exp(2*math.pi/phi)

def bezier(cps,t=None):
    def bez(t):
        bino = 1
        r = 0
        for i in range(len(cps)):
            r += cps[i] * (bino*((1-t)**(len(cps)-i-1))*(t**i))
            bino *= (len(cps)-1-i)/(i+1)
        return r
    if t==None:
        return bez
    return bez(t)

def spiral(pos,rate,t=None):
    def spi(t):
        return pos+(eone**(1j*t))*t*rate
    if t == None:
        return spi
    return spi(t)

def cl_spiral(pos,rate,t=None):
    def spi(t):
        return pos+(eone**(1j*t))*rate*math.sqrt(t)
    if t == None:
        return spi
    return spi(t)
def cd_spiral(pos,rpt,t=None):
    def spi(t):
        return pos+(eone**(1j*math.sqrt(t)))*rpt*math.sqrt(t)
    if t == None:
        return spi
    return spi(t)

def disk(pos,radius,t=None,ph=phi):
    b2 = math.exp(2*math.pi/ph)
    def dsk(t):
        return pos+radius*(eone**(1j*t)+b2**(1j*t))
    if t == None:
        return dsk
    return dsk(t)
def cl_disk(pos,radius,t=None,ph=phi):
    b2 = math.exp(2*math.pi/ph)
    def f(t):
        return (ph/(ph-1)/math.pi)*\
            ((\
              math.acos(\
                        .5*math.sqrt(\
                                     abs(\
                                         (\
                                          (t/math.pi-4)%8\
                                         )-4\
                                     )\
                        )\
              )*(((math.floor(t/math.pi/4)+1)%2)*2-1)+math.pi*math.floor(t/math.pi/8)\
            ))
    """*(\
               (\
                (t/math.pi/4)%2\
               )*2-1\
            )+\
             math.pi*math.floor(\
                                t/8/math.pi\
             )\
            )"""
    def disk(t):
        #radial slew rate = 2π/r
        #circle(1) + circle(phi)
        v = f(t*(ph-1)/(ph))
        return pos+radius*(eone**(1j*v)+b2**(1j*v))
    if t == None:
        return disk
    return disk(t)

def cl_diskGen(pos,radius,rate=0.01):
    def deriv(t):
        #return abs(1j*math.log(2*math.pi)*(eone**(1j*t))+1j*math.log(math.pi*2/phi)*(ephi**(1j*t)))
        #return (cl_disk(pos,radius,t+.001)-cl_disk(pos,radius,t-.001))*500
        return 1.01+math.cos(2*math.pi*t/phi/phi)
    t = 0
    while 1:
        t += rate/abs(deriv(t))
        yield cl_disk(pos,radius,t)

def magbound(n,m=1):
    if abs(n)>m:
        return n/abs(n)*m
    return n

def magpeg(n,m=1):
    if n==0:
        return m
    return n/abs(n)*m

#use cl_disks to balance dc
def balance(g,mag=.7,rad=.1,dt=.1):
    dc = 0
    f = cl_disk(0,rad)
    t = 0
    for i in g:
        if i == None:
            while abs(dc)>mag:
                v = f(t)-magbound(dc,mag)
                yield v
                t += dt
                dc += v
        else:
            dc += i
            yield i

def lbalance(g,mag=.7,rad=.1,dt=.1):
    dc = 0
    f = cl_disk(0,rad)
    t = 0
    for i in g:
        if i == None:
            v = f(t)-magpeg(dc,mag)
            yield v
            t += dt
            dc += v
        else:
            dc += i
            yield i    

def periodicNones(g,p=1000):
    a = 0
    for i in g:
        yield i
        a += 1
        if a>=p:
            yield None
            a -= p


def point(pos):
    pass

def fibSphere(n):
    points = []
    for i in range(n):
        y = i*2/n-1+1/n
        phi = i*math.pi*(3-2**.5)
        r = math.sqrt(1-y*y)
        points.append([math.cos(phi)*r,y,math.sin(phi)*r])
    return points

def pointsToComplexes(l):
    return [i[0]+1j*i[1] for i in l]

def firework(pos=0,vel=1+.1j,parts=20,size=1,drag=.4,life=2,plife=1,plifeVar=.3,g=-.7,dt=1/48000,stay=40):
    while life > 0:
        pos += vel*dt
        vel += g*dt
        yield pos
        life -= dt
    pts = [[pos,i*size,plife+random.random()*plifeVar] for i in pointsToComplexes(fibSphere(parts))]
    prev = pos
    while len(pts)>0:
        for p in pts:
            if p[2] > 0:
                for i in traverseAlongBeamDump(randBD(prev),randBD(p[0])):
                    yield i

                for j in range(stay):
                    p[0] += p[1]*dt*len(pts)
                    p[1] += g*dt*len(pts)-p[1]*drag*dt*len(pts)
                    p[2] -= dt*len(pts)
                    yield p[0]
                prev = p[0]
        pts = list(filter(lambda x: x[2]>0,pts))

        

def zcont(g):
    for i in g:
        yield i
    while 1:
        yield 0

def ringcont(g,m=1):
    for i in g:
        yield i
    t=0
    while 1:
        t += .2
        yield e**(1j*t)

def rayToXAxis_t(p,r,inf=1.0e300):
    try:
        return -p.real/r.real
    except ZeroDivisionError:
        return inf
def rayToYAxis_t(p,r,inf=1.0e300):
    try:
        return -p.imag/r.imag
    except ZeroDivisionError:
        return inf
def rayToSquareBeamDump(p,r,bd=1+1j):
    dp = bd.real*((r.real<0)*2-1)+1j*bd.imag*((r.imag<0)*2-1)
    t = min(rayToXAxis_t(p+dp,r),rayToYAxis_t(p+dp,r))
    return p+t*r

def randBD(p):
    return rayToSquareBeamDump(p,eone**(1j*random.random()))

def ccros(a,b):
    return a.real*b.imag-b.real*a.imag

def traverseAlongBeamDump(p0,p1,bd=1+1j):
    pts = [bd,-bd.conjugate(),-bd,bd.conjugate()]
    i0,i1=0,0
    for i in range(len(pts)):
        if (p0/pts[i-1]).imag>0 and (p0/pts[i]).imag<0:
            i0 = i
            break
    for i in range(len(pts)):
        if (p1/pts[i-1]).imag>0 and (p1/pts[i]).imag<0:
            i1 = i
            break
    
    if ccros(p0,p1)<0:
        yield p0
        while (i0-i1)%len(pts)>0:
            i0 -= 1
            yield pts[i0%len(pts)]
        yield p1
    else:
        yield p0
        while (i1-i0)%len(pts)>0:
            yield pts[i0%len(pts)]
            i0 += 1
            
        yield p1


class vg:
    def __init__(self):
        pass
    def __call__(self,t):
        return None
    def __len__(self):
        return 0
class point(vg):
    def __init__(self,pos,brightness):
        self.pos = pos
        self.brightness = brightness
    def __call__(self,t):
        if t<self.brightness:
            return self.pos
        return None
    def __len__(self):
        return 1#self.brightness
        return 1

class vg_bezier(vg):
    def __init__(self,cps,brightness):
        self.cps = cps
        self.brightness = brightness
    def __call__(self,t):
        t /= self.brightness
        if t > 1:
            return None
        return bezier(self.cps,t)
    def __len__(self):
        return 1
        
class cl_bezier(vg):
    def __init__(self,cps,brightness):
        self.cps = cps
        self.dcps = [(len(cps)-1)*(cps[i+1]-cps[i]) for i in range(len(cps)-1)]
        self.nintgT=0
        self.prevT = 0
        self.brightness = brightness
    def __call__(self,t):
        t /= self.brightness
        #if t > 1:
        #    return None
        if t < self.prevT/4:
            self.nintgT=0
            self.prevT=t
        if t != self.prevT:
            self.nintgT += (t-self.prevT)/abs(bezier(self.dcps,self.nintgT))
        if self.nintgT>1:
            return None
        self.prevT = t
        return bezier(self.cps,self.nintgT)
    def __len__(self):
        return 1
        
    
class scene(vg):
    def __init__(self,*objs):
        self.dc = 0
        self.objs = objs
        self.t = 0
        self.totT = 0
        self.i = 0
        self.dt = 0.01
        self.prev = 0
        self.prevG = 0
        self.gen = (0 for i in range(0))
    def __call__(self,t):
        return rayToSquareBeamDump(0,eone**(1j*t))
        
    def __iter__(self):
        self.t=0
        self.dc=0
        self.i=0
        return self
    def __next__(self):
        self.totT += self.dt
        try:
            self.prev = next(self.gen)
        except StopIteration:
            self.t += self.dt
            route = False
            v = 0
            while self.i<len(self.objs) and self.t>self.objs[self.i].__len__():
                self.t -= self.objs[self.i].__len__()
                self.i += 1
                route = True
            if self.i >= len(self.objs):
                self.i = 0
                v = self(self.totT)
            else:
                v = self.objs[self.i](self.t)
            if v == None:
                route = (self.prevG != None)
                self.prevG = v
                v = rayToSquareBeamDump(0,magpeg(self.dc*(-1+1j*(1/(.1+abs(self.dc))))))
            else:
                route = (self.prevG == None) or route
                self.prevG = v
            if route:
                p = self.prev
                def g():
                    yield p
                    for i in traverseAlongBeamDump(randBD(p),randBD(v)):
                        yield i
                    yield v
                self.gen = g()

            self.prev = v
            
        self.dc += self.prev
        return self.prev

def sign(n):
    return (n>0)-(n<0)

def choose(a,b):
    return math.factorial(a)/math.factorial(b)/math.factorial(a-b)

def b_spline(g,ratio=1,order=0):
    prev = [next(g) for i in range(order+1)]
    i = 0
    a = 0
    def f(t,l):
        for o in range(order):
            for j in range(order-o):
                a = (1+j-t)/(order-o)
                l[j] *= a
                l[j] += l[j+1]*(1-a)
        return l[0]
    while 1:
        yield f(a,[prev[(i+j)%len(prev)] for j in range(order+1)])
        a += ratio
        if a>1:
            a-=1
            try:
                prev[i] = next(g)
            except StopIteration:
                break
            i = (i+1)%(order+1)
        
