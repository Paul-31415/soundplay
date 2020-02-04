
def makeGen(n):
    if type(n) != type((1 for i in ())):
        while 1:
            yield n
    else:
        for i in n:
            yield i


def timer(f,sps = 48000):
    p = sps/f
    a = 0
    while 1:
        a = (a+1)%p
        yield int(a) == 0

def gentimer(g):
    a = next(g)
    while 1:
        while a > 0:
            a -= 1
            yield 0
        a = next(g)
        yield 1
        
def mul(a,b):
    if type(b) == type(a):
        while 1:
            yield next(a)*next(b)
    else:
        while 1:
            yield next(a)*b
    
def run(g,rg):
    rg = makeGen(rg)
    for i in g:
        for j in range(next(rg)):
            yield i

def ramp(f,sps = 48000):
    p = sps/f
    a = 0
    while 1:
        a = (a+1)%p
        yield a/p

def framp(f,sps = 48000):
    f = makeGen(f)
    a = 0
    while 1:
        p = sps/next(f)
        a = (a+1)%p
        yield a/p

def intRamp(p):
    while 1:
        for i in range(p):
            yield i
            

def tflip(gen):
    v = 0
    while 1:
        v ^= next(gen)
        yield v

def orGen(g1, g2):
    while 1:
        yield next(g1) | next(g2)
def avg(*g):
    while 1:
        yield sum([next(i) for i in g ])/len(g)

def comparator(g,c):
    while 1:
        if type(c) == type(g):
            yield next(g)>next(c)
        else:
            yield next(g)>c
        
def exponential(s,v=0.99):
    v = makeGen(v)
    while 1:
        yield s
        s *= next(v)

def extent(g,t,sps = 48000):
    s = int(sps*t)
    for i in range(s):
        yield next(g)
        
def sequencer(vs):
    while 1:
        for v in vs:
            g = v()
            for i in g:
                yield i
def semis(n):
    return 2**(n/12)

def rrca8(v):
    return ((v>>1)&0xff)|((v<<7)&0xff)
def rand8():
    v = 0
    while 1:
        v ^= rrca8(v-37)
        yield v
def rand16():
    v = [0,0]
    while 1:
        v[1] = (rrca8((v[0]^0xff)-1)+v[1])&0xff
        v[0] = ((v[1]-v[0])^162)&0xff
        yield v[1]
def rand24():
    v = [0,0,0]
    while 1:
        v[2] = (v[2]+v[0])&0xff
        v[1] = ((rrca8((v[0]^0xff)-1)+v[1])^v[2])&0xff
        v[0] = ((v[1]-v[0])^124)&0xff
        yield v[2]

def resample(g,newR,oldR=48000):
    if type(newR) != type(g):
        t = 0
        oldR = 1/oldR
        newR = 1/newR
        v = 0
        while 1:
            t = t + oldR
            while t > newR:
                v = next(g)
                t -= newR
            yield v
    else:
        t = 0
        oldR = 1/oldR
        v = 0
        nrv = 1/next(newR)
        while 1:
            t = t + oldR
            while t > nrv:
                v = next(g)
                t -= nrv
                nrv = 1/next(newR)
            yield v

        
def strToNotes(s,noteF):
    #format: "A6,2 a4,1"... lowercase is flat, A4 is 440 hz, other numbers are additional args to noteF
    #returns a list of lambdas
    class note:
        def __init__(self,func,f,args):
            self.func = func
            self.f = f
            self.a = args
        def __call__(self):
            return self.func(self.f,*self.a)
    a = []
    for n in s.split(' '):
        f = 440*semis('CdDeEFgGaAbB'.index(n[0])+12*eval(n[1])-9-12*4)
        cargs = [eval(i) for i in n.split(',')[1:]]
        a += [note(noteF,f,cargs)]
    return a
        


    
exS =  strToNotes(
    "C5,2 A4,1 B4,1 C5,2 A4,2 E5,3 E5,2 F5,1 E5,1 C5,1 D5,3 D5,3 D5,2 F5,2 E5,2 D5,2 C5,2",
    lambda f,t,v=.5,d=.9998: extent(comparator(ramp(f),exponential(v,d)),t/8)
)
exampleSequence = sequencer(exS)

exS2 = strToNotes(
    "C5,2 A4,1 B4,1 C5,2 A4,2 E5,3 E5,2 F5,1 E5,1 C5,1 D5,3 D5,3 D5,2 C5,1 D5,1 C5,1 B4,2 C5,1 B4,1 A4,1",
    lambda f,t,v=.5,d=.9998: extent(comparator(ramp(f),exponential(v,d)),t/8)
)
exampleSequence2 = sequencer(exS2)

diphonicChordTest = sequencer(strToNotes(
    "A4,5,2 A4,8,2 B4,9,2 B4,8,2",
    lambda f,h,t : extent(comparator(ramp(f),0.5/semis(h)),t/8)
))


drumSeq1 = strToNotes(
    "A3,2,128,0.9999 A4,2,128,0.9995 A4,2,128,0.998 A4,2,128,0.9995",
    lambda f,t,v,d : extent(resample(comparator(rand24(),exponential(v,d)),1/f,1/440),t/8)
)
drumSeq2 = strToNotes(
    "A3,2,128,0.9999 A4,1,128,0.9995 A4,1,128,0.998 A4,2,128,0.9995 E4,2,255,0.9998",
    lambda f,t,v,d : extent(resample(comparator(rand24(),exponential(v,d)),1/f,1/440),t/8)
)
    
exampleSequenceB = sequencer(strToNotes(
    "A3,8,1 C4,8,1 G3,8,1 F3,8,1",
    lambda f,t,v=.5,d=.99995: extent(comparator(exponential(v,d),ramp(f)),t/8)
))
exfull=sequencer(exS+exS2)
drumTrack = sequencer(drumSeq1*3+drumSeq2)

from functools import reduce

def op(f,*g):
    while 1:
        yield reduce(f,[next(i) for i in g])


def sq(t):
    return ((t%1)>.5)*2-1










def adaptiveDithering(g=0,e=0.001,r = rand16()):
    if g == 0:
        g = avg(exfull,exampleSequenceB,drumTrack)
    v = 0
    while 1:
        p = next(g)
        d = next(r)/128
        v = v*(1-e)+p*e
        yield p>d*v
        
def polyphonicTest1(dsl=48000):
    return comparator(avg(drumTrack,exfull,exampleSequenceB),resample(mul(rand24(),1/256),dsl))
def polyphonicTestF(f):
    while 1:
        yield f(next(exfull),next(exampleSequenceB),next(drumTrack))
def polyphonicTest(d):
    return comparator(avg(drumTrack,avg(exfull,exampleSequenceB)),d)
polyphonicEx = avg(exfull,exampleSequenceB,drumTrack)

#so, for 2 chords, it looks like if you xor them, you get sound at f=a-b and f=a+b
#so, if you want f1 and f2, f1=a+b, f2=a-b; 2a = f1+f2, 2b = f1-f2
def diphonic(f1,f2):
    f2,f1 = sorted([f1,f2])
    return op(lambda a,b : a^b,comparator(ramp((f1+f2)/2),0.5),comparator(ramp((f1-f2)/2),0.5))
