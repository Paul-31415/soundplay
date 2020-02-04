import math
class harmOsc:
    def __init__(self,freq=440,complexAmplitude=0+0j,damp=0.001,sampleRate=48000):
        #damp is per oscillation
        rads=math.pi*2*freq/sampleRate
        self.dampf = 1-((1-damp)**(freq/sampleRate))
        self.f=(math.cos(rads)+math.sin(rads)*1j)*(1-self.dampf)
        self.saveF = self.f
        self.a=complexAmplitude
        self.sampleRatio = freq/sampleRate
        self.sampleRate=sampleRate
    def __next__(self):
        self.a *= self.f
        return self.a*self.dampf
    def add(self,e):
        self.a+=e
    def addVel(self,v):
        #vel = -im(a)*im(f)*sr
        #to add to vel:
        # a += 1j*v/im(f)/sr
        self.a += 1j*v/self.f.imag/self.sampleRate
    def damp(self,amt,posfac=1):
        #imag is vel qty
        self.a -= 1j*(self.a.imag*amt*abs(posfac))
    def setDamp(self,amt):
        self.f*=(1-amt)**self.sampleRatio
    def clearDamp(self):
        self.f = self.saveF
    def gen(self):
        while 1:
            yield next(self)
    def deriv(self):
        return self.a*(self.f-1)*self.sampleRate

class string:
    def __init__(self,freq=440,damp=0.99,numHarmonics=0,inharmicity=0,sampleRate=48000):
        #https://www.acs.psu.edu/drussell/Demos/Stiffness-Inharmonicity/Stiffness-B.html
        if numHarmonics == 0:
            numHarmonics = int(20000/freq)
        effrq = freq-inharmicity
        freqs = [effrq*(n+1)+inharmicity*(n+1)**2 for n in range(numHarmonics)]
        self.harmonics = [harmOsc(f,0,damp,sampleRate) for f in freqs]
        self.dampWeights = [0 for i in self.harmonics]
    def __next__(self):
        
        r = [next(o) for o in self.harmonics]
        p = sum([self.harmonics[i].deriv()*self.dampWeights[i] for i in range(len(r))])
        for i in range(len(r)):
            self.harmonics[i].addVel(-p*self.dampWeights[i])
        return r
    def add(self,e):
        for o in self.harmonics:
            o.add(e)
    def addPt(self,e,pos=.5**.5):
        for i in range(len(self.harmonics)):
            self.harmonics[i].add(math.sin((i+1)*pos*math.pi)*e)
    def setDampPt(self,amt,pos=.5**.5):
        for i in range(len(self.harmonics)):
            self.dampWeights[i] = amt*math.sin(pos*(i+1)*math.pi)
    def dampPt(self,amt,pos=.5**.5):
        for i in range(len(self.harmonics)):
            self.harmonics[i].damp(amt,math.sin(pos*(i+1)*math.pi))
    def clearDamp(self):
        for i in range(len(self.harmonics)):
            self.harmonics[i].clearDamp()
    def setDamp(self,amt,pos=.5**.5,dl=0.04):
        def f(x):
            return -math.cos((x%1)*math.pi)+2*int(x)
        for i in range(len(self.harmonics)):
            self.harmonics[i].setDamp(abs(amt*(f((pos+dl)*(i+1))-f((pos-dl)*(i+1)))))
    def damp(self,amt,pos=.5**.5,dl=0.04):
        #integral of abs(sin) is -cos(x%π)+2*int(x/π)
        def f(x):
            return -math.cos((x%1)*math.pi)+2*int(x)
        for i in range(len(self.harmonics)):
            self.harmonics[i].damp(amt,f((pos+dl)*(i+1))-f((pos-dl)*(i+1)))
    def gen(self):
        while 1:
            yield next(self)
    def addEdges(self,e1,e2):
        together = (e1+e2)/2
        apart = (e1-e2)/2
        #fft of apart is 2/(kπ) on all
        #fft of together is 4/(kπ) on odds
        for i in range(len(self.harmonics)):
            self.harmonics[i].add((apart*2+together*4*((i+1)%2))/(i+1)/math.pi)

class soundboard:
    def __init__(self,inertia,moment,strings):
        self.I = moment
        self.m = inertia
        self.strings = strings
    def __next__(self):
        for i in self.strings:
            return 0
    








        

class stringGroup:
    def __init__(self,strings,coupling=0.1):
        self.strings = strings
        self.coupling=coupling
    def __next__(self):
        r = sum([next(s) for s in self.strings])
        self.add((r*self.coupling).real)
        return r
    def add(self,e):
        for s in self.strings:
            s.add(e)
    def addPt(self,e,pos=.5**.5):
        for s in self.strings:
            s.addPt(e,pos)
    def dampPt(self,amt,pos=.5**.5):
        for s in self.strings:
            s.dampPt(e,pos)
    def damp(self,amt,pos=.5**.5,dl=0.04):
        for s in self.strings:
            s.damp(e,pos,dl)
    def gen(self):
        while 1:
            yield next(self)
