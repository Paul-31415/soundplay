from filters import IIR, polymult

testIPA = "ðə beɪʒ hju ɑn ðə ˈwɔtərz ʌv ðə lɑk ɪmˈprɛst ɔl, ɪnˈkludɪŋ ðə frɛnʧ kwin, bɪˈfɔr ʃi hɜrd ðæt ˈsɪmfəni əˈgɛn, ʤʌst æz jʌŋ ˈɑrθər ˈwɑntəd."

vowelTest = "aeiou"

#tongue, lips, jaw
vowels = {
    "a":(0,0,0),
    "e":(1,0,0),
    "i":(1,0,1),
    "o":(0,1,0),
    "u":(0,1,1),
}

"""consonants = {
    "g":(
    "d":
    "b":
"""

def l(g,f):
    for i in g:
        yield f(i)

def tri(s):
    while 1:
        for i in range(s):
            yield i/s

def testVoiceFunc(p,f):
    return f.setPolys(f.rootPair(p[0]*440+220,.99),polymult(f.rootPair(440,1.01+p[1]*.1),f.rootPair(440+440*p[2],1.1-.09*p[2])))



def testVoice(s,time=.25,pitch=220,volume=.2,sr=48000,func = testVoiceFunc):
    gen = l(tri(sr//pitch),lambda x: x<volume)
    filt = IIR()
    out = filt(gen)
    for c in s:
        p = vowels[c]
        func(p,filt)
        for i in range(int(sr*time)):
            yield next(out)

class testVoiceSource:
    def __init__(self):
        pass

class testVoiceFilter:
    def __init__(self):
        pass
    
class testVoiceUtterance:
    def __init__(self,phonemes):
        self.phonemes = phonemes
    def __iter__(self):
        pass



            
class Voice:
    def __init__(self,gen,filt):
        self.gen = gen
        self.filt = filt
    def __call__(self,utterance):
        pass
    


#https://www.gnu.org/software/gnuspeech/trm-write-up.pdf
class tube:
    def __init__(self,length):#,end1_termination=-0.5,end2_termination=-0.5):
        self.forward = [0]*length
        self.backwards = [0]*length
        self.i = 0
        #self.ends = (end1_termination,end2_termination)
    def __getitem__(self,i):
        l = len(self.forward)
        i = (self.i+i)%l
        f,b = self.forward[-i],self.backwards[i]
        return f+b+1j*(f-b)
    def __setitem__(self,i,v):
        l = len(self.forward)
        i = (self.i+i)%l
        f,b = (v.real+v.imag)/2,(v.real-v.imag)/2
        self.forward[-i],self.backwards[i] = f,b
    def additem(self,i,v):
        l = len(self.forward)
        i = (self.i+i)%l
        self.forward[-i] += (v.real+v.imag)/2
        self.backwards[i] += (v.real-v.imag)/2
    def step(self,af=0,ab=0):
        f,b = self.forward[-self.i],self.backwards[self.i]
        self.backwards[self.i] = af#f*self.ends[0]+af
        self.forward[-self.i] = ab#b*-self.ends[1]+ab
        self.i = (self.i+1)%len(self.forward)
        #return f*(1+self.ends[0])+1j*b*(1-self.ends[1])
    def fout(self):
        return self.forward[-self.i]
    def bout(self):
        return self.backwards[self.i]
    def __call__(self,v):
        o = self.fout()+self.bout()*1j
        self.step(v.real,v.imag)
        return o

class tubes(tube):
    def __init__(self,*tubes):
        self.tubes = tubes
        self.ks = [0 for i in range(len(tubes)+1)]
    def fout(self):
        return self.tubes[-1].fout()*(1+self.ks[-1])
    def bout(self):
        return self.tubes[0].bout()*(1-self.ks[0])
    def step(self,af=0,ab=0):
        af *= (1+self.ks[0])
        for i in range(len(self.tubes)-1):
            f = self.tubes[i].fout()
            self.tubes[i].step(af-self.ks[i]*self.tubes[i].bout(),\
                               f*self.ks[i+1]+(1-self.ks[i+1])*self.tubes[i+1].bout())
            af = f*(1+self.ks[i+1])
        f = self.tubes[-1].fout()
        self.tubes[-1].step(af-self.ks[-2]*self.tubes[-1].bout(),\
                            f*self.ks[-1]+(1-self.ks[-1])*ab)
    def __call__(self,v):
        o = self.fout()+self.bout()*1j
        self.step(v.real,v.imag)
        return o

import numpy as np
class varying_reflectance_tube:
    def __init__(self,length):
        self.ks = np.zeros(length+2,dtype=float)
        self.forward = np.zeros(length,dtype=float)
        self.backwards = np.zeros(length,dtype=float)
        self.scratch1 = np.zeros(length-1,dtype=float)
        self.scratch2 = np.zeros(length-1,dtype=float)
        self.scratch3 = np.zeros(length-1,dtype=float)
    def step(self,pi=0,po=0):
        l = len(self.forward)
        o = self.forward[-1]
        b = self.backwards[0]
        self.scratch1[:] = self.forward[:-1]
        self.scratch1 *= self.ks[1:-2]
        self.scratch3[:] = self.scratch1[:]
        self.scratch1 += self.forward[:-1]
        self.scratch2[:] = self.backwards[1:]
        self.scratch2 *= self.ks[1:-2]
        self.scratch1 -= self.scratch2
        
        self.scratch2 -= self.backwards[1:]
        self.scratch3 -= self.scratch2
        
        self.forward[1:] = self.scratch1
        self.backwards[:-1] = self.scratch3
        #self.forward[1:],self.backwards[:-1] = \
        #    self.forward[:-1]*(1+self.ks[1:-2]) - self.ks[1:-2]*self.backwards[1:],\
        #    self.backwards[1:]*(1-self.ks[1:-2]) + self.ks[1:-2]*self.forward[:-1]
        self.forward[0] = pi*(1+self.ks[0])-self.ks[0]*b
        self.backwards[-1] = po*(1-self.ks[-1])-self.ks[-1]*o
        return o*(1+self.ks[-1]),b*(1-self.ks[0])
    def __call__(self,v):
        a,b = self.step(v.real,v.imag)
        return a+1j*b



def genify(v):
    try:
        for i in v:
            yield i
    except:
        while 1:
            yield v
    
def f0(freq,duty=.22,noise=.2):
    p = 0
    f = genify(freq)
    d = genify(duty)
    n = genify(noise)
    from random import random
    for a in f:
        p = (p+a)%1
        yield 1+1j+(random()+1j*random())*next(n) if p < next(d) else 0


# http://www.rothenberg.org/Glottal/Glottalprinterfriendly.htm
