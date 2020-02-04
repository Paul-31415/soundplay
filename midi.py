import mido
from filters import *


class NoteGen:
    def __next__(self):
        raise StopIteration()
    def note_off(self,channel,note,velocity):
        pass
    def __init__(self,channel,note,velocity,vals,rate):
        pass
    def polytouch(self,channel,note,value):
        pass

def bytesToNum(b):
    r = 0
    for c in b:
        r = r/256 + c
    return r/256
    
def loadSample(name="voice_sans.wav"):
    import wave
    f = wave.open(name,'rb')
    w = f.getsampwidth()
    n = f.getnchannels()
    if True:
        if w == 1:
            return [bytesToNum(f.readframes(1)[:w])*2-1 for i in range(f.getnframes())]
        else:
            return [((bytesToNum(f.readframes(1)[:w])+.5)%1)*2-1 for i in range(f.getnframes())]
        
    

def genWrap(genfunc):
    class GenWrap(NoteGen):
        def __init__(self,channel,note,velocity,vals,rate):
            self.gen = genfunc(channel,note,velocity,vals,rate)
        def note_off(self,channel,note,velocity):
            self.gen = (0 for i in range(0))#.close()
        def __next__(self):
            return next(self.gen)
    return GenWrap

class NoteSet:
    class llnode:
        def __init__(self,val,next,key):
            self.v = val
            self.k = key
            self.n = next
    def __init__(self):
        self.dict = dict()
        self.root = NoteSet.llnode(None,None,None)

    def __next__(self):
        val = 0
        p = self.root
        gn = p.n
        while gn != None:
            try:
                val += next(gn.v)
            except StopIteration:
                p.n = gn.n
                try:
                    del self.dict[gn.k]
                except KeyError:
                    pass
            else:
                p = gn
            gn = gn.n
        return val
    def nonempty(self):
        return self.root.n != None

    def addNote(self,ch,n,note):
        k = (ch<<7)|n
        n = NoteSet.llnode(note,self.root.n,k)
        self.dict[k] = n
        self.root.n = n

    def getNote(self,ch,n):
        return self.dict[(ch<<7)|n].v

    

peek = [0]
def toGenerator(mid,instruments=[NoteGen for i in range(16)],rate=48000):
    #init
    turnOffWhenSubsequentOns = True
    timeError = 0
    msgs = (msg for msg in mid if not msg.is_meta)
    gens = NoteSet()
    peek[0] = gens
    vals = {"control":[[0 for i in range(128)] for j in range(16)],"pitch":[0 for i in range(16)]}
    #print(instruments)
    for msg in msgs:
        #print(msg)
        #gen samples till message
        timeError += rate*msg.time
        for i in range(int(timeError)):
            yield next(gens)
        timeError %= 1
        #do message
        if msg.type == "note_off":
            try:
                gens.getNote(msg.channel,msg.note).note_off(msg.channel,msg.note,msg.velocity)
            except KeyError:
                pass#print("tried to note_off without note_on")
        elif msg.type == "note_on":
            if (turnOffWhenSubsequentOns):
                try:
                    gens.getNote(msg.channel,msg.note).note_off(msg.channel,msg.note,msg.velocity)
                except KeyError:
                    pass#print("tried to note_off without note_on")
            if msg.channel < len(instruments) and instruments[msg.channel] != None:
                gens.addNote(msg.channel,\
                             msg.note,\
                             instruments[msg.channel]\
                             (msg.channel,msg.note,msg.velocity,vals,rate))
        elif msg.type == "polytouch":
            gens.getNote(msg.channel,msg.note).polytouch(msg.channel,msg.note,msg.value)
        elif msg.type == "control_change":
            vals["control"][msg.channel][msg.control] = msg.value
        elif msg.type == "program_change":
            pass
        elif msg.type == "aftertouch":
            pass
        elif msg.type == "pitchwheel":
            vals["pitch"][msg.channel] = msg.pitch
        elif msg.type == "sysex":
            pass
        elif msg.type == "quarter_frame":
            pass
        elif msg.type == "songpos":
            pass
        elif msg.type == "song_select":
            pass
        #print(gens.dict)

    while gens.nonempty():
        yield next(gens)
        
                
    
def pad(gen):
    for v in gen:
        yield v
    while 1:
        yield 0

import asyncio

def buffer(g,maxLen=32768):
    buf = [next(g) for i in range(maxLen)]
    bufnew = [next(g) for i in range(maxLen)]
    loop = asyncio.get_event_loop()
    async def gen(b):
        for i in range(len(b)):
            b[i] = next(g)
    while 1:
        for v in buf:
            yield v
        buf,bufnew = bufnew,buf
        loop.run_until_complete(loop.create_task(gen(bufnew)))

import threading

def threadBuffer(g,maxLen=480000):
    index = [0,1]
    buf = [0 for i in range(maxLen)]
    def gen():
        while 1:
            n = (index[1]+1)%len(buf)
            while n==index[0]:
                pass
            buf[index[1]] = next(g)
            index[1] = n
    thread = threading.Thread(target = gen, args = [])
    thread.start()
    while 1:
        n = (index[0]+1)%len(buf)
        while n==index[1]:
            pass
        yield buf[index[0]]
        index[0] = n
    thread.join()
        


from time import time_ns
        
def timeBuffer(g,maxLen=32768,maxTime = 1e9/48000/2):
    items = []
    while 1:
        s = time_ns()
        while time_ns()-s < maxTime and len(items) < maxLen:
            items.append(next(g))
        yield items.pop()

def downsample(g,ratio=.5):
    p = next(g)
    n = next(g)
    t = 0
    while 1:
        yield (1-t)*p+n*t
        t += ratio
        while t > 1:
            t -= 1
            p,n = n,next(g)

def downsample0(g,ratio=.5):
    n = next(g)
    t = 0
    while 1:
        yield n
        t += ratio
        while t > 1:
            t -= 1
            n = next(g)

def cut(g,samples):
    for i in range(int(samples)):
        yield next(g)

def affine(g,k,m):
    for s in g:
        yield s*m+k

def raffine(g,k,m):
    for s in g:
        yield (s+k)*m

def quadAttack(g,k):
    i = 0
    for s in g:
        i += k
        yield s*min(i*i,1)
    
        
def expDecay(g,k=.999,cut=0):
    a = 1
    for s in g:
        yield s*a
        a *= k
        if a < cut:
            break

import math

globalThresh = 1/(1<<16)
globalShift = 0
globalDecay = 1
globalAttack= 1
sinInst = genWrap(lambda channel,note,velocity,vals,rate: quadAttack(expDecay((math.sin(440/rate*2**((globalShift+note-69)/12)*i*2*math.pi)*velocity/128 for i in range(100000)),1-.00005*globalDecay,velocity/128*globalThresh),.01*globalAttack))
sample = [sum((math.sin(i*j*2*math.pi/1024)/j/j for j in range(1,9))) for i in range(1024)]
sample = [-0.6157655100001697, -0.7013966664582482, -0.4905448861269029, -0.1428438329055604, -0.005448668002576262, -0.14788033009468837, -0.2510975136149633, -0.2686017689213382, -0.25888166711355615, -0.30256257072803144, -0.40323989013264394, -0.36305754662854434, -0.3090994058817692, -0.29558905907289784, 0.112434565821727, 0.22164818093313177, -0.2924450043996164, -0.5306488770617424, -0.29413275802696864, 0.017585229342117092, 0.19772193994447163, 0.4027217667626407, 0.30035405122608994, -0.21196746744365463, -0.11688877709688361, 0.4781685713109605, 0.7043726106011731, 0.7680095984311976, 0.9248213206538822, 0.6779313924516378, 0.19636998654435595, -0.20298622552133]

sampleInstr = genWrap(lambda channel,note,velocity,vals,rate: quadAttack(\
                                                                        expDecay(\
                                                                                 (sample[int(440/rate*2**((globalShift+note-69)/12)*i*len(sample))%len(sample)]*velocity/128 for i in range(100000)),1-.00005*globalDecay,velocity/128*globalThresh),.01*globalAttack))

def reSampleGen(s,r):
    i = 0
    while int(i) < len(s):
        yield s[int(i)]
        i += r
        
sampleRate = 44100

sampleInst = genWrap(lambda channel,note,velocity,vals,rate: quadAttack(\
                                                                        expDecay((i*velocity/128 for i in reSampleGen(sample,2**((globalShift+note-69)/12)*sampleRate/rate))\
                                                                                 ,1-.00005*globalDecay,velocity/128*globalThresh),.01*globalAttack))

def ex(name='testMidi.mid',sr=0.5,l=48000,vol=.1,speed=1,shift=0,inst = None):
    mid = mido.MidiFile(name)
    #inst = genWrap(lambda channel,note,velocity,vals,rate: fir(vowel(440*48000/rate*2**((note-69)/12)),[velocity/128]))
    if (inst == None):
        inst = genWrap(lambda channel,note,velocity,vals,rate: expDecay(raffine(pulse(440*48000/rate*2**((shift+note-69)/12)/speed,.5),-.5,vol*velocity/128),1-.00005/sr,vol*velocity/128*globalThresh))
    if type(inst) == type([]):
        insts = [inst[i%len(inst)] for i in range(16)]
        for i in range(16):
            if insts[i] == 0:
                insts[i] = genWrap(lambda channel,note,velocity,vals,rate: expDecay(raffine(pulse(440*48000/rate*2**((shift+note-69)/12)/speed,.5),-.5,vol*velocity/128),1-.00005/sr,vol*velocity/128*globalThresh))
    else:
        insts = [inst for i in range(16)]
    #insts[1] = NoteGen
    return downsample(\
                      #threadBuffer(
                      pad(toGenerator(mid,insts,l*sr/speed))\
                      #,l)
                      ,sr)
