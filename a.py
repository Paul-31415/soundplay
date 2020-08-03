#quick import stuff

import math
from squareThings import *
from sampleGen import *
from importlib import reload
def saveM(gen,t,name = "save.aiff"):
    saveSample(monoChannelWrapper(extent(gen,t)),name,1,1)

#print("run(cycle([0,1]),fof(rand24(),lambda x: int(math.log(256/(x+1),2))))")
def cycle(l):
    while 1:
        for i in l:
            yield i
def runNoise(b=2,e=1):
    b = makeGen(b)
    e = makeGen(e)
    return run(cycle([0,1]),fof(rand24(),lambda x: int(1+math.log(256/(x+1),next(b))**next(e))))


def c(g):
    for i in g:
        yield i
    while 1:
        yield 0

import audioIn as aud
from filters import *
from distort import *
flt = IIR()
import fmsynth as fm
import signals as sig
from itools import *
f1 = aud.audioFile("/Users/paul/Music/other/chipzel\ -\ Spectra\ -\ 10\ Sonnet-FR7MMf4A5Ic.mkv".replace("\\",""))
f2 = aud.audioFile("/Users/paul/Music/other/Virtual\ Riot\ -\ Idols\ \(EDM\ Mashup\)\ \(Official\ Video\)-vZyenjZseXA.mkv".replace("\\",""))
f3 = aud.audioFile("/Users/paul/Music/other/_-5IOVkstxkdE.mkv".replace("\\",""))
f4 = aud.audioFile("/Users/paul/Music/other/Bossfight\ -\ Milky\ Ways-_5w8SJ3yVsc.webm".replace("\\",""))
f5 = aud.audioFile("/Users/paul/Music/other/Mr.\ Blue\ Sky-wuJIqmha2Hk.webm".replace("\\",""))
f6 = aud.audioFile("/Users/paul/Music/other/Initial\ D\ -\ Night\ Of\ Fire-SRbhLtjOiRc.mp4".replace("\\",""))
g1 = None
g2 = None
g3 = None
g4 = None
g5 = None
def rg():
    global g1,g2,g3,g4,g5
    g1 = it((i for i in f1))
    g2 = it((i for i in f2))
    g3 = it((i for i in f3))
    g4 = it((i for i in f4))
    g5 = it((i for i in f5))
    g6 = it((i for i in f6))
#rg()
l1 = lm(lambda i: f1[i*48000])
l2 = lm(lambda i: f2[i*48000])
l3 = lm(lambda i: f3[i*48000])
l4 = lm(lambda i: f4[i*48000])
l5 = lm(lambda i: f5[i*48000])
l6 = lm(lambda i: f6[i*48000])
from brailleG import *

def bpm(b,sr = 48000):
    return (60/b)*sr
 

primitives = aud.audioFile("/Users/paul/Music/other/osciloscope music/Primitives.wav")


import numpy as np
def fconj(tst):
    tst_r = [x.real for x in tst]
    tst_i = [x.imag for x in tst]
    f_tst_r = np.fft.fft(tst_r)
    f_tst_i = np.fft.fft(tst_i)
    for i in range(len(tst)//2):
        f_tst_r[-i-1] = 0
        f_tst_i[-i-1] = 0

    r_r = np.fft.ifft(f_tst_r)
    r_i = np.fft.ifft(f_tst_i)
    res = [2*(r_r[i].imag*1j+r_i[i].imag)for i in range(len(tst))]
    return res


def csign(v):
    return (v.real>0)-(v.real<0)+1j*((v.imag>0)-(v.imag<0))

import time
def preload(g,pl=48000,bl=48000,tps=.7/48000,bu=4800):
    buf = [next(g) for i in range(pl)]+[0]*(bl-pl)
    i = pl
    o = bl-1
    e = tps
    while 1:
        st = time.monotonic()
        while time.monotonic() < st+e and i != o:
            buf[i] = next(g)
            i = (i+1)%bl
        o = (o+1)%bl
        if i == o:
            o = (o-bu)%bl
        e += tps-(time.monotonic()-st)
        yield buf[o]
            
import filters as filt
import distort as di

fld = filt.feedbackl(filt.chainl(filt.delayl(22154-2400),di.tritapsl(.5,2*2400)),o(lambda x:di.roundy(x*.49,.1)),o(lambda x:math.tanh(x*10)*.7))

fldq = filt.feedbackl(filt.chainl(filt.delayl(22154//4-2000),di.tritapsl(.5,2*2000)),o(lambda x:di.roundish(x*.75,.05,2)),o(lambda x:math.tanh(x*10)*.7))
fldh = filt.feedbackl(filt.chainl(filt.delayl(22154//2-2000),di.tritapsl(.5,2*2000)),o(lambda x:di.roundish(x*.75,.05,2)),o(lambda x:math.tanh(x*10)*.7))
fld2 = filt.feedbackl(filt.chainl(filt.delayl(2*22154-2000),di.tritapsl(.5,2*2000)),o(lambda x:di.roundish(x*.75,.05,2)),o(lambda x:math.tanh(x*10)*.7))
fld4 = filt.feedbackl(filt.chainl(filt.delayl(22154*4-2000),di.tritapsl(.5,2*2000)),o(lambda x:di.roundish(x*.75,.05,2)),o(lambda x:math.tanh(x*10)*.7))

fla = filt.feedbackl(filt.chainl(filt.delayl(22154)),di.maxholdfuncl(lambda v,m: v/(m+1),.995),o(lambda x:x))
flao = filt.feedbackl(filt.chainl(filt.delayl(22154-2000),filt.parrl(filt.delayl(2000),di.tritapsl(.5,2*2000))),di.maxholdfuncl(lambda v,m: v/(m+1),.995),o(lambda x:x))




dinf =  aud.audioFile("/Users/paul/Music/other/chiptune/Deathro\ -\ I\ Never\ Forget.wav".replace("\\",""),165)
v = aud.audioFile("/Users/paul/Music/other/virtual\ riot/Virtual\ Riot\ -\ Transmission-AJlIYAsSTws.mkv".replace("\\",""),130)
bh = aud.audioFile("/Users/paul/Music/other/Basshunter\ -\ DotA\ \(HQ\)-8d44ykdKvCw.webm".replace("\\",""))
h = aud.audioFile("/Users/paul/Music/other/muse/Muse\ -\ Knights\ Of\ Cydonia\ \ \(Video\)-G_sBOsh-vyI.mkv".replace("\\",""))
cd = aud.audioFile("/Users/paul/Music/other/すてらべえ\ BREAKCORE\ \(Stella\ Hiragana\)\ -\ Caramelldansen\ Remix\ _\ ｳｯｰｳｯｰｳﾏｳﾏ\(ﾟ∀ﾟ\)-87fh7xHZVvc.webm".replace("\\",""))
fd = aud.audioFile("/Users/paul/Music/other/XI\ -\ Freedom\ Dive\ ↓-OI3C9qQlb1U.webm".replace("\\",""))
cdbr = aud.audioFile("/Users/paul/Music/other/Weird\ russian\ singer\ -\ Chum\ Drum\ Bedrum-tVj0ZTS4WF4.mkv".replace("\\",""))
um = aud.audioFile("/Users/paul/Music/other/TheFatRat\ -\ Unity\ vs\ Megalovania\ \(by\ LiterallyNoOne\)-um_yP-2ix9k.webm".replace("\\",""))
rwby = aud.audioFile("/Users/paul/Music/other/RWBY\ Volume\ 1\ Soundtrack\ -\ 8.\ I\ May\ Fall-GGoiOfHsuUA.webm".replace("\\",""))
nbp = aud.audioFile("/Users/paul/Music/other/NOMA\ -\ Brain\ Power\ -\ LYRICS\!-h-mUGj41hWA.mkv".replace("\\",""))
sf = aud.audioFile("/Users/paul/Music/other/Getsix\ -\ Sky\ Fracture-_PDCcQ3v1MY.mkv".replace("\\",""))
sf2=aud.audioFile("/Users/paul/Music/other/getsix/Getsix\ -\ Sky\ Fracture\ VIP\ \(ft.\ Miss\ Lina\)-4zeZgdQX1SE.mkv".replace("\\",""))
f74 = aud.audioFile("/Users/paul/Music/other/f-777/F\ 777\ -\ 4.\ I\ Love\ You\ All\ \(Ludicrous\ Speed\ Album\)-EorbESm9LSc.mkv".replace("\\",""))
f7lia = aud.audioFile("/Users/paul/Music/other/f-777/jesseValentineMusic/F-777\ -\ Life\ Is\ Awesome-5ExTKw3y7q0.webm".replace("\\",""))
f7lisa = aud.audioFile("/Users/paul/Music/other/f-777/jesseValentineMusic/F-777\ -\ Life\ Is\ Still\ Awesome\ \(COSMIC\ BLASTER\)-K6AAql-rPqI.webm".replace("\\",""))
    
