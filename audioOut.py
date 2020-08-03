import aifc
import random
import math

def frame(v,b=2,c=2):
    r = []
    r2 = []
    for i in range(b):
        r += [int(v.real%256)]
        if c == 2:
            r2 += [int(v.imag%256)]
        v /= 256
    if c == 2:
        r = r2+r
    return bytes(reversed(r))
            
    

def out(g,name="out.aiff",extent = -1,channels=2,rate=48000,gain = .5,prec=16,ditherlsbs=1):
    with aifc.open(name,'w') as f:
        b = int(math.ceil(prec/8))
        
        f.setnchannels(channels)
        f.setsampwidth(b)
        f.setframerate(rate)
        factor = (1<<(prec-1))*gain

        bf = 1<<(8*b-1)
        for i in g:
            f.writeframes(frame(i*factor + (prec<=8)*bf+ditherlsbs*(random.random()-.5),b,channels))
            if extent > 0:
                extent -= 1
                if extent == 0:
                    break
        f.close()
    return

import wave
def fout(d,name = "out.wav"):
    with wave.open(name,"wb") as f:
        f.setparams(2,4,48000,0)

        for e in d:
            f.writeframes((int(e.real*(1<<31))%(1<<32)).to_bytes(4,"little")+(int(e.imag*(1<<31))%(1<<32)).to_bytes(4,"little"))
