import aifc
import random
import math
import numpy as np

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
            


def alaw_c(v):
    import struct
    bits = struct.unpack("<I",struct.pack("<f",v))[0]
    sign = bits>>31
    exponent = (bits>>23)&0xff
    mantissa = (bits&0x7fffff) | (0x800000)
    se = exponent-127
    re = se+3
    if re > 7:
        return ((not sign)<<7)|0x7f
    if re <= 0:
        mantissa >>= 1-re
        re = 0
    return ((not sign)<<7)|(re<<4)|(0xf & (mantissa>>19))

def valaw_c(v):
    bits = np.frombuffer(v.astype('<f').tobytes(),dtype="<I")
    sign = bits>>31
    exponent = (bits>>23)&0xff
    mantissa = ((bits&0x7fffff) | (0x800000)).astype('l')
    se = exponent.astype('h')-127
    re = se+3
    r = np.zeros(len(bits),dtype='B')
    ov = re>7
    r[ov] = ((sign[ov])<<7)^0xff
    un = re<=0
    mantissa[un] >>= 1-re[un]
    re[un] = 0
    nv = ~ov
    r[nv]=(((sign[nv])<<7)|(re[nv]<<4)|(0xf & (mantissa[nv]>>19)))^0x80
    return r
    

mono = (1,lambda v:(v.real,))
def float_out(g,name="out.wav",rate=48000,show=0,phony=(2,lambda v:(v.real,v.imag))):
    if type(g) == np.ndarray:
        g = np.array(phony[1](g),dtype="<f")
    wav_out(g,name,rate,show,phony+('<'+str(phony[0])+'f',),fmt=(3,32),dsize=4,order='f')
def alaw_out(g,name="out.wav",scale=1,rate=48000,show=0,phony=(2,lambda v:(v.real,v.imag))):
    if type(g) == np.ndarray:
        g = np.array([0x55 ^ valaw_c(e) for e in phony[1](g)],dtype='B')
    else:
        phony = (phony[0],lambda v,p=phony[1],s=scale: [0x55^alaw_c(e*s) for e in p(v)])
    wav_out(g,name,rate,show,phony+('<'+str(phony[0])+'B',),fmt=(6,8),dsize=1,order='f')
def wav_out(g,name="out.wav",rate=48000,show=0,phony=(2,lambda v:(v.real,v.imag),'<ff'),fmt=(3,32),dsize=4,order='f'):
    import struct
    with open(name,"wb") as f:
        f.write(b'RIFF')
        f.write(struct.pack("<I",0))#write size here
        f.write(b'WAVE')
        f.write(b'fmt ')#16
        
        f.write(struct.pack("<IHHIIHHH",18,fmt[0],phony[0],rate,rate*dsize*phony[0],dsize*phony[0],fmt[1],0))#34
        f.write(b'fact')#38
        f.write(struct.pack("<II",4,0)) #write num samples here 
        f.write(b'data')#50
        f.write(struct.pack("<I",0)) #write num bytes here
        bpf = dsize*phony[0]
        s = 0
        def fin(s,bpf=bpf):
            r = f.tell()
            f.seek(4)
            f.write(struct.pack("<I",4+4+18+4+8+4+4+s*bpf))
            f.seek(46)
            f.write(struct.pack("<I",s))
            f.seek(54)
            f.write(struct.pack("<I",s*bpf))
            return r
        if type(g) == np.ndarray:
            b = g.tobytes(order)
            f.write(b)
            return fin(len(g))
        for v in g:
            f.write(struct.pack(phony[2],*phony[1](v)))
            if show and (s%rate == 0):
                print(s,end="\r")
                f.seek(fin(s))
            s += 1

        return fin(s)

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
