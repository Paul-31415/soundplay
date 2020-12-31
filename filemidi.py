def readMidi(bg,skipChunks = [0]):
    attrs = readHeader(bg)
    i = 0
    while 1:
        if skipChunks[i]:
            skipChunk(bg)
        else:
            for dt,c in readChunk(bg):
                yield dt,c
        i = (i+1)%len(skipChunks)

def count(i=0):
    while 1:
        yield i
        i += 1
            
import math
def monophPlay(g,sr=48000,nf = lambda n,v: ((1+1j)*v*math.sin(i*math.pi*2*(2**((n-69)/12))*440/48000) for i in count())):
    t = 0
    delt = 1
    tpq = 1
    ng = nf(0,0)
    for dt,c in g:
        cmd,ch,dat,meta = c
        #advance time
        t += dt
        while t > 0:
            yield next(ng)
            t -= delt/tpq/sr
        if not meta:
            if cmd == 0x8:
                ng = nf(0,0)
            elif cmd == 0x9:
                ng = nf(dat[0],dat[1]/127)
        else:
            if cmd == 0x51:
                delt = (1e-6)*((dat[0]<<16)+(dat[1]<<8)+dat[2])
                print("tempo:", 60/delt,dat)
            elif cmd == 0x58:
                tpq = dat[2]
                print("ticks per qn:",tpq,dat)
        
def skipChunk(bg):
    for	h in "MTrk":
        assert next(bg) == ord(h)
    l = readBE(bg,4)
    for i in range(l):
        skip = next(bg)
    return
def readChunk(bg):
    for	h in "MTrk":
        assert next(bg) == ord(h)
    l = readBE(bg,4)
    dt,ev,n = readEvent(bg)
    yield dt,ev
    while not (ev[3] and ev[0] == 0x2f):
        dt,ev,n = readEvent(bg if n == None else prependToGen(n,bg))
        yield dt,ev
        
def prependToGen(v,g):
    yield v
    for v in g:
        yield v

    
cmdNames = ["Note off",
            "Note on",
            "Key after-touch",
            "Control change",
            "Program change",
            "Channel after-touch",
            "Pitch wheel change",
            "System Message"]
cmdLens = [2,2,2,2,1,1,2,None]    

def readEvent(bg,read_dt=True,running=False):
    dt = None
    meta = False
    n = None
    if read_dt:
        dt = readVL(bg)
    if running:
        cmd = None
        ch = None
    else:
        cmd = next(bg)
        if cmd == 0xff:
            meta = True
            ch = None
            cmd = next(bg)
            l = next(bg)
            dat = [next(bg) for i in range(l)]
        else:
            ch = cmd&0xf
            cmd >>= 4
            dat,n = readDat(bg)
    return  dt,(cmd,ch,dat,meta),n
    
def readDat(bg,which=0):
    d = []
    for b in bg:
        if b>>7 == which:
            d += [b]
        else:
            return d,b

def readVL(bg):
    r = 0
    for b in bg:
        r <<= 7
        r += b&0x7f
        if b>>7 == 0:
            return r
        
    
def readHeader(bg):
    for h in "MThd":
        assert next(bg) == ord(h)
    assert readBE(bg,4) == 6
    formt = readBE(bg,2) #0 - single track, 1 - multi, synchronous, 2 - async
    tracks = readBE(bg,2)#number of tracks
    tpb = readBE(bg,2)   #ticks per beat
    return formt,tracks,tpb

def readBE(bg,byt=4):
    r=0
    for i in range(byt):
        r <<= 8
        r += next(bg)
    return r
