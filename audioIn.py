import wave
import audioread


                
        




class _audioFile:
    def __init__(self,path):
        self.path = path
        with audioread.audio_open(self.path) as f:
            self.length = int(f.duration*f.samplerate)
            self.dat = [i for i in f]
            self.n = f.channels
            self.d = 1<<15
            self.length = sum((len(i) for i in self.dat))//self.n//2
            self.rate = f.samplerate
            if self.n == 1:
                self.conv = lambda x: int.from_bytes(x,"little",signed=True)/self.d
            else:
                self.conv = lambda x: int.from_bytes(x[0:2],"little",signed=True)/self.d+1j*int.from_bytes(x[2:4],"little",signed=True)/self.d
    def __iter__(self,start=0):
        with audioread.audio_open(self.path) as f:
            n = f.channels
            d = 1<<15
            fct = 2 if n == 1 else 4
            def fit(f,s = start*fct):
                fi = iter(f)
                b = next(fi)
                while s >= len(b):
                    s -= len(b)
                    b = next(fi)
                yield b[s:]
                for b in fi:
                    yield b
            if n == 1:
                for buf in fit(f):
                    for i in range(len(buf)>>1):
                        yield int.from_bytes(buf[i<<1:(i+1)<<1],"little",signed=True)/d
            else:
                for buf in fit(f):
                    for i in range(len(buf)>>2):
                        yield int.from_bytes(buf[i<<2:(i<<2)+2],"little",signed=True)/d+1j*int.from_bytes(buf[(i<<2)+2:(i+1)<<2],"little",signed=True)/d
    def __len__(self):
        return self.length
    def __getitem__(self,i):
        div = (4096>>1)//self.n
        mul = 2*self.n
        return self.conv(self.dat[i//div][(i*mul)%4096:(i*mul%4096)+mul])
    def reverse(self):
        mul = 2*self.n
        for bufi in range(len(self.dat)-1,-1,-1):
            buf = self.dat[bufi]
            for i in range(len(buf)//mul-1,-1,-1):
                yield self.conv(buf[i*mul:(i+1)*mul])
    def start(self,d=0):
        if type(d) == float:
            d = int(self.length*d)
        div = (4096>>1)//self.n
        bi = d//div
        oi = d%div
        mul = 2*self.n
        for b in range(bi,len(self.dat)):
            buf = self.dat[b]
            for i in range(oi,len(buf)//mul):
                yield self.conv(buf[i*mul:(i+1)*mul])
            oi = 0
            
class _waveFile:
    def __init__(self,path):
        self.path = path
        self.dat = None
        self.rate = None
        self.d = 1
        self.w = 0
        self.n = 0
    def __iter__(self,start=0):
        with wave.open(open(self.path,"rb"),"rb") as wav:
            n = wav.getnchannels()
            w = wav.getsampwidth()
            if start:
                wav.readframes(start)
            r = wav.readframes(1)
            d = 1<<(8*w-1)
            if n == 1:
                while len(r):
                    yield int.from_bytes(r,"little",signed=w>1)/d - (w==0)
                    r = wav.readframes(1)
            elif n == 2:
                while len(r):
                    yield (int.from_bytes(r[0:w],"little",signed=w>1)/d - (w==0))\
                        +1j*(int.from_bytes(r[w:w*2],"little",signed=w>1)/d - (w==0))
                    r = wav.readframes(1)
            else:
                while len(r):
                    yield [(int.from_bytes(r[w*i:w*(i+1)],"little",signed=w>1)/d - (w==0)) for i in range(n)]
                    r = wav.readframes(1)

    def load(self):
        with wave.open(open(self.path,"rb"),"rb") as wav:
            self.rate = wav.getframerate()
            l = wav.getnframes()
            n = wav.getnchannels()
            w = wav.getsampwidth()
            self.w = w
            self.n = n
            self.d = 1<<(8*w-1)
            self.o = -(w==0)
            self.dat = wav.readframes(l)
    def __len__(self):
        if self.dat == None:
            self.load()
        return len(self.dat)
    def __getitem__(self,i):
        if self.dat == None:
            self.load()
        do = i*self.w*self.n
        if self.n == 1:
            return int.from_bytes(self.dat[do:do+self.w],"little",signed=self.w>1)/self.d-(self.w==0)
        elif self.n == 2:
            return int.from_bytes(self.dat[do:do+self.w],"little",signed=self.w>1)/self.d-(self.w==0) \
                +1j*(int.from_bytes(self.dat[do+self.w:do+self.w*2],"little",signed=self.w>1)/self.d-(self.w==0))
        else:
            return [int.from_bytes(self.dat[do+self.w*i:do+self.w*(i+1)],"little",signed=self.w>1)/self.d-(self.w==0) for i in range(self.n)]
        
def resamp(g,r,intrp=0):
    #    if intrp == 0:
    t = 0
    v = 0
    while 1:
        t += r
        while t>0:
            v = next(g)
            t-= 1
        yield v
"""elif intrp == 1:
        t = 0
        v = 0
        p = 0
        while 1:
            t += r
            
            while t>0:
                v = """

formats = {"wav":_waveFile,

}  
    
class audioFile:
    def __init__(self,path,bpm=None,startOffset=0,zeroPad = False, loop=False,tsr = None):
        self.bpm = bpm
        self.start = startOffset
        self.path = path
        self.zeroPad = zeroPad
        self.loop = loop
        self.enc = path.split(".")[-1]
        self.interp = 0
        self.rate = tsr
        self.loaded = False
    def load(self):
        try:
            self.file = formats[self.enc](self.path)
            self.file.load()
        except:
            self.file = _audioFile(self.path)
        if self.rate == None:
            self.rate = self.file.rate
        self.loaded = True
    def loadAsAF(self):
        self.file = _audioFile(self.path)
        if self.rate == None:
            self.rate = self.file.rate
        self.loaded = True
    def play(self,d=0):
        if not self.loaded:
            self.load()
        for i in self.file.start(d):
            yield i
    def __iter__(self):
        if not self.loaded:
            self.load()
        def g(self,s):
            for i in s.__iter__(self.start):
                yield i
            if self.zeroPad:
                while 1:
                    yield 0
            while self.loop:
                for i in s:
                    yield i
        if self.file.rate == self.rate:
            return g(self,self.file)
        return resamp(g(self,self.file),self.file.rate/self.rate,self.interp)

    def __getitem__(self,i):
        if not self.loaded:
            self.load()
        i *= self.file.rate/self.rate
        i += self.start
        if self.zeroPad and (int(i) < 0 or int(i) >= len(self.file)):
            return 0
        else:
            if self.loop:
                i = i % len(self.file)
            if self.interp == 0:
                return self.file[int(i)]
            else:
                f = (i-int(i))
                return self.file[int(i)]*(1-f)+self.file[int(i+1) % len(self.file)]*f
    def __len__(self):
        if not self.loaded:
            self.load()
        return (len(self.file)*self.file.rate)//self.rate
    def c(self,i):
        return self[i.real].real+self[i.imag].imag*1j
    def spb(self):
        return 60*self.rate/self.bpm
    def reverse(self):
        if not self.loaded:
            self.load()
        for v in self.file.reverse():
            yield v
        
