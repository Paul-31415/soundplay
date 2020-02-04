import wave
import audioread


class waveFile:
    def __init__(self,path):
        self.path = path

    def __iter__(self):
        with wave.open(open(self.path,"rb"),"rb") as wav:
            n = wav.getnchannels()
            w = wav.getsampwidth()
            r = wav.readframes(1)
            d = 1<<(8*w-1)
            if n == 1:
                while len(r):
                    yield int.from_bytes(r,"little",signed=w>1)/d - (w==0)
                    r = wav.readframes(1)
            else:
                while len(r):
                    yield (int.from_bytes(r[0:w],"little",signed=w>1)/d - (w==0))\
                        +1j*(int.from_bytes(r[w:w*2],"little",signed=w>1)/d - (w==0))
                    r = wav.readframes(1)
                
        

formats = {#"wav":waveFile,

}  


class _audioFile:
    def __init__(self,path):
        self.path = path
        with audioread.audio_open(self.path) as f:
            self.length = int(f.duration*f.samplerate)
            self.dat = [i for i in f]
            self.n = f.channels
            self.d = 1<<15
            self.rate = f.samplerate
            if self.n == 1:
                self.conv = lambda x: int.from_bytes(x,"little",signed=True)/self.d
            else:
                self.conv = lambda x: int.from_bytes(x[0:2],"little",signed=True)/self.d+1j*int.from_bytes(x[2:4],"little",signed=True)/self.d
    def __iter__(self):
        with audioread.audio_open(self.path) as f:
            n = f.channels
            d = 1<<15
            if n == 1:
                for buf in f:
                    for i in range(len(buf)>>1):
                        yield int.from_bytes(buf[i<<1:(i+1)<<1],"little",signed=True)/d
            else:
                for buf in f:
                    for i in range(len(buf)>>2):
                        yield int.from_bytes(buf[i<<2:(i<<2)+2],"little",signed=True)/d+1j*int.from_bytes(buf[(i<<2)+2:(i+1)<<2],"little",signed=True)/d
    def __len__(self):
        return self.length
    def __getitem__(self,i):
        div = (4096>>1)//self.n
        mul = 2*self.n
        return self.conv(self.dat[i//div][(i*mul)%4096:(i*mul%4096)+mul])
        

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

    
class audioFile:
    def __init__(self,path,zeroPad = True, loop=False):
        self.path = path
        self.zeroPad = zeroPad
        self.loop = loop
        self.enc = path.split(".")[-1]
        self.interp = 0
        try:
            self.file = formats[self.enc](path)
        except:
            self.file = _audioFile(path)
        self.rate = 48000
    def __iter__(self):
        def g(self,s):
            for i in s:
                yield i
            if self.zeroPad:
                while 1:
                    yield 0
            while self.loop:
                for i in s:
                    yield i
        return resamp(g(self,self.file),self.file.rate/self.rate,self.interp)

    def __getitem__(self,i):
        i *= self.file.rate/self.rate
        if self.zeroPad and (i < 0 or i >= len(self.file)):
            return 0
        else:
            if self.loop:
                i = i % len(self.file)
            if self.interp == 0:
                return self.file[int(i)]
            else:
                f = (i-int(i))
                return self.file[int(i)]*(1-f)+self.file[int(i+1) % len(self.file)]*f
    def c(self,i):
        return self[i.real].real+self[i.imag].imag*1j
