#python3.9

"""make a tone"""

help_message = """stereo
"""

import code
import pyaudio
import sys
import time
import wave
from itertools import *
from math import pi
import numpy as np

class FeynmanOsc:
    def __init__(self, w):
        self.w = w
        self.cos = 1.0 - 0.5*self.w
        self.sin = 0.0

    def __iter__(self):
        return self

    def __next__(self):
        w = self.w
        self.sin += self.cos * w
        self.cos -= self.sin * w
        return self.cos+(0+1j)*self.sin

class Osc(FeynmanOsc):
    def __init__(self, f):
        self.freq = f
        FeynmanOsc.__init__(self, f*2*pi / 48000)

    def setFreq(self, f):
        self.freq = f
        self.w = f*2*pi / 48000

    def normalize(self):
        l = (self.sin * self.sin + self.cos * self.cos) ** 0.5 
        self.sin /= l
        self.cos /= l
        
def makeGen(it):
    try:
        for i in it:
            yield i
    except TypeError:
        while True:
            yield it
    
def gain(it, g):
    for v in it:
        yield g * v

def add(*itis):
    its = [makeGen(i) for i in itis]
    while True:
        yield sum((next(it) for it in its))

def pprod(o1, o2):
    o1 = makeGen(o1)
    o2 = makeGen(o2)
    while True:
        s = next(o1)
        v = next(o2)
        yield s.real*v.real + (0+1j)*s.imag*v.imag

def prod(o1, o2):
    o1 = makeGen(o1)
    o2 = makeGen(o2)
    while True:
        yield next(o1)*next(o2)
def const(i):
    while True:
        yield i
def zeros():
    while True:
        yield 0

def yield_n_scaled(it, n, mix):
    for i in range(n):
        v = next(it) * mix.scale
        if max(abs(v.real),abs(v.imag))>mix.hardMax:
            oldS = mix.scale
            mix.scale /= max(abs(v.real),abs(v.imag))/mix.hardMax
            v /= oldS
            v *= mix.scale
        yield v


class Thing:
    pass

def yield_unpacked(g):
    for i in g:
        yield i.real
        yield i.imag

class OA:
    def __init__(self, bytes_per_sample):
        self.bytes_per_sample = bytes_per_sample
        self.sf = (1<<(bytes_per_sample*8-1)) - 1

    def get_n(self, osc, n,mix=None):
        try:
            a = np.fromiter(yield_unpacked(yield_n_scaled(osc, n, mix)), np.float32, n*2)
        except RuntimeError as er:
            #print(er)
            mix.out = zeros()
            a = np.fromiter(yield_unpacked(yield_n_scaled(zeros(), n, mix)), np.float32, n*2)
        rv = a.tobytes()
        return rv

def help():
    print(help_message)


class ringbuffer:
    def __init__(self,l):
        self.buf = np.zeros(l,dtype=complex)
        self.ri = 0
        self.wi = 0
    def write(self,v):
        try:
            wl = len(v)
        except:
            self.buf[self.wi] = v
            self.wi = (self.wi+1)%len(self.buf)
            return
        self.buf[self.wi:self.wi+wl] = v
        if (i:=(self.wi+wl - len(self.buf))) > 0:
            self.buf[0:i] = v[-i:]
        self.wi = (self.wi+wl)%len(self.buf)
    def __next__(self):
        v = self.buf[self.ri]
        self.ri = (self.ri+1)%len(self.buf)
        return v
    def __iter__(self):
        return self
    def seek(self,pos):
        self.ri = (self.wi-pos)%len(self.buf)
        return self
    
def main(argv):
    #wf = wave.open(argv[1], 'rb')
    depth = 2
    # instantiate PyAudio (1)
    p = pyaudio.PyAudio()

    mute = zeros()
    mix = Thing()
    mix.out = mute
    mix.scale = 1
    #mix.compress = 1
    mix.hardMax = 2
    mix.rate = 48000
    mix.mic = ringbuffer(1<<16)
    mix.mica = np.zeros(0)
    adapter = OA(depth)
    __oldSampleRate = mix.rate
    # define callback (2)
    def callback(in_data, frame_count, time_info, status):
        #data = wf.readframes(frame_count)
        mix.mica = np.frombuffer(in_data,dtype=np.complex64)
        mix.mic.write(mix.mica)
        try:
            try:
                mix.out = iter(mix.out)
            except TypeError:
                data = adapter.get_n(mute, frame_count, mix)
                return (data, pyaudio.paContinue)
            data = adapter.get_n(mix.out, frame_count, mix)
            #print(frame_count, time_info, status, end=': ')
            #print(' '.join('%02x%02x' % (data[2*i], data[2*i+1]) for i in range(10)))
            return (data, pyaudio.paContinue)
        except Exception as e:
            mix.error = e
            mix.out = mute
            data = adapter.get_n(mix.out, frame_count, mix)
            return (data, pyaudio.paContinue)

    # open stream using callback (3)
    stream = p.open(format=pyaudio.paFloat32,
                    channels=2,
                    rate=mix.rate,
                    output=True,
                    input=True,
                    stream_callback=callback)

    def update():
        global __oldSampleRate,stream,p
        if __oldSampleRate != mix.rate:
            __oldSampleRate = mix.rate
            stream.stop_stream()
            stream.close()
            p.terminate()
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paFloat32,
                            channels=2,
                            rate=int(mix.rate),
                            output=True,
                            input=True,
                            stream_callback=callback)
            stream.start_stream()
    def pause():
        global stream
        stream.stop_stream()
        stream.close()
    def rate(r=None):
        if r != None:
            mix.rate = r
            update()
        return mix.rate
    # start the stream (4)
    stream.start_stream()

    # wait for stream to finish (5)
    #try:
    #    while stream.is_active():
    #        time.sleep(0.1)
    #except KeyboardInterrupt:
    #    pass

    # Get a repl to facilitate live modifications
    d = globals()
    d.update(locals())
    code.interact(banner=help_message, local=d)#, exitmsg='bye')

    # stop stream (6)
    stream.stop_stream()
    stream.close()
    #wf.close()

    # close PyAudio (7)
    p.terminate()


if __name__ == '__main__':
    main(sys.argv)
