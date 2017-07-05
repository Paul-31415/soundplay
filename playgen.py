"""make a tone"""

help_message = """Audio hackery: a generator you compose feeds the audio output.
Assign your generator to mix.out, e.g.:

mix.out = cycle(v/100 for v in range(-100,100)) # Play a sawtooth
mix.out = Osc(440) # Play A below middle C at full volume
mix.out = gain(Osc(660), 0.1) # play G 20db down
o1, o2, o3, o4 = Osc(440), Osc(550), Osc(660), Osc(0.3)
mix.out = prod(gain(add(o1, o2, o3), 1/3), o4)
mix.out = mute     # Play 0's continuously
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
        return self.sin

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
        

def gain(it, g):
    for v in it:
        yield g * next(it)

def add(*its):
    while True:
        yield sum(next(it) for it in its)

def prod(o1, o2):
    while True:
        yield next(o1) * next(o2)

def zeros():
    while True:
        yield 0

def yield_n_scaled(it, n, sf=1.0):
    for i in range(n):
        yield next(it) * sf

class Thing:
    pass

class OA:
    def __init__(self, bytes_per_sample):
        self.bytes_per_sample = bytes_per_sample
        self.sf = (1<<(bytes_per_sample*8-1)) - 1

    def get_n(self, osc, n):
        # FIXME: np.int16
        a = np.fromiter(yield_n_scaled(osc, n, self.sf), np.int16, n)
        rv = a.tobytes()
        return rv

def help():
    print(help_message)

def main(argv):
    #wf = wave.open(argv[1], 'rb')

    # instantiate PyAudio (1)
    p = pyaudio.PyAudio()

    mute = zeros()
    mix = Thing()
    mix.out = mute

    adapter = OA(2)

    # define callback (2)
    def callback(in_data, frame_count, time_info, status):
        #data = wf.readframes(frame_count)
        data = adapter.get_n(mix.out, frame_count)
        #print(frame_count, time_info, status, end=': ')
        #print(' '.join('%02x%02x' % (data[2*i], data[2*i+1]) for i in range(10)))
        return (data, pyaudio.paContinue)

    # open stream using callback (3)
    stream = p.open(format=p.get_format_from_width(2),
                    channels=1,
                    rate=48000,
                    output=True,
                    stream_callback=callback)

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
