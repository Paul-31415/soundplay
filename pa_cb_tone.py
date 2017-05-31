"""make a tone"""

import pyaudio
import wave
import time
import sys
from math import pi
import numpy as np

class Osc:
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

def gain(it, g):
    for v in it:
        yield g * next(it)

def add(o1, o2):
    while True:
        yield next(o1) + next(o2)

def prod(o1, o2):
    while True:
        yield next(o1) * next(o2)

def yield_n_scaled(it, n, sf=1.0):
    for i in range(n):
        yield next(it) * sf


class OA:
    def __init__(self, bytes_per_sample):
        self.bytes_per_sample = bytes_per_sample
        self.sf = (1<<(bytes_per_sample*8-1)) - 1

    def get_n(self, osc, n):
        # FIXME: np.int16
        a = np.fromiter(yield_n_scaled(osc, n, self.sf), np.int16, n)
        rv = a.tobytes()
        return rv



def main(argv):
    if len(argv) < 2:
        print("Plays a tone.\n\nUsage: %s freq" % argv[0])
        sys.exit(-1)

    #wf = wave.open(argv[1], 'rb')

    # instantiate PyAudio (1)
    p = pyaudio.PyAudio()

    o1 = Osc(float(argv[1]) * 2*pi / 48000)
    o2 = Osc(float(argv[2]) * 2*pi / 48000)
    o3 = Osc(float(argv[3]) * 2*pi / 48000)

    mix = prod(gain(add(o1, o2), 0.5), o3)

    adapter = OA(2)

    # define callback (2)
    def callback(in_data, frame_count, time_info, status):
        #data = wf.readframes(frame_count)
        data = adapter.get_n(mix, frame_count)
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
    try:
        while stream.is_active():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    # stop stream (6)
    stream.stop_stream()
    stream.close()
    #wf.close()

    # close PyAudio (7)
    p.terminate()


if __name__ == '__main__':
    main(sys.argv)
