#lambda music

import time

def buffer(g,bufMaxLen=48000,sampleRate=48000,prerun = 0):
    maxTimePerSample = 1/sampleRate
    overtime = 0
    buf = [0 for i in range(bufMaxLen)]
    i = 0
    gi = 0
    for ind in range(min(bufMaxLen,prerun)):
        gi = ind
        buf[gi] = next(g)
    while 1:
        s = time.monotonic()
        while time.monotonic()-s<maxTimePerSample-overtime and gi < i+bufMaxLen:
            buf[gi%bufMaxLen] = next(g)
            gi += 1
        overtime += time.monotonic()-s-maxTimePerSample
        if i < gi:
            yield buf[i%bufMaxLen]
            i += 1
        

"""
class note:
    def __init__(self,
    
class instrument:
    def __init__(self):
        pass
    def __call__(self,*params):
        return note(*params)

"""
