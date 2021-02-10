import argparse

parser = argparse.ArgumentParser(description='combine some songs.')
parser.add_argument('--source', required=1,type=str,help="input filename to warp into another song")
parser.add_argument('--target', required=1,type=str,help="input filename template to warp to")
parser.add_argument('-o', required=1,type=str,help="output filename")
parser.add_argument('--q1', default=.95,type=float,help="quantile for source peaks")
parser.add_argument('--q2', default=.95,type=float,help="quantile for target peaks")
parser.add_argument('--s1',default=1<<12,type=int,help="fourier window size of source (should be power of 2)")
parser.add_argument('--st',default=1<<12,type=int,help="fourier window size of target (should be power of 2)")
parser.add_argument('--s2',default=1<<12,type=int,help="fourier window size of output (should be power of 2)")
parser.add_argument('-a',default=1,type=float,help="alpha, how much to apply the transformation")
parser.add_argument('--size',default=-1,type=int,help="compressed mode size cap")


args = parser.parse_args()

import pitch
import audioIn as aud
import audioOut
import numpy as np
import scipy.signal

src = aud.audioFile(args.source)
to = aud.audioFile(args.target)

print("loading source file",end="\r")
src[0]
g = src.play(0)
print("loading target file",end="\r")
to[0]
gt = to.play(0)

print ("progress (in samples):")

t = pitch.mt_pq(gt,args.q1,args.q2,args.a,args.s1,args.s2,args.st)
gen = pitch.flat((t(v) for v in g))
if args.size != -1:
    print("(working...)",end="\r")
    r = np.fromiter(gen,dtype=complex)
    sr = 48000
    if len(r)*2 > args.size:
        sr = ((args.size//2)*48000)//len(r)
        r = scipy.signal.resample(r,args.size//2)
        print("resampling down to",sr)
    audioOut.alaw_out(r*16,args.o,rate=sr)
else:
    audioOut.float_out(gen,args.o,show=1)
print("done!                 ")
    
