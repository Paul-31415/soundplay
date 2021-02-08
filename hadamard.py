

import numpy as np

ROOT2 = 2**.5

"""def fht_p2_ip(inp,l=0,h=None,chain=0): #input must be power of 2 length, computes in place
    if h==None:
        h = len(inp)
    if h <= l+1:
        return chain
    m = (l+h)>>1
    inp[l:h] += np.concatenate((inp[m:h],-inp[l:m]))
    inp[m:h] *= -1
    fht_p2_ip(inp,l,m,chain+1)
    c = fht_p2_ip(inp,m,h,chain+1)
    if chain == 0:
        inp[l:h] /= 2**(c/2)
        return inp
    return c
"""

def fht_p2_ip(inp,scale=True):
    #adds along the rows
    l = len(inp)
    s = 0
    inp = np.reshape(inp,(1,l))
    tmp = inp[:,:l>>1]+0
    while l > 1:
        l >>= 1
        r = np.reshape(inp,(2<<s,l))
        tmp = np.reshape(tmp,(1<<s,l))
        s += 1
        tmp[:] = inp[:,:l]
        r[::2,:] += inp[:,l:]
        r[1::2,:] -= tmp
        r[1::2,:] *= -1
        inp = r
    if scale:
        r /= 2**(s/2)
    return np.reshape(r,(1<<s,))

class slider:
    def __init__(self,tf=lambda x: x,val=0.5,m=0,M=1,show=True):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
        self.plt = plt
        plt.figure()
        axm = plt.axes([0.1, 0.2, 0.8, 0.6])
        self.tf = tf
        self.v = tf(val)
        sm = Slider(axm, 'val', m, M,valinit=self.v)
        def update(val,self=self,sm=sm):
            self.v = self.tf(val)
            sm.label.set_text('val %.5f'%self.v)
        self.nogc = update
        sm.on_changed(update)
        if show:
            self.show()
    def __next__(self):
        return self.v
    def __iter__(self):
        return self
    def show(self,block=0):
        self.plt.show(block=block)


def genify(g):
    try:
        for v in g:
            yield v
    except:
        while 1:
            yield g

def mag2(b,*ignored):
    return b.real**2+b.imag**2
bitlen = np.vectorize(lambda x: int(x).bit_length())
def bitrev(l):
    return np.vectorize(lambda x: eval('0b'+bin(x|(1<<l))[:2:-1]))
def gray(b):
    return b ^ (b>>1)
def hadamard_peaks(gen,bsize=64,quantile=63/64,order=mag2):
    quantile = genify(quantile)
    freqs = gray(bitrev(bsize.bit_length())(np.arange(bsize)))
    try:
        while 1:
            buf = np.fromiter(gen,complex,bsize)
            fht_p2_ip(buf)
            m2 = order(buf,freqs)
            buf[m2<np.quantile(m2,next(quantile))] = 0
            fht_p2_ip(buf)
            for i in range(len(buf)):
                yield buf[i]
    except StopIteration:
        pass

def fourier_peaks(gen,bsize=64,quantile=63/64,order=mag2):
    quantile = genify(quantile)
    freqs = .5-np.abs(np.arange(bsize)/bsize-.5)
    import scipy
    try:
        while 1:
            buf = np.fromiter(gen,complex,bsize)
            buf[:] = scipy.fft.fft(buf)
            m2 = order(buf,freqs)
            buf[m2<np.quantile(m2,next(quantile))] = 0
            buf[:] = scipy.fft.ifft(buf)
            for i in range(len(buf)):
                yield buf[i]
    except StopIteration:
        pass

def wavelet_peaks(gen,bsize=64,quantile=63/64,wavelet='haar',order=mag2):
    import pywt
    quantile = genify(quantile)
    freqs = [len(v) for v in pywt.wavedec(np.zeros(bsize),wavelet)]
    freqs = np.concatenate([np.ones(v)*v for v in freqs])
    try:
        while 1:
            buf = np.fromiter(gen,complex,bsize)
            wl = pywt.wavedec(buf,wavelet)
            t = np.concatenate(wl)
            m2 = order(t,freqs)
            t[m2<np.quantile(m2,next(quantile))] = 0
            o = 0
            for i in range(len(wl)):
                wl[i][:] = t[o:o+len(wl[i])]
                o += len(wl[i])
            buf[:] = pywt.waverec(wl,wavelet)
            for i in range(len(buf)):
                yield buf[i]
    except StopIteration:
        pass
    
def hadamard_cf_peaks(gen,bsize=64,quantile=63/64,order=mag2):
    quantile = genify(quantile)
    freqs = gray(bitrev(bsize.bit_length())(np.arange(bsize)))
    try:
        buf = np.zeros(bsize,dtype=complex)
        outbuf = np.zeros(bsize,dtype=complex)
        crossfade = (1-np.cos(np.arange(bsize)/bsize*np.pi*2))/2
        faded = np.zeros(bsize,dtype=complex)
        while 1:
            buf[:bsize//2] = buf[bsize//2:]
            buf[bsize//2:] = np.fromiter(gen,complex,bsize//2)
            faded[:] = buf
            fht_p2_ip(faded)
            m2 = order(faded,freqs)
            faded[m2<np.quantile(m2,next(quantile))] = 0
            fht_p2_ip(faded)
            faded *= crossfade
            outbuf[:bsize//2] = outbuf[bsize//2:]
            outbuf[bsize//2:] = 0
            outbuf += faded
            for i in range(bsize//2):
                yield outbuf[i]
    except StopIteration:
        pass

def fourier_cf_peaks(gen,bsize=64,quantile=63/64,order=mag2):
    quantile = genify(quantile)
    freqs = .5-np.abs(np.arange(bsize)/bsize-.5)
    import scipy
    try:
        buf = np.zeros(bsize,dtype=complex)
        outbuf = np.zeros(bsize,dtype=complex)
        crossfade = (1-np.cos(np.arange(bsize)*np.pi*2/bsize))/2
        faded = np.zeros(bsize,dtype=complex)
        while 1:
            buf[:bsize//2] = buf[bsize//2:]
            buf[bsize//2:] = np.fromiter(gen,complex,bsize//2)
            faded[:] = scipy.fft.fft(buf)
            m2 = order(faded,freqs)
            faded[m2<np.quantile(m2,next(quantile))] = 0
            faded[:] = scipy.fft.ifft(faded)*crossfade
            outbuf[:bsize//2] = outbuf[bsize//2:]
            outbuf[bsize//2:] = 0
            outbuf += faded
            for i in range(bsize//2):
                yield outbuf[i]

    except StopIteration:
        pass
                          

def wavelet_cf_peaks(gen,bsize=64,quantile=63/64,wavelet='haar',order=mag2):
    import pywt
    quantile = genify(quantile)
    freqs = [len(v) for v in pywt.wavedec(np.zeros(bsize),wavelet)]
    freqs = np.concatenate([np.ones(v)*v for v in freqs])
    try:
        buf = np.zeros(bsize,dtype=complex)
        outbuf = np.zeros(bsize,dtype=complex)
        crossfade = (1-np.cos(np.arange(bsize)*np.pi*2/bsize))/2
        faded = np.zeros(bsize,dtype=complex)
        while 1:
            buf[:bsize//2] = buf[bsize//2:]
            buf[bsize//2:] = np.fromiter(gen,complex,bsize//2)
            wl = pywt.wavedec(buf,wavelet)
            t = np.concatenate(wl)
            m2 = order(t,freqs)
            t[m2<np.quantile(m2,next(quantile))] = 0
            o = 0
            for i in range(len(wl)):
                wl[i][:] = t[o:o+len(wl[i])]
                o += len(wl[i])
            faded[:] = pywt.waverec(wl,wavelet)*crossfade
            outbuf[:bsize//2] = outbuf[bsize//2:]
            outbuf[bsize//2:] = 0
            outbuf += faded
            for i in range(bsize//2):
                yield outbuf[i]

        while 1:
            buf = np.fromiter(gen,complex,bsize)
            for i in range(len(buf)):
                yield buf[i]
    except StopIteration:
        pass
        

        

    

def hadamard_peaks_ac(gen,bsize=64,quantile=63/64,order=mag2):
    quantile = genify(quantile)
    freqs = gray(bitrev(bsize.bit_length())(np.arange(bsize)))
    try:
        outbuf = np.zeros(bsize,dtype=complex)
        a = bsize//4
        while 1:
            buf = np.fromiter(gen,complex,bsize)
            for i in range(a):
                yield outbuf[i]
            fht_p2_ip(buf)
            for i in range(a,a*2):
                yield outbuf[i]
            m2 = order(buf,freqs)
            for i in range(a*2,a*3):
                yield outbuf[i]
            buf[m2<np.quantile(m2,next(quantile))] = 0
            fht_p2_ip(buf)
            for i in range(a*3,len(outbuf)):
                yield outbuf[i]
            outbuf[:] = buf
    except StopIteration:
        pass

def fourier_peaks_ac(gen,bsize=64,quantile=63/64,order=mag2):
    quantile = genify(quantile)
    freqs = .5-np.abs(np.arange(bsize)/bsize-.5)
    import scipy
    try:
        outbuf = np.zeros(bsize,dtype=complex)
        a = bsize//4
        while 1:
            buf = np.fromiter(gen,complex,bsize)
            for i in range(a):
                yield outbuf[i]
            buf[:] = scipy.fft.fft(buf)
            for i in range(a,a*2):
                yield outbuf[i]
            m2 = order(buf,freqs)
            for i in range(a*2,a*3):
                yield outbuf[i]
            buf[m2<np.quantile(m2,next(quantile))] = 0
            buf[:] = scipy.fft.ifft(buf)
            for i in range(a*3,len(outbuf)):
                yield outbuf[i]
            outbuf[:] = buf
    except StopIteration:
        pass

def wavelet_peaks_ac(gen,bsize=64,quantile=63/64,wavelet='haar',order=mag2):
    import pywt
    quantile = genify(quantile)
    freqs = [len(v) for v in pywt.wavedec(np.zeros(bsize),wavelet)]
    freqs = np.concatenate([np.ones(v)*v for v in freqs])
    try:
        outbuf = np.zeros(bsize,dtype=complex)
        a = bsize//4
        while 1:
            buf = np.fromiter(gen,complex,bsize)
            for i in range(a):
                yield outbuf[i]
            wl = pywt.wavedec(buf,wavelet)
            t = np.concatenate(wl)
            for i in range(a,a*2):
                yield outbuf[i]
            m2 = order(t,freqs)
            t[m2<np.quantile(m2,next(quantile))] = 0
            for i in range(a*2,a*3):
                yield outbuf[i]
            o = 0
            for i in range(len(wl)):
                wl[i][:] = t[o:o+len(wl[i])]
                o += len(wl[i])
            buf[:] = pywt.waverec(wl,wavelet)
            for i in range(a*3,len(outbuf)):
                yield outbuf[i]
            outbuf[:] = buf
    except StopIteration:
        pass
    
def hadamard_cf_peaks_ac(gen,bsize=64,quantile=63/64,order=mag2):
    quantile = genify(quantile)
    freqs = gray(bitrev(bsize.bit_length())(np.arange(bsize)))
    try:
        buf = np.zeros(bsize,dtype=complex)
        outbuf = np.zeros(bsize,dtype=complex)
        a = bsize//2//4
        crossfade = (1-np.cos(np.arange(bsize)/bsize*np.pi*2))/2
        faded = np.zeros(bsize,dtype=complex)
        while 1:
            buf[:bsize//2] = buf[bsize//2:]
            buf[bsize//2:] = np.fromiter(gen,complex,bsize//2)
            for i in range(a):
                yield outbuf[i]
            faded[:] = buf
            fht_p2_ip(faded)
            for i in range(a,a*2):
                yield outbuf[i]
            m2 = order(faded,freqs)
            for i in range(a*2,a*3):
                yield outbuf[i]
            faded[m2<np.quantile(m2,next(quantile))] = 0
            fht_p2_ip(faded)
            faded *= crossfade
            for i in range(a*3,bsize//2):
                yield outbuf[i]
            outbuf[:bsize//2] = outbuf[bsize//2:]
            outbuf[bsize//2:] = 0
            outbuf += faded
    except StopIteration:
        pass

def fourier_cf_peaks_ac(gen,bsize=64,quantile=63/64,order=mag2):
    quantile = genify(quantile)
    freqs = .5-np.abs(np.arange(bsize)/bsize-.5)
    import scipy
    try:
        buf = np.zeros(bsize,dtype=complex)
        outbuf = np.zeros(bsize,dtype=complex)
        a = bsize//2//4
        crossfade = (1-np.cos(np.arange(bsize)*np.pi*2/bsize))/2
        faded = np.zeros(bsize,dtype=complex)
        while 1:
            buf[:bsize//2] = buf[bsize//2:]
            buf[bsize//2:] = np.fromiter(gen,complex,bsize//2)
            for i in range(a):
                yield outbuf[i]
            faded[:] = scipy.fft.fft(buf)
            for i in range(a,a*2):
                yield outbuf[i]
            m2 = order(faded,freqs)
            for i in range(a*2,a*3):
                yield outbuf[i]
            faded[m2<np.quantile(m2,next(quantile))] = 0
            faded[:] = scipy.fft.ifft(faded)*crossfade
            for i in range(a*3,bsize//2):
                yield outbuf[i]
            outbuf[:bsize//2] = outbuf[bsize//2:]
            outbuf[bsize//2:] = 0
            outbuf += faded

    except StopIteration:
        pass
                          

def wavelet_cf_peaks_ac(gen,bsize=64,quantile=63/64,wavelet='haar',order=mag2):
    import pywt
    quantile = genify(quantile)
    freqs = [len(v) for v in pywt.wavedec(np.zeros(bsize),wavelet)]
    freqs = np.concatenate([np.ones(v)*v for v in freqs])
    try:
        buf = np.zeros(bsize,dtype=complex)
        outbuf = np.zeros(bsize,dtype=complex)
        a = bsize//2//4
        crossfade = (1-np.cos(np.arange(bsize)*np.pi*2/bsize))/2
        faded = np.zeros(bsize,dtype=complex)
        while 1:
            buf[:bsize//2] = buf[bsize//2:]
            buf[bsize//2:] = np.fromiter(gen,complex,bsize//2)
            for i in range(a):
                yield outbuf[i]
            wl = pywt.wavedec(buf,wavelet)
            t = np.concatenate(wl)
            for i in range(a,a*2):
                yield outbuf[i]
            m2 = order(t,freqs)
            t[m2<np.quantile(m2,next(quantile))] = 0
            for i in range(a*2,a*3):
                yield outbuf[i]
            o = 0
            for i in range(len(wl)):
                wl[i][:] = t[o:o+len(wl[i])]
                o += len(wl[i])
            faded[:] = pywt.waverec(wl,wavelet)*crossfade
            for i in range(a*3,bsize//2):
                yield outbuf[i]
            outbuf[:bsize//2] = outbuf[bsize//2:]
            outbuf[bsize//2:] = 0
            outbuf += faded

        while 1:
            buf = np.fromiter(gen,complex,bsize)
            for i in range(len(buf)):
                yield buf[i]
    except StopIteration:
        pass
        

        

    

