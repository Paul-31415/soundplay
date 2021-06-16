



import numpy as np
import scipy as sp

#integrate impulses method
class sah_bwl:
    def __init__(self,sinc_size=8):
        self.ts = np.arange(sinc_size*2-1)-sinc_size
        self.buf = np.zeros(sinc_size*2-1,dtype=complex)
        self.tot = 0
        self.prev = 0
        self.width = sinc_size
    def kern(self,ts):
        r = np.sinc(ts)*(.5+.5*np.cos(ts*(np.pi/self.width)))
        t = np.sum(r)
        return r/t
    def __call__(self,v,dt=1):
        if int(dt) >= len(self.buf):
            r = np.concatenate((self.buf,np.zeros(int(dt)-len(self.buf))))
            self.buf[:] = 0
        else:
            r = np.copy(self.buf[:int(dt)])
            if int(dt):
                self.buf[:-int(dt)] = self.buf[int(dt):]
                self.buf[-int(dt):] = 0
        dt %= 1
        self.buf += self.kern(self.ts+dt)*(v-self.prev)
        self.prev = v
        if len(r):
            r[0] += self.tot
            np.cumsum(r,out=r)
            self.tot = r[-1]
        return r
