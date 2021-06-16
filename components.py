#fancy components
import math

class exp_resamp:
    def __init__(self,hl=.5):
        self.hl = hl
        self.a = 0
        self.v = 0
    def tick(self,dt):
        m = 2**(-dt/self.hl)
        self.a += (1-m)*(self.v-a)
    def __call__(self,v,dt=1):
        self.v = v
        self.tick(dt)
        return self.a
    def peek(self):
        return self.a


class delayline:
    def __init__(self,l=32,dl=0,bl=4,rs=exp_resamp):
        self.b = [0]*(l*bl)
        self.l = l
        self.i = 0
        self.dl = dl
        self.r = rs()
        self.cv = 0
        self.c = False
    def plot(self,plot=True):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider,CheckButtons
        from matplotlib.animation import FuncAnimation
        axm = plt.axes([0.1, 0.2, .8, .1])
        axl = plt.axes([0.1, 0.5, .8, .1])
        sm = Slider(axm, 'dl', -2, 2, valinit=self.dl)
        sl = Slider(axl, 'l', 0, len(self.b), valinit=self.l)
        def ud(val,s=self):
            s.dl = val
        def ul(val,s=self):
            s.l = val
        def an(i,s=sl,f=self):
            sl.set_val(f.l)
            return sl,
        sm.on_changed(ud)
        sl.on_changed(ul)
        fig = plt.figure()
        ani = FuncAnimation(fig,an)
        self._nogc = (ud,ul,an,ani)
        if plot:
            plt.show(block=0)
        return plt

    def peek(self):
        if self.c:
            return self.cv
        i = int(self.l+self.i)%len(self.b)
        if self.r:
            self.cv = self.r.peek()
        else:
            self.cv = self.b[i]
        self.c = True
        return self.cv
    """    def push(self,v):
        self.c = False
        if -self.dl > self.l:
            self.dl = -self.l
        self.b[self.i] = v
        self.i = (self.i-1)%len(self.b)
        if self.r:
            d = self.dl-1
            self.l += 1
            if d > 0:
                while d != 0:
                    self.l += 
        
        else:
            self.l += self.dl
        if self.l > len(self.b)-2:
            self.b = self.b[self.i:]+[0]*len(self.b)+self.b[:self.i]
            self.i = len(self.b)-1"""
    def __call__(self,v):
        r = self.peek()
        self.push(v)
        return r
        
class waveguide:
    def __init__(self,l=32,dl=0,rs=exp_resamp):
        self.fb = [0]*l
        self.bb = [0]*l
        self.l = l
        self.i = 0
        self.rsf = rs()
        self.rsb = rs()
