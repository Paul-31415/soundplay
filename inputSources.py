import math


#matplotlib keyboard
noteKeys = "awsedftgyhujkolp;'"
class keyboard:
    def __init__(self,keystring="awsedftgyhujkolp;'"):
        self.keys = keystring
        self._keepinscope = None
        self.keysDown = set()
    def plot(self,ival=50):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        fig, ax = plt.subplots()
        fig.canvas.mpl_disconnect(
            fig.canvas.manager.key_press_handler_id)
        def press(event):
            if event.key in self.keys:
                self.keysDown.add(event.key)
                return True
        def release(event):
            if event.key in self.keysDown:
                self.keysDown.remove(event.key)
                return True
        ax.set_title("keyboard")
        fig.canvas.mpl_connect("key_release_event",release)
        fig.canvas.mpl_connect('key_press_event', press)
        def no(i):
            pass
        ani = animation.FuncAnimation(fig,no,interval=50)
        self._keepinscope = [press,release,no]
        return plt

eone = math.exp(2*math.pi)
def sinKeyboard(kb,fbase=440,offs=7,nk=noteKeys,sr=48000):
    factors = [eone**(1j*fbase*2**((i-offs)/12)/sr) for i in range(len(nk))]
    oscilators = [1]*len(factors)
    while 1:
        t = 0
        for k in kb.keysDown:
            i = nk.index(k)
            oscilators[i] *= factors[i]
            t += oscilators[i]
        yield t

def sindKeyboard(kb,fbase=440,start=.01,attak=1.01,decay=.99,offs=7,nk=noteKeys,sr=48000):
    factors = [eone**(1j*fbase*2**((i-offs)/12)/sr) for i in range(len(nk))]
    oscilators = [start]*len(factors)
    alive = set()
    while 1:
        t = 0
        for i in range(len(factors)):
            if nk[i] in kb.keysDown:
                s = abs(oscilators[i])
                alive.add(i)
                if s*attak>1:
                    oscilators[i] *= factors[i]/s
                else:
                    oscilators[i] *= factors[i]*attak
            elif i in alive:
                oscilators[i] *= factors[i]*decay
                if abs(oscilators[i]) < start:
                    alive.remove(i)
            else:
                continue
            t += oscilators[i]
        yield t

def delaylineKeyboard(kb,f=lambda x:x*.75,fbase=440,offs=7,nk=noteKeys,sr=48000):
    delays = [sr/(fbase*2**((i-offs)/12)) for i in range(len(nk))]
    #nearest interp for now
    delays = [round(d) for d in delays]

    lines = [[0]*d for d in delays]
    def do(v,i=[0],l=lines):
        t = 0
        n = 0
        for o in range(len(l)):
            p = l[o][i[0]%len(l[o])]
            l[o][i[0]%len(l[o])] = v
            if nk[o] in kb.keysDown:
                l[o][i[0]%len(l[o])] += f(p)
            t += l[o][i[0]%len(l[o])]
            n += 1
        i[0] += 1
        return t/max(1,n)
    return do

from filters import biquadPeak

l2 = math.log(2)

def filtkeyb(kb,fbase=440,offs=7,fg = lambda f:biquadPeak(f,f*l2/12,10),nk=noteKeys,sr=48000):
    filts = [fg((2*fbase*2**((i-offs)/12))/sr) for i in range(len(nk))]
    def do(v):
        for o in range(len(filts)):
            if nk[o] in kb.keysDown:
                v = filts[o](v)
        return v
    return do
