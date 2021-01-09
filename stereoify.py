#techniques for making a mono signal 'sound stereo'





### ML section
from Clockwork_RNN import NN
from NNBlocks import *



class NCF:#net controled FIR
    def __init__(self,firlen=256,netDesc = (256,256),recs=256,afs=tanh,hist=2):
        nd = (netDesc[0]+recs,)+netDesc[1:]+(firlen+recs,)
        self.net = NN(nd,afs,hist)
        self.past = np.zeros(max(firlen+recs,nd[0]),dtype=float)
        self.fl = firlen
        self.r = recs
        self.ni = netDesc[0]
    def __call__(self,inp):
        l = self.past.shape[0]
        sr = l-self.r
        self.past[:sr-1] = self.past[1:sr]
        self.past[sr-1] = inp
        o = self.net(self.past[sr-self.ni:])
        self.past[sr:] = o[:self.r]
        return np.inner(o[self.r:],self.past[sr-self.fl:sr])
    def past_inp(self,o=0):
        return self.past[self.past.shape[0]-self.r-o-1]
    def grad(self,v,a=0.001):
        l=self.past.shape[0]
        sr = l-self.r
        prop = np.concatenate((np.zeros(self.r),self.past[sr-self.fl:sr]*v))
        for h in range(self.net.hist):
            prop[:self.r] = self.net.grad(prop,a,h)[sr:]
            prop[self.r:] = 0
    def teach(self,vel=0):
        self.net.grad_apply(vel)
    def reset_mem(self):
        self.past.fill(0)

class NCF_stereoify:
    def __init__(self,ncf,delay=128):
        self.ncf = ncf
        self.delay = delay
    def train(self,g,tm=100,af=lambda l:0.001,*ta):
        i = 0
        al = 0
        dly = np.zeros(self.delay,dtype=float)
        cdly = np.zeros(self.delay,dtype=float)
        n = self.ncf
        for v in g:
            c = v.real+v.imag
            d = dly[0]
            dly[:-1] = dly[1:]
            dly[-1] = v.real-v.imag
            
            p = n(c)
            e = (d-p)
            l = e*e/2
            al += l
            n.grad(e,af(l))
            if i*tm==0:
                n.teach(*ta)
        
            if i%1000 == 0:
                print(i,"avg Loss:",al/1000,end="   \r")
                al = 0
            i += 1
            r = (cdly[0]+p*1j)*(.5+.5j)
            cdly[:-1] = cdly[1:]
            cdly[-1] = c
            yield r
    def run(self,g):
        for v in g:
            p = self.ncf(v.real+v.imag)
            yield (self.ncf.past_inp(self.delay)+1j*p)*(.5+.5j)
            
def rung(g):
    for i in g:
        pass
