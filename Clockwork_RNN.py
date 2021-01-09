"""OA

"""
import numpy as np
from NNBlocks import tanh,sigm,relu,iden,isNone,expo,lrelu


def boxcars(periods):
    periods = np.array(periods,dtype=int)
    phases = np.zeros(periods.shape,dtype=int)
    def filt(v,ph=phases,pd=periods,stored=[None]):
        if isNone(stored[0]):
            stored[0] = np.zeros(periods.shape+v.shape)
        stored[0][ph==0] = 0
        stored[0] += v
        ph += 1
        f = ph>=pd
        ph[f] = 0
        return stored[0],f
    return filt

REGULARIZATION_FACTOR = 0.01
MAX_GRADIENT_STEP = 0.1
class NN:
    one = np.array([1])
    def __init__(self,shape=[1,8],af=tanh,history=4):
        self.hist = history
        self.hist_ind = 0
        try:
            len(af)
            self.af = af
        except:
            self.af = [af]*(len(shape)-1)
        self.shape = shape
        self.weights = [np.zeros((shape[i+1],shape[i]+1),dtype=float) for i in range(len(shape)-1)]
        self.gd_deltas = [np.copy(w) for w in self.weights]
        self.vals = [np.zeros((history,s),dtype=float) for s in shape]
        self.fvals = [np.zeros((history,s),dtype=float) for s in shape]
    def reset_mem(self):
        for i in range(len(self.vals)):
            self.vals[i].fill(0)
            self.fvals[i].fill(0)
        
    def scramble(self,keep=0,mag=1):
        for w in self.weights:
            w *= keep
            w += np.random.normal(w)*mag
    def len(self):
        return self.shape[-1]
    def hi(self,o=0):
        return (self.hist_ind+o)%self.hist
    def __call__(self,inp=None):
        if not isNone(inp):
            h = self.hist_ind = self.hi(1)
            self.fvals[0][h][:]= inp
            for i in range(len(self.weights)):
                self.vals[i+1][h][:]=self.weights[i]@np.concatenate((self.fvals[i][h],NN.one))
                self.fvals[i+1][h][:]=self.af[i](self.vals[i+1][h][:])
        return self.fvals[len(self.weights)][self.hi()]
    def grad_desc(self,prop,a=0.001,to=0):
        assert to < self.hist
        assert to >= 0
        h = self.hi(-to)
        for i in range(len(self.weights)-1,-1,-1):
            #print("1:","prop:",prop,"vals[i+1]:",self.vals[i+1][h],"weights[i]:",self.weights[i])
            prop = self.af[i].gradient(self.vals[i+1][h],self.fvals[i+1][h])*prop
            d = np.outer(prop,np.concatenate((self.fvals[i][h],NN.one)))
            #print("2:","prop:",prop,"fvals[i]:",self.fvals[i][h],"outer:",d)
            self.gd_deltas[i] -= d*a + self.weights[i]*REGULARIZATION_FACTOR*a
            prop = self.weights[i].transpose()[:-1]@prop
            #print("3:","prop:",prop)
        #print(" ")
        return prop
    def grad_apply(self,vel=0):
        for i in range(len(self.weights)):
            self.weights[i] += np.clip(self.gd_deltas[i],-MAX_GRADIENT_STEP,MAX_GRADIENT_STEP)
            self.gd_deltas[i] = np.clip(self.gd_deltas[i]*vel,-MAX_GRADIENT_STEP,MAX_GRADIENT_STEP)
    #gradient ascent
    def grad(self,prop,a=0.001,to=0):
        assert to < self.hist
        assert to >= 0
        h = self.hi(-to)
        for i in range(len(self.weights)-1,-1,-1):
            prop = self.af[i].gradient(self.vals[i+1][h],self.fvals[i+1][h])*prop
            d = np.outer(prop,np.concatenate((self.fvals[i][h],NN.one)))
            self.gd_deltas[i] += d*a #- self.weights[i]*REGULARIZATION_FACTOR*a
            prop = self.weights[i].transpose()[:-1]@prop
        return prop
    
def graph(func):
    from matplotlib import pyplot as plt
    plt.plot([i/100 for i in range(-2000,2000)],[func(inp=i/100)[0] for i in range(-2000,2000)])
    return plt
    plt.show(block=0)


def grad_test_nn():
    print("making (1,2,1) relu net:")
    print("        -3                ")
    print("    1 -> @ 2 \            ")
    print("->@           -> @        ")
    print("   -1 -> @ 3 /   -1       ")
    print("        +2                ")
    n = NN((1,2,1),relu)
    n.weights[0] = np.array([[1.,-3],[-1,2]])
    n.weights[1] = np.array([[2.,3,-1]])
    print("expect n(0) = 5")
    print("    -3 | 0          ")
    print(" 0          5 | 5   ")
    print("     2 | 2          ")
    print("got",n(0))
    print("====")
    print("expect n(1) = 2")
    print("    -2 | 0          ")
    print(" 1          2 | 2   ")
    print("     1 | 1          ")
    print("got",n(1))
    print("====")
    print("gradient step size 0.01")
    print("teaching n(1) = 0")
    print("expect this:")
    print(" [ 1,-3]  f  [2  3 -1]  f  -> 2 ")
    print(" [-1, 2]^  ^            ^     ^ ")
    print("gradients: ^ [0  2  2]  2     2 ")
    print("    ^   ^ 4,6 ")
    print("    ^  0,6")
    print(" [ 0, 0]   ")
    print(" [ 6, 6]   ")
    print("^< -6 ")
    g = n.grad_desc(n(1),0.01)
    print("got:",g,n.gd_deltas)
    n.grad_apply()
    print("now n(1) = ",n(1))

    

    
def test_nn(a=0.1,vel=0,shape=(1,2,1),t=sigm):
    #print("Making",shape,"sigm net")
    n = NN(shape,t)
    n.scramble()
    from matplotlib import pyplot as plt
    
    fig, ax = plt.subplots()
    ax.set_ylim((-2,2))
    
    p, = ax.plot([i/10 for i in range(-200,200)],[n(inp=i/10)[0] for i in range(-200,200)])

    def anim(i):
        for j in range(100):
            for x,y in [(0,1),(1,0),(-1,0)]:
                v = n(x)
                n.grad_desc(v-y,a)
            n.grad_apply(vel)
        p.set_ydata([n(inp=i/10)[0] for i in range(-200,200)])
        return p,
    import matplotlib.animation as animation
    ani = animation.FuncAnimation(fig,anim,interval=1)
            
    plt.show(block=0)
    return ani,anim,fig,ax,p

def test_nn_stack(a=0.1,vel=0,shape=[(1,2),(2,1)],t=[sigm,sigm],xr=[-1,0,1],yr=[0,1,0]):
    #print("Making",shape,"sigm net")
    ns = [NN(shape[i],t[i%len(t)]) for i in range(len(shape))]
    for n in ns:
        n.scramble()
    from matplotlib import pyplot as plt
    
    fig, ax = plt.subplots()
    ax.set_ylim((-2,2))

    def f(i):
        v = i
        for n in ns:
            v = n(v)
        return v

    def gd(g,a):
        for n in ns[::-1]:
            g = n.grad(g,a)
        return g
    
    p, = ax.plot([i/10 for i in range(-200,200)],[f(i/10)[0] for i in range(-200,200)])
    ax.plot(xr,yr,'o')
    def anim(i,a=a,vel=vel):
        for i in range(len(xr)):
            v = f(xr[i])
            gd(yr[i]-v,a)
        for n in ns:
            n.grad_apply(vel)
        p.set_ydata([f(i/10)[0] for i in range(-200,200)])
        return p,
    import matplotlib.animation as animation
    ani = animation.FuncAnimation(fig,anim,interval=1)
            
    plt.show(block=0)
    return ani,anim,fig,ax,p


class NoGC:
    def __init__(self,*stuff):
        self.stuff = stuff
    def __repr__(self):
        return "NoGC()"
import random
def test_rnn(a=0.1,vel=0,l=2,h=5,shape=(2,2),t=sigm,noise=0):
    #print("Making",shape,"sigm net")
    n = NN(shape,t,h)
    n.scramble()
    from matplotlib import pyplot as plt
    if type(l) == int:
        l = lambda i,l=l: i == l
    if type(a) != type(l):
        a = lambda i,l,v=a : v
    if type(vel) != type(l):
        vel = lambda i,l,v=vel : v

    def f(l,res=[n()]):
        n.reset_mem()
        r = []
        n(np.zeros(n.len()))
        #n(res[0])
        #for i in range(int((random.random()*10+1)*h)):
        #    n(np.concatenate((np.zeros(1),n()[1:])))
        #res[0] = np.concatenate((np.zeros(1),n()[1:]))
        for i in range(h-l):
            r += [np.copy(n(np.concatenate((np.zeros(1),n()[1:]))))]
        #r += [np.zeros(n.len())]*(h-l)
        for i in range(l):
            r += [np.copy(n(np.concatenate((np.ones(1),n()[1:]))))]
        r += [np.copy(n(np.concatenate((np.zeros(1),n()[1:]))))]
        return r
    

    def teach(p,a):
        prop = np.zeros(n.len(),dtype=float)
        prop[0] = p
        for i in range(h):
            prop[1:] = n.grad(np.tanh(prop),a,i)[1:]
            #print(i,prop[1:])
            prop[0] = 0
    
    fig, ax = plt.subplots()
    ax.set_ylim((-2,2))
    d = [f(i) for i in range(h)]
    def dat(i,j,d):
        return [e[j] for e in d[i]]
    p = [[ax.plot([i-1+(o+1)/(h+1) for o in range(h+1)],dat(i,j,d))[0] for j in range(n.len())] for i in range(h)]
    r, = ax.plot(range(h),[p[-1][0] for p in d],'o')
    g, = ax.plot(range(h),[0 for p in d],'1')
    t, = ax.plot(range(h),[l(i) for i in range(h)],'x')
    def anim(ind):
        #for j in range(100):
        d = []
        gd = []
        for i in range(h):
            v = f(i)
            gd += [l(i)-v[-1][0]]
            teach(gd[-1],a(ind,gd[-1]))
            d += [v]
            n.grad_apply(vel(ind,gd[-1]))
        n.scramble(1,noise)
        r.set_ydata([p[-1][0] for p in d])
        for i in range(len(p)):
            for j in range(len(p[i])):
                p[i][j].set_ydata(dat(i,j,d))
        g.set_ydata([-gd[i] for i in range(h)])
        return r,g,*sum(p,[])
    import matplotlib.animation as animation
    ani = animation.FuncAnimation(fig,anim,interval=1)
    plt.show(block=0)
    return n,NoGC(ani,anim,fig,ax,p)












DEFAULT_SUBNET = [8]
            
class Clockwork_RNN:
    def __init__(self,**ka):
        params = {'feedback_activation_function':tanh,
                  'output_activation_function':iden,
                  'resample_filter':boxcars,
                  'clock_periods':[1<<i for i in range(16)],
                  'subnet_sizes':[DEFAULT_SUBNET for i in range(16)],
                  'subnet_activation_function':tanh,
                  'outputs':2,
                  'inputs':1,
                  'feedback_history':3}
        params.update(ka)
        
        netdescs = [(params['clock_periods'][i],params['subnet_sizes'][i]) for i in range(len(params['clock_periods']))]
        netdescs.sort(key=lambda a:a[0])
        self.subnets = []
        chain = 0
        self.inps = params['inputs']
        self.hist = params['feedback_history']
        self.netdescs = netdescs
        self.net_offs = [params['outputs']]
        self.bp_accs = []


        #separate the output weight matrix into ones for each net
        self.output_weights = []
        self.gd_delta = []
        self.output_af = params['output_activation_function']
        self.output_parts = [np.zeros(params['outputs']) for i in netdescs]
        self.output = np.zeros(params['outputs'])
        self.output_unaf = np.zeros(params['outputs'])
        
        #build from slowest to fastest
        for nd in netdescs[::-1]:
            s = nd[1][-1]
            chain += s #recurrent inputs
            self.output_weights += [np.zeros((params['outputs'],1+s),dtype=float)]
            self.bp_accs += [np.zeros(s)]
            self.net_offs += [chain+params['outputs']]
            subnet = NN([chain+params['inputs']]+nd[1],
                        params['subnet_activation_function'],
                        self.hist)
            self.subnets += [subnet]
        self.subnets = self.subnets[::-1] #have [0] be fastest
        self.net_offs = self.net_offs[::-1]
        self.output_weights = self.output_weights[::-1]
        self.gd_delta = [np.zeros(w.shape) for w in self.output_weights]
        self.ran = None


        self.alphas = [1]*len(self.subnets)
        self.decays = [.5]*len(self.subnets)
        
        
        self.decimation_filt = params['resample_filter']([n[0] for n in netdescs])
        self.feedback_af = params['feedback_activation_function']

        
        
        
    def _construct_subnet_input(self,netid,inp):
        #input of net is concat(inp, slowest,...,recurrent)
        assert len(inp) == self.inps
        cons = (inp,)
        for i in range(len(self.subnets)-1,netid-1,-1):
            sn = self.subnets[i]
            #nol = self.net_offs[netid]
            #noh = self.net_offs[netid+1]
            cons += (self.feedback_af(sn()),)
        return np.concatenate(cons)
    def _construct_output_layer_input(self):
        cons = (NN.one,)
        for sn in self.subnets:
            cons += (sn(),)
        return np.concatenate(cons)
    def __call__(self,inp=None):
        if not isNone(inp):
            #fiter
            inps,self.ran = self.decimation_filt(inp)
            self.output_unaf.fill(0)
            for i in range(len(self.ran)):
                if self.ran[i]:
                    sni = self._construct_subnet_input(i,inps[i])
                    self.output_parts[i][:] = self.output_weights[i] @ np.concatenate((self.subnets[i](sni),NN.one))
                self.output_unaf += self.output_parts[i]
            self.output[:] = self.output_af(self.output_unaf)
        return self.output
    def grad_desc(self,prop,a=0.001):
        prop = self.output_af.gradient(self.output_unaf,self.output)*prop
        #self.gd_delta += a*np.outer(prop,self._construct_output_layer_input())
        #prop = self.output_weights[1:].transpose()@prop
        for i in range(len(self.bp_accs)):
            self.gd_delta[i] += a*np.outer(prop,np.concatenate((self.subnets[i](),NN.one))) + self.output_weights[i]*REGULARIZATION_FACTOR*a
            self.bp_accs[i] += self.output_weights[i].transpose()[:-1]@prop
        #apply grad descent to the nets that ran
        for i in range(len(self.ran)):
            if self.ran[i]:
                prop = self.bp_accs[i]
                for h in range(self.hist):
                    prop = self.subnets[i].grad_desc(prop,a*self.alphas[i])
                    #now take apart this to get the gradients for the slower nets
                    offs = self.inps
                    for si in range(len(self.subnets)-1,i,-1):
                        sn = self.subnets[si]
                        l = sn.len()
                        self.bp_accs[si] += self.feedback_af.gradient(prop[offs:offs+l])*self.alphas[i]
                        offs += l
                    prop = self.feedback_af.gradient(self.subnets[i]())*prop[offs:]*self.decays[i]
                    #recurse
                #reset the backprop gradient accumulator
                self.bp_accs[i].fill(0)
    def grad_apply(self,vel=0):
        for i in range(len(self.ran)):
            if self.ran[i]:
                self.subnets[i].grad_apply(vel)
                self.output_weights[i] -= np.clip(self.gd_delta[i],-MAX_GRADIENT_STEP,MAX_GRADIENT_STEP)
                self.gd_delta[i] *= vel
    def scramble(self,keep=0,mag=1):
        for i in range(len(self.subnets)):
            self.output_weights[i] *= keep
            self.output_weights[i] += np.random.normal(self.output_weights[i])*mag
        for sn in self.subnets:
            sn.scramble(keep,mag)

n = Clockwork_RNN()
