"""
Building blocks for neural nets and data processing stuff

"""

import numpy as np

def isNone(o): #numpy arrays make == messy
    return type(o)==type(None)

class ActFunc:
    #activation function
    def __init__(self,f,d,n="activation function"):
        self.f = f
        self.d = d
        self.name = n
    def __call__(self,*a):
        return self.f(*a)
    def gradient(self,val,fv=None):
        try:
            return self.d(val,fv)
        except:
            return self.d(val,self.f(val))
        
tanh = ActFunc(np.tanh,lambda x,v: 1 - v*v,"tanh")
sigm = ActFunc(lambda x: (np.tanh(x/2)+1)/2,lambda x,v: v*(1 - v),"sigmoid")
relu = ActFunc(lambda x: (np.abs(x)+x)/2,lambda x,v: x>0,"relu")
lrelu = ActFunc(lambda x: (np.abs(x)+x*1.05)/2,lambda x,v: .1+(x>0),"leaky relu")
iden = ActFunc(lambda x: x, lambda x,v: np.ones(x.shape),"x->x")
expo = ActFunc(lambda x: np.exp(x), lambda x,v: v+0, "e^x")

def Combine_Acts(a1,a2,r=0.5):
    def do(v,a=a1,b=a2,r=r):
        s = int(v.shape[0]*r)
        return np.concatenate((a(v[:s]),b(v[s:])))
    def grad(v,pv=None,a=a1,b=a2,r=r):
        s = int(v.shape[0]*r)
        if isNone(pv):
            pva = None
            pvb = None
        else:
            pva = pv[:s]
            pvb = pv[s:]
        return np.concatenate((a.gradient(v[:s],pva),b.gradient(v[s:],pvb)))
    return ActFunc(do,grad,"["+a1.name+" |"+str(r)+"| "+a2.name+"]")

tanh_and_sigm = Combine_Acts(tanh,sigm)




class DelayedGradient:
    def __init__(self,func,*args):
        self.f = func
        self.a = args
    def __call__(self,prop):
        return self.f(prop,*self.a)
    



    
#I know I could use tensorflow to do this, but I want to be able
# to implement it on a system that doesn't have tf.
"""class FCL: #Fully Connected Layer
    def __init__(self,i,o,f=iden,dt=float):
        self.w = np.zeros((o,i),dtype=dt)
        self.af = f
        self.o = np.zeros(o,dtype=dt) #before application of af
        self.i = np.zeros(i,dtype=dt)
    def __call__(self,inps=None):
        if not isNone(inps):
            self.i[:len(inps)] = inps
            self.o[:] = self.w @ self.i
        return self.af(self.o)
    
    def grad(self,prop=None):
        #returns (this's gradient, back prop)
        def gf(p,inp,mat,outp,f):
            #grad f(Ax) = ∂f(Ax) * ∂(Ax)
            op = f.gradient(outp)*prop
            mg = np.array([op]).transpose()@np.array([inp])
            ig = mat.transpose()@op
            return mg,ig
        if isNone(prop):
            co = np.copy(self.o)
            ci = np.copy(self.i)
            return DelayedGradient(gf,ci,self.w,co,self.af)
        return gf(prop,self.i,self.w,self.o,self.af)
    def descend(self,mat,alpha=0.001):
        self.w -= mat*alpha
    def scramble(self,scale=1,keep=0):
        self.w *= keep
        self.w += np.random.normal(np.zeros(self.w.shape))*scale
"""


def chain(*f):
    if len(f) == 0:
        def nop(*a):
            pass
        return nop
    if len(f) == 1:
        return f[0]
    def chainCurry(a,f=f):
        for func in f:
            f(*a)
    def chained(*a):
        chainCurry(a)
    return chained

#I need to figure out a hookup scheme before proceeding
#probably the pushthrough calls form that works well with audio.
# but, for concat cells, it'll have to do pull? or everything else
# needs to push and it only pushes when all inputs have pushed?
#  Let's try that for now?
##Actually, gradient descent needs to pass info from outputs to
#  inputs, so a generator or pull through method would be simpler
# i.e. call get on the final output and it requests the requred
#   inputs be calculated


#recursive gradient with **kwargs passthrough for other args like
# depth?

#simpler atomics than fcl, vector pipes and whatnot
#components:
#    (legend a[b] is a vecs of b size)
#        const -> 1[]  
# n[] -> concat -> 1[]
# 1[] -> split -> n[]
# 1[m] -> mat -> 1[n]
# 1[n] -> func -> 1[n]

REPR_DEPTH = 10
ARROW = "→"
GRAD_MAX = 0.1
class Component: #default is identity layer
    def __init__(self):
        self.inp = None
        self.name = "Component"
    def connect(self,inp):
        self.inp = inp
        return inp
    def rec(self,f):
        f(self)
        if self.inp != None:
            self.inp.rec(f)
    def children(self):
        if isNone(self.inp):
            for i in ():
                yield i
        try:
            for i in self.inp:
                yield i
        except:
            yield self.inp
    def __call__(self,*a,**ka):
        self.clear()
        return self.calc(*a,**ka)
    def calc(self,*a,**ka):
        return self.inp.calc(*a,**ka)
    def clear(self):
        for c in self.children():
            c.clear()
    def grad_desc(self,p,a=0.001):
        r = dict()
        for c in self.children():
            r.update(c.grad_desc(p,a))
        return r
    def grad_desc_curry(self):
        return chain(*(cg for cg in (c.grad_desc_curry() for c in self.children()) if cg != None))
    def grad_desc_fin(self):
        for c in self.children():
            c.grad_desc_fin()
            #by explicitly only applying the descent here,
            # we let any component be a Tee
    def __repr__(self,depth=REPR_DEPTH):
        if type(self.name) != str:
            name = self.name()
        else:
            name = self.name
        if depth <= 0:
            if type(self.inp) != type(None):
                return "..."+ARROW+name
            return self.name
        try:
            return self.inp.__repr__(depth-1)+ARROW+name
        except:
            try:
                return "("+",".join((i.__repr__(depth-1) for i in self.inp))+")"+name
            except:
                return repr(self.inp)+ARROW+name


    #def __getitem__(self,s):
    #    return Slice(s,self)

class Matmult(Component):
    CURRY_SAVE_MATRIX=True
    def __init__(self,i,o,dt=float):
        self.weights = np.zeros((o,i),dtype=dt)
        self.gd_delta = np.zeros(self.weights.shape,dtype=dt)
        self.oldInp = np.zeros(i,dtype=dt)
        self.a = 1
        self.inp = None
        self.name = "Matmult["+str(i)+","+str(o)+"]"
        self.cached = False
    def scramble(self,scale=1,keep=0):
        self.weights *= keep
        self.weights += np.random.normal(np.zeros(self.weights.shape))*scale
    def calc(self,*a,**ka):
        if not self.cached:
            self.oldInp[:] = self.inp.calc(*a,**ka)
            self.cached = True
        return self.weights @ self.oldInp
    def clear(self):
        self.cached = False
        super().clear()
    def grad_desc(self,p,a=0.001):
        ig = self.weights.transpose()@p
        self.gd_delta += self.a*a*np.array([p]).transpose()@np.array([self.oldInp])
        return super().grad_desc(ig,a)
    def grad_desc_fin(self):
        self.weights += self.gd_delta
        self.gd_delta.fill(0)
        super().grad_desc_fin()
    def grad_desc_curry(self):
        prop = self.inp.grad_desc_curry()
        if CURRY_SAVE_MATRIX:
            save = np.copy(self.weights)
        else:
            save = self.weights
        if prop != None:
            def g_curried(p,a=0.001,oi=np.copy(self.oldInp),ref=self,m=save,prop=prop):
                ref.gd_delta += ref.a*a*np.array([p]).transpose()@np.array([oi])
                prop(m.transpose()@p,a)
        else:
            def g_curried(p,a=0.001,oi=np.copy(self.oldInp),ref=self):
                ref.gd_delta += ref.a*a*np.array([p]).transpose()@np.array([oi])
        return g_curried

class Affine(Component):
    one = np.array([1.])
    def __init__(self,i,o,dt=float):
        self.weights = np.zeros((o,i+1),dtype=dt)
        self.gd_delta = np.zeros(self.weights.shape,dtype=dt)
        self.oldInp = np.zeros(i,dtype=dt)
        self.a = 1
        self.inp = None
        self.name = "Affine["+str(i)+","+str(o)+"]"
        self.cached = False
    def scramble(self,scale=1,keep=0):
        self.weights *= keep
        self.weights += np.random.normal(np.zeros(self.weights.shape))*scale
    def calc(self,*a,**ka):
        if not self.cached:
            self.oldInp[:] = self.inp.calc(*a,**ka)
            self.cached = True
        return self.weights @ np.concatenate((self.oldInp,Affine.one))
    def clear(self):
        self.cached = False
        super().clear()
    def grad_desc(self,p,a=0.001):
        ig = self.weights.transpose()[:-1]@p
        self.gd_delta -= self.a*a*np.array([p]).transpose()@np.array([np.concatenate((self.oldInp,Affine.one))])
        return super().grad_desc(ig,a)
    def grad_desc_fin(self):
        self.weights += np.clip(self.gd_delta,-GRAD_MAX,GRAD_MAX)
        self.gd_delta.fill(0)
        super().grad_desc_fin()
            
class Func(Component):
    def __init__(self,af):
        self.af = af
        self.oldInp = None
        self.inp = None
        self.name = af.name
        self.cached = False
    def calc(self,*a,**ka):
        if not self.cached:
            self.oldInp = self.inp.calc(*a,**ka)
            self.cached = True
        return self.af(self.oldInp)
    def clear(self):
        self.cached = False
        super().clear()
    def grad_desc(self,p,a=0.001):
        return super().grad_desc(p*self.af.gradient(self.oldInp),a)
    def grad_desc_curry(self):
        prop = self.inp.grad_desc_curry()
        if prop == None:
            return prop
        goi = self.af.gradient(self.oldInp)
        def g_curried(p,a=0.001,g=goi,prop=prop):
            prop(p*g,a)
        return g_curried

class Input(Component):
    def __init__(self,key,verbatim=False):
        self.key = key
        self.verbatim = verbatim
    def calc(self,*a,**ka):
        if self.verbatim:
            return ka[self.key]
        try:
            l = len(ka[self.key])
            return np.array(ka[self.key])
        except:
            return np.array([ka[self.key]])
    def children(self):
        for i in ():
            yield i
    def clear(self):
        pass
    def grad_desc(self,p,a=0.001):
        return {self.key:p}
    def __repr__(self,depth=REPR_DEPTH):
        return "kwargs["+repr(self.key)+"]"
class Const(Component):
    def __init__(self,val,name="Const"):
        self.val = val
        self.name = name
    def calc(self,*a,**ka):
        return self.val
    def clear(self):
        pass
    def children(self):
        for i in ():
            yield i
    def grad_desc(self,p,a=0.001):
        return {self:p}
    def grad_desc_fin(self):
        pass
    def rec(self,f):
        f(self)
    def __repr__(self,depth=REPR_DEPTH):
        return self.name
class Concat(Component):
    def __init__(self,*inps):
        self.inp = inps
        self.name = "=>"
        self.insizes = None
    def connect(self,inp):
        self.inp += (inp,)
        return inp
    def calc(self,*a,**ka):
        parts = [i(*a,**ka) for i in self.inp]
        if self.insizes == None:
            self.insizes = [len(p) for p in parts]
        return np.concatenate(parts)
    def clear(self):
        for i in self.inp:
            i.clear()
    def grad_desc(self,p,a=0.001):
        i = 0
        offs = 0
        r = dict()
        for inp in self.inp:
            try:
                rp = inp.grad_desc(p[offs:offs+self.insizes[i]],a)
            except AttributeError:
                pass
            r.update(rp)
            offs += self.insizes[i]
            i += 1
        return r
    def grad_desc_fin(self):
        for inp in self.inp:
            inp.grad_desc_fin()
    def grad_desc_curry(self):
        ch = [c.grad_desc_curry() for c in self.children()]
        def g_curried(p,a=0.001,ch=ch):
            i = 0
            offs = 0
            for c in ch:
                if c != None:
                    c(p[offs:offs+self.insizes[i]],a)
                offs += self.insizes[i]
                i += 1
        return g_curried
    def rec(self,f):
        f(self)
        for inp in self.inp:
            inp.rec(f)
class Slice(Component):
    def __init__(self,slce=slice(None),inp=None):
        self.slice = slce
        self.inp = inp
        self.cached = False
        self.oldInp = None
        self.gradc = None
        if type(slce) == slice:
            slem = [slce.start,slce.stop] + \
                ([slce.step] if slce.step != None and slce.step != 1 else [])
            slstr = ":".join(("" if e == None else repr(e) for e in slem))
        else:
            slstr = repr(slce)
        self.name = "["+slstr+"]"
    def clear(self):
        self.cached = False
        super().clear()
    def calc(self,*a,**ka):
        if not self.cached:
            self.oldInp = self.inp.calc(*a,**ka)
            self.cached = True
        return self.oldInp[self.slice]
    def grad_desc(self,p,a=0.001):
        if isNone(self.gradc):
            self.gradc = np.zeros(self.oldInp.shape)
        self.gradc[self.slice] = p
        return self.inp.grad_desc(self.gradc,a)
    def grad_desc_curry(self):
        ch = self.inp.grad_desc_curry()
        def g_curried(p,a=0.001,z=np.zeros(self.oldInp.shape),s=self.slice,ch=ch):
            z[s] = p
            ch(z,a)
        return g_curried
    
#delay allows for rnns
class Delay(Component):
    #This'll make gradient descent more annoying.
    # requires 'currying' of gradients without p.
    # but maybe I can avoid that if I make the
    #  assumption that grad_desc will be called
    #  either between each step, or not at all...
    #
    #i.e. Normal RNN gradient descent seems to operate
    # on the assumption that the net is not learning
    # while it's being used. I wonder if removing
    # that assumption could help make a smaller algorithm.
    #
    #
    def __init__(self,dly=1,gradProps=5):
        self.i = 0
        self.dbuf = [None]*dly
        self.got = False
        self.outc = None
        self.gc_buf = []
        self.prop_lim = gradProps
        self.props = 0
        self.inp = None
        self.name = "Delay("+str(dly)+")"
    def calc(self,*a,**ka):
        if not self.got:
            self.outc,self.dbuf[self.i] = self.inp.calc(*a,**ka),self.dbuf[i]
            self.got = True
        if isNone(self.outc):
            return np.zeros(self.dbuf[self.i].shape)
        return self.outc
    def clear(self):
        self.got = False
        self.i = (self.i+1)%len(self.dbuf)
        super().clear()
    def grad_desc(self,p,a=0.001):
        return None #block prop
        
            
    def grad_desc_curry(self):
        return None #block prop
        if self.props < self.prop_lim:
            self.props += 1
            c=self.inp.grad_desc_curry(p,a)

"""class Tee(Component):
    def __init__(self):
        self.inp = None
        self.oldInp = None
        self.cached = False
        self.grad = None
        self.a = 0
    def calc(self,*a,**ka):
        if not self.cached:
            self.oldInp = self.inp.calc(*a,**ka)
            self.cached = True
        return self.oldInp
    def clear(self):
        self.cached = False
        super().clear()
    def grad_desc(self,p,a=0.001):
        if self.grad == None:
            self.grad = np.copy(p)
            self.a = a
        else:
            self.grad += p*(a/self.a)
        return dict()
    def grad_desc_fin(self):
        self.inp.grad_desc(self.grad,self.a)
        self.grad = None
        self.inp.grad_desc_fin()
"""
class DSF:
    def __init__(self):
        self.v = None
    def send(self,v):
        self.v = v
    def get(self):
        v = self.v
        self.v = None
        return v
class DSF_BoxCar(DSF):
    def send(self,v):
        if isNone(self.v):
            self.v = v
        else:
            self.v += v
class Upsample(Component):
    def __init__(self,factor=2,filt=DSF_BoxCar):
        self.phase = factor-1
        self.factor = factor
        self.oldInp = None
        self.inp = None
        self.clock = []
        self.gfilt = filt()
    def name(self):
        return "Upsample("+str(self.phase+1)+"%"+str(self.factor)+")"
    def calc(self,*a,**ka):
        #since this call goes recursively backwards from the end,
        # it looks like downsampling here.
        self.phase = (self.phase+1)%self.factor
        if self.phase == 0:
            self.oldInp = self.inp.calc(*a,**ka)
        #clock stuff in clock
        for c in self.clock:
            c.calc(*a,**ka)
        return self.oldInp
    def clear(self):
        if self.phase == 0:
            self.inp.clear()
        for c in self.clock:
            c.clear()
    def grad_desc(self,p,a=0.001):
        self.gfilt.send(p*a)
        if self.phase == self.factor-1:
            self.inp.grad_desc(self.gfilt.get(),1)
    def grad_desc_fin(self):
        if self.phase == self.factor-1:
            self.inp.grad_desc_fin()
    def grad_desc_curry(self):
        return None
class Downsample(Component):
    #this'll be more annoying to do, since
    # because of the recursive structure of
    # the calls, we will have to do workarounds
    # to get this to be clocked often enough.
    def __init__(self,factor=2,filt=DSF_BoxCar):
        #self.phase = factor-1
        #self.factor = factor
        self.inp = None
        self.filt = filt() 
        #self.oldg = None #no learning past this yet.
        self.name = "Downsampler"
        self.calced = False
        class Slow_Side(Component):
            def __init__(self,ds):
                self.ds = ds
                self.name = "Downsampled"
                self.cached = False
                self.cache = None
            def calc(self,*a,**ka):
                if not self.cached:
                    self.cache = self.ds.filt.get()
                    self.cached = True
                return self.cache
            def clear(self):
                self.cached = False
            def grad_desc(self,p,a=0.001):
                return dict()
            def grad_desc_fin(self):
                pass
            def grad_desc_curry(self):
                return None
        self.slow_side = Slow_Side(self)
    def calc(self,*a,**ka):
        if not self.calced:
            self.filt.send(self.inp.calc(*a,**ka))
            self.calced = True
    def clear(self):
        self.calced = False
        
    #for now, these components will stop gradient descent
    # because of the whole issue of having to advance the
    # slowest net at least once to do any learning on the
    # faster ones.
    #This shouldn't be an issue because slow nets don't
    # get any inputs from faster nets, only from the whole
    # net's inputs, so nothing before these can learn.
    def grad_desc(self,p,a=0.001):
        return dict()
    def grad_desc_fin(self):
        pass
    def grad_desc_curry(self):
        return None
    #something to try when revisiting this is allowing the
    # slow nets to learn at the same rate as the fast ones
    # by basically making them all the same fast speed and
    # just downsampling the inputs to them.
    #Unfortunately, that causes this architecture to lose
    # the speedup from the underclocking, but at least you
    # don't have to travel a full step of the slowest net
    # to get any learning done.
    

#simple fully connected net
def net(layer_sizes = [5,1],af=tanh,inp_key = "inp"):
    netpart = Func(af)
    out = netpart
    for i in range(len(layer_sizes)-2,-1,-1):
        # (...(((inp . 1) -> @ -> f . 1) -> @ -> f . 1) ... -> @ -> f
        
        n,p = layer_sizes[i+1],layer_sizes[i]
        m = Matmult(p+1,n)
        m.scramble()
        netpart = netpart.connect(m) # @
        netpart = netpart.connect(Func(af)) # f
        conc = Concat(Const(np.array([1]),'c1')) # (1 . <>)
        netpart = netpart.connect(conc)
    netpart.connect(Input(inp_key))
    return out

def l2norm_teach(net,inp,ans,a=0.001):
    res = net(**inp)
    #loss = ∑((res[i]-ans[i])^2) / 2
    #∂loss/∂res = (res-ans)
    p = res-ans
    r = net.grad_desc(p,a)
    return res,r

def csis(c,pb=[],ib=[]):
    return "\x1b["+"".join((chr(p+0x30) for p in pb))+"".join((chr(i+0x20) for i in ib))+c
    
def teachloop(net,stuff,times=1000,a=0.001,printmod=1000):
    print(net,"\n\n\n\n\n"+csis('A',[5]))
    for i in range(times):
        if not (i%printmod):
            print("\r"+csis('J')+"i:",i,"/",times,end="")
        for t in stuff:
            if not (i%printmod):
                print("\r"+csis('B')+csis('J'),t[0],end="")
                print("\r"+csis('B')+csis('J')+"expect:",t[1],end="")
            res,prop = l2norm_teach(net,t[0],t[1],a)
            if not (i%printmod):
                print("\r"+csis('B')+csis('J')+"got:",res,end="")
                l = (res-t[1])
                l = np.sum(l*l)
                print("\r"+csis('B')+csis('J')+"loss:",l,csis("A",[5]),end="")
                print("\r"+csis('B'),end="")
    print("\r"+csis('B',[15]))
    
