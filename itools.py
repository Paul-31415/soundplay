


def time(start=0,rate=1/48000):
    while 1:
        yield start
        start += rate

def const(v):
    while 1:
        yield v
def it(v):
    try:
        return v.__iterWr__()
    except:
        try:
            return interWr(iter(v))
        except:
            return iterWr(v)

def seq(*gens):
    for g in gens:
        try:
            for v in g:
                yield v
        except RuntimeError:
            pass
    
        
class indexWr:
    def __init__(self,a):
        self.a = a
        self.round = True
        self.loop = True
        self.rate = 1
    def __getitem__(self,i):
        if self.loop:
            i = i%self.a.__len__()
        if self.round:
            return self.a[int(round(i))]
        else:
            return self.a[i]
    def __iter__(self):
        i = 0
        while i < self.a.__len__():
            yield self[i]
            i += self.rate
            if self.loop:
                i = i%self.a.__len__()
                


def tseq(iters,times):
    i = 0
    t = 0
    activeIter = it((0 for i in range(times[-1])))
    while i < len(iters):
        while times[i] <= t:
            activeIter += it(iters[i])
            i += 1
        yield next(activeIter)

        t += 1
    for i in activeIter:
        yield i

                



                

class iterWr:
    def __init__(self,it):
        if type(it) != type((i for i in range(0))):
            try: self.it = it.__iter__()
            except:
                self.it = const(it)
        else:
            self.it = it
    def __iterWr__(self):
        return self
    
    def __add__(self,o):
        o = it(o)
        return iterWr(seq((i+next(o) for i in self),self,o))
    def __radd__(self,o):
        o = it(o)
        return iterWr(seq((next(o)+i for i in self),self,o))
    def __sub__(self,o):
        o = it(o)
        return iterWr(seq((i-next(o) for i in self),self,o))
    def __rsub__(self,o):
        o = it(o)
        return iterWr(seq((next(o)-i for i in self),self,o))
    def __mul__(self,o):
        o = it(o)
        return iterWr((i*next(o) for i in self))
    def __rmul__(self,o):
        o = it(o)
        return iterWr((next(o)*i for i in self))
    def __div__(self,o):
        o = it(o)
        return iterWr((i/next(o) for i in self))
    def __rdiv__(self,o):
        o = it(o)
        return iterWr((next(o)/i for i in self))
    def __truediv__(self,o):
        o = it(o)
        return iterWr((i/next(o) for i in self))
    def __rtruediv__(self,o):
        o = it(o)
        return iterWr((next(o)/i for i in self))
    def __floordiv__(self,o):
        o = it(o)
        return iterWr((i//next(o) for i in self))
    def __rfloordiv__(self,o):
        o = it(o)
        return iterWr((next(o)//i for i in self))
    def __mod__(self,o):
        o = it(o)
        return iterWr((i%next(o) for i in self))
    def __rmod__(self,o):
        o = it(o)
        return iterWr((next(o)%i for i in self))
    def __lshift__(self,o):
        o = it(o)
        return iterWr((i<<next(o) for i in self))
    def __rlshift__(self,o):
        o = it(o)
        return iterWr((next(o)<<i for i in self))
    def __rshift__(self,o):
        o = it(o)
        return iterWr((i>>next(o) for i in self))
    def __rrshift__(self,o):
        o = it(o)
        return iterWr((next(o)>>i for i in self))
    def __pow__(self,o):
        o = it(o)
        return iterWr((i**next(o) for i in self))
    def __rpow__(self,o):
        o = it(o)
        return iterWr((next(o)**i for i in self))
    def __divmod__(self,o):
        o = it(o)
        return iterWr((divmod(i,next(o)) for i in self))
    def __rdivmod__(self,o):
        o = it(o)
        return iterWr((divmod(next(o),i) for i in self))
    def __or__(self,o):
        o = it(o)
        return iterWr(seq((i|next(o) for i in self),self,o))
    def __ror__(self,o):
        o = it(o)
        return iterWr(seq((next(o)|i for i in self),self,o))
    def __and__(self,o):
        o = it(o)
        return iterWr((i&next(o) for i in self))
    def __rand__(self,o):
        o = it(o)
        return iterWr((next(o)&i for i in self))
    def __xor__(self,o):
        o = it(o)
        return iterWr(seq((i^next(o) for i in self),self,o))
    def __rxor__(self,o):
        o = it(o)
        return iterWr(seq((next(o)^i for i in self),self,o))
    def __call__(self,o):
        o = it(o)
        return iterWr((i(next(o)) for i in self))
    def __eq__(self,o):
        o = it(o)
        return iterWr((i==next(o) for i in self))
    def __ne__(self,o):
        o = it(o)
        return iterWr((i!=next(o) for i in self))
    def __le__(self,o):
        o = it(o)
        return iterWr((i<=next(o) for i in self))
    def __lt__(self,o):
        o = it(o)
        return iterWr((i<next(o) for i in self))
    def __ge__(self,o):
        o = it(o)
        return iterWr((i>=next(o) for i in self))
    def __gt__(self,o):
        o = it(o)
        return iterWr((i>next(o) for i in self))
    def __abs__(self):
        return iterWr((abs(i) for i in self))
    def __pos__(self):
        return iterWr((+i for i in self))
    def __neg__(self):
        return iterWr((-i for i in self))
    def __bool__(self):
        return iterWr((bool(i) for i in self))
    def __int__(self):
        return iterWr((int(i) for i in self))
    def cint(self):
        return iterWr((int(i.real)+1j*int(i.imag) for i in self))
    def __float__(self):
        return iterWr((float(i) for i in self))
    def real(self):
        return iterWr((i.real for i in self))
    def imag(self):
        return iterWr((i.imag for i in self))
    def l(self,f):
        return iterWr((f(i) for i in self))
    def cToT(self):
        return iterWr(((i.real,i.imag) for i in self))
    def s(self,r=1,t=0,o=0):
        def it(t):
            v = next(self)
            while 1:
                yield v
                t += r
                while t>1:
                    v = next(self)
                    t -= 1
        return iterWr(it(t))
    def e(self,n):
        return iterWr((next(self) for i in range(n)))
    def __next__(self):
        return next(self.it)
    def __iter__(self):
        try:
            for i in self.it:
                yield i
        except RuntimeError:
            raise StopIteration
    def then(self,o):
        o = it(o)
        def f():
            for i in self.it:
                yield i
            for i in o.it:
                yield i
        return it(f())
    def index(self,o):
        return it((o[i] for i in self))
    def cindex(self,o):
        return it((o[i.real].real+1j*o[i.imag].imag for i in self))


def lm(v):
    try:
        return v.__lmbdWr__()
    except:
        if type(v) == type(it):
            return lmbdWr(v)
        else:
            return lmbdWr(lambda *args:v)

def linterp(l,stepSize=1):
    lrnd = l.sal(lambda a: int(a/stepSize)*stepSize)
    return lm(lambda t: (lambda a:lrnd(t)*(1-a) + a*lrnd(t+stepSize))((t%stepSize)/stepSize))

def tquant(l,stepSize=1):
    return l.sal(lambda a: int(a/stepSize)*stepSize)

class lmbdWr:
    def __init__(self,f):
        self.f = f
    def __call__(self,*args):
        return self.f(*args)

    def __lmbdWr__(self):
        return self
    def __iterWr__(self):
        return iterWr(iter(self))
    def __iter__(self,ts=0,tr=1/48000):
        return it((self(i) for i in time(ts,tr)))
    
    def __add__(self,o):
        o = lm(o)
        return lmbdWr(lambda *a: self(*a)+o(*a))
    def __radd__(self,o):
        o = lm(o)
        return lmbdWr(lambda *a:o(*a)+self(*a) )
    def __sub__(self,o):
        o = lm(o)
        return lmbdWr(lambda *a:self(*a)-o(*a) )
    def __rsub__(self,o):
        o = lm(o)
        return lmbdWr(lambda *a:o(*a)-self(*a) )
    def __mul__(self,o):
        o = lm(o)
        return lmbdWr(lambda *a:self(*a)*o(*a) )
    def __rmul__(self,o):
        o = lm(o)
        return lmbdWr(lambda *a:o(*a)*self(*a) )
    def __div__(self,o):
        o = lm(o)
        return lmbdWr(lambda *a:self(*a)/o(*a) )
    def __rdiv__(self,o):
        o = lm(o)
        return lmbdWr(lambda *a:o(*a)/self(*a) )
    def __truediv__(self,o):
        o = lm(o)
        return lmbdWr(lambda *a:self(*a)/o(*a) )
    def __rtruediv__(self,o):
        o = lm(o)
        return lmbdWr(lambda *a:o(*a)/self(*a) )
    def __floordiv__(self,o):
        o = lm(o)
        return lmbdWr(lambda *a:self(*a)//o(*a) )
    def __rfloordiv__(self,o):
        o = lm(o)
        return lmbdWr(lambda *a:o(*a)//self(*a) )
    def __mod__(self,o):
        o = lm(o)
        return lmbdWr(lambda *a:self(*a)%o(*a) )
    def __rmod__(self,o):
        o = lm(o)
        return lmbdWr(lambda *a:o(*a)%self(*a) )
    def __lshift__(self,o):
        o = lm(o)
        return lmbdWr(lambda *a:self(*a)<<o(*a) )
    def __rlshift__(self,o):
        o = lm(o)
        return lmbdWr(lambda *a:o(*a)<<self(*a) )
    def __rshift__(self,o):
        o = lm(o)
        return lmbdWr(lambda *a:self(*a)>>o(*a) )
    def __rrshift__(self,o):
        o = lm(o)
        return lmbdWr(lambda *a:o(*a)>>self(*a) )
    def __pow__(self,o):
        o = lm(o)
        return lmbdWr(lambda *a:self(*a)**o(*a) )
    def __rpow__(self,o):
        o = lm(o)
        return lmbdWr(lambda *a:o(*a)**self(*a) )
    def __divmod__(self,o):
        o = lm(o)
        return lmbdWr(lambda *a:divmod(self(*a),o(*a)) )
    def __rdivmod__(self,o):
        o = lm(o)
        return lmbdWr(lambda *a:divmod(o(*a),self(*a)) )
    def __or__(self,o):
        o = lm(o)
        return lmbdWr(lambda *a:self(*a)|o(*a) )
    def __ror__(self,o):
        o = lm(o)
        return lmbdWr(lambda *a:o(*a)|self(*a) )
    def __and__(self,o):
        o = lm(o)
        return lmbdWr(lambda *a:self(*a)&o(*a) )
    def __rand__(self,o):
        o = lm(o)
        return lmbdWr(lambda *a:o(*a)&self(*a) )
    def __xor__(self,o):
        o = lm(o)
        return lmbdWr(lambda *a:self(*a)^o(*a) )
    def __rxor__(self,o):
        o = lm(o)
        return lmbdWr(lambda *a:o(*a)^self(*a) )
    def call(self,o):
        o = lm(o)
        return lmbdWr(lambda *a:self(*a)(o(*a)) )
    def __index__(self,o):
        o = lm(o)
        return lmbdWr(lambda *a:self(*a)[o(*a)] )
    def __eq__(self,o):
        o = lm(o)
        return lmbdWr(lambda *a:self(*a)==o(*a) )
    def __ne__(self,o):
        o = lm(o)
        return lmbdWr(lambda *a:self(*a)!=o(*a) )
    def __le__(self,o):
        o = lm(o)
        return lmbdWr(lambda *a:self(*a)<=o(*a) )
    def __lt__(self,o):
        o = lm(o)
        return lmbdWr(lambda *a:self(*a)<o(*a) )
    def __ge__(self,o):
        o = lm(o)
        return lmbdWr(lambda *a:self(*a)>=o(*a) )
    def __gt__(self,o):
        o = lm(o)
        return lmbdWr(lambda *a:self(*a)>o(*a) )
    def __abs__(self):
        return lmbdWr(lambda *a:abs(self(*a)) )
    def __pos__(self):
        return lmbdWr(lambda *a:+self(*a) )
    def __neg__(self):
        return lmbdWr(lambda *a:-self(*a) )
    def __bool__(self):
        return lmbdWr(lambda *a:bool(self(*a)) )
    def __int__(self):
        return lmbdWr(lambda *a:int(self(*a)) )
    def __float__(self):
        return lmbdWr(lambda *a:float(self(*a)) )
    def real(self):
        return lmbdWr(lambda *a:self(*a).real )
    def imag(self):
        return lmbdWr(lambda *a:self(*a).imag )
    def cToT(self):
        return lmbdWr(lambda *a:(self(*a).real,self(*a).imag) )
    def l(self,f):
        return lmbdWr(lambda *a:f(self(*a)) )
    def al(self,f):
        return lmbdWr(lambda *a:self(f(*a)) )
    def sal(self,f):
        return lmbdWr(lambda *a:self(*((f(a[0]),)+a[1:])))
    def c(self,o):
        return lmbdWr(lambda *a:self(o(*a)(*a)) )
    def s(self,s=0,r=1):
        return lmbdWr(lambda *a:self(*((a[0]*r+s,)+a[1:])))

