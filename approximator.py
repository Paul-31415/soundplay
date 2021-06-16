import numpy as np
import scipy

from filters import SamplingFilter,s_exp

            
def lpw(f=0,d=.5):
    p = 0
    while 1:
        e1 = 1-p
        e2 = (d-p)%1
        if e1 > e2:
            e1,e2 = e2,e1
        v = (p<d)
        if e1<f:
            if e2<f:
                r = (e2-e1)/f
            else:
                r = e1/f
        else:
            r = 0
        p = (p+f)%1
        yield 1-r if v else r
        
def siw(f=0,d=.5,n=2):
    p = 0
    while 1:
        e1 = 1-p
        e2 = (d-p)%1
        r = 0
        for i in range(n):
            r += nsi((i+e1)/f)+nsi((e1-1-i)/f)-nsi((i+e2)/f)-nsi((e2-1-i)/f)
        v = p<d
        p = (p+f)%1
        yield r+v
    
def sfw(f=0.01,d=.5,s=None):
    if s == None:
        s = s_exp()
    p = 0
    while 1:
        yield s.read()
        if p+f >= 1:
            s.merge_signal([(1,(1-p)/f),(0,(1-p+d)/f)])
            s.step(1)
            p += f-1
        else:
            s.step(1)
            p += f


#scipy has scipy.special.sinc and sici
def nsi(a):
    return scipy.special.sici(a*np.pi)[0]/np.pi

class instrument:
    def __init__(self):
        assert 0
    def params_shape(self):
        return (0,)
    def params_grad(self,p):
        return p
    def params_clamp(self,p):
        return p
    def peek(self,params):
        while 1:
            yield 0
    def do(self,params):
        while 1:
            yield 0
class pulse_waves(instrument):
    def __init__(self,waves=2,prec=3,lowest=1/256,filt=None):
        self.phases = np.zeros(waves,dtype=float)
        self.prec = prec
        self.filt = filt
        self.lowf = lowest
    def params_shape(self):
        return (4,len(self.phases))
    def params_init(self):
        p = np.zeros((4,len(self.phases)),dtype=float)
        p[0,:]=1/64
        p[1,:]=0.5
        return p
    def params_grad(self,p):
        v = np.copy(p)
        v[0,:] = 1
        v[1,:] = 1
        v[2,:] = 1
        v[3,:] = 1
        return v
    def params_clamp(self,p):
        p[0,:] = np.clip(p[0,:],self.lowf,0.5)
        p[1,:] = np.clip(p[1,:],p[0,:],1-p[0,:])
        return p
    def p(self,p,f,d,v):
        #wave dc bias = d
        #instantaneous value is np.sum(((p<d)-d)*v)
        #but bandwidth limited value is different
        #can use scipy.special.sici
        return np.sum(((p<d)-d)*v)
    def peek(self,params):
        for i in self.c(params,np.copy(self.phases),self.filt.copy() if self.filt != None else None):
            yield i
    def do(self,params):
        for i in self.c(params,self.phases,self.filt):
            yield i
    def c(self,params,p,flt):
        while 1:
            if flt == None:
                p += params[0]
                p -= p>=1
                yield self.p(p,params[0],params[1],params[2]+1j*params[3])
            else:
                p += params[0]
                for i in range(len(p)):
                    f = params[0][i]
                    d = params[1][i]
                    m = params[2][i]+1j*params[3][i]
                    if p[i] >= 1:
                        flt.add_signal([((1-d)*m,(p[i]-1)/f)])
                        p[i] -= 1
                    if p[i] >= d:
                        flt.add_signal([(-d*m,(p[i]-d)/f)])
                flt.step(1)
                yield flt.read()
            #yield self.p(p,params[0],params[1],params[2])
import random
def afl(s=256,l=1024):
    r = np.arange(l)/l
    approx_f_loss_mask = np.cos((r-(s/l/2))*np.pi*2)/2+.5
    fr = 1-2*np.abs(r-.5)
    approx_f_loss_weight = fr*np.exp(-fr*8)
    #approx_f_loss_weight[0] = 0
    #approx_f_loss_weight[1] = 0
    #approx_f_loss_weight[-1] = 0
    def approx_f_loss(a,c,p,lm=approx_f_loss_mask,lw=approx_f_loss_weight):
        fa = scipy.fft.fft(a*lm)
        ca = scipy.fft.fft(c*lm)
        return np.sum(((np.abs(fa)-np.abs(ca))**2)*lw)
    return approx_f_loss

def l2l(a,c,p):
    return np.sum(np.abs(a-c)**2)

class approximator:
    def __init__(self,instrument,l=256,step=256,loss=l2l):#afl(256,256)):
        self.instrument = instrument
        self.loss = loss
        self.buf = np.zeros(l,dtype=complex)
        self.pred = None
        self.l = l
        self.s = step
        self.signal = None
        self.outsig = []
        self.params = instrument.params_init()
        self.paramsig = []
        self.ploss = 0
    def get_dat(self,l=None):
        assert self.signal != None
        if l == None:
            l = self.s
        self.buf[:-l] = self.buf[l:]
        self.buf[-l:] = np.fromiter(self.signal,complex,l)
        
    def peek_pred(self,p):
        self.instrument.params_clamp(p)
        return np.fromiter(self.instrument.peek(p),complex,len(self.buf))
    def get_pred(self):
        self.pred = self.peek_pred(self.params)
        return self.pred
    def advance(self):
        self.params = self.instrument.params_clamp(self.params)
        self.paramsig += [np.copy(self.params)]
        self.outsig += [np.fromiter(self.instrument.do(self.params),complex,self.s)]
    def get_loss(self):
        self.ploss = self.loss(self.buf,self.pred,self.params)
    def ind_gradient(self,i,step_size=0.01):
        self.get_loss()
        dparams = np.copy(self.params)
        flat = dparams.ravel()
        d = self.instrument.params_grad(dparams).ravel()[i]*step_size
        flat[i] += d
        poslos = self.loss(self.buf,self.peek_pred(dparams),dparams)
        flat[i] = self.params.ravel()[i] - d
        neglos = self.loss(self.buf,self.peek_pred(dparams),dparams)
        return (poslos-neglos)/2/d,(poslos-2*self.ploss+neglos)/d
    def gradient(self,step_size=0.01):
        d = np.copy(self.params)
        dflat = d.ravel()
        dd = d+0
        ddflat = dd.ravel()
        for i in range(len(dflat)):
            dflat[i],ddflat[i] = self.ind_gradient(i,step_size)
        return d,dd
        
    def step(self,dparams,dlos,ddlos,t=1,a=0.5,m=0.5,eps=0.001):
        dparams += (np.abs(dparams)<eps)*np.clip(ddlos,None,0)*((dparams>=0)*2-1)
        wparams = self.params+dparams*t
        l = self.loss(self.buf,self.peek_pred(wparams),wparams)
        expected = np.inner(dlos.ravel(),(wparams-self.params).ravel())
        i = 0
        while l > expected+self.ploss:
            t *= a
            wparams = self.params+dparams*t
            l = self.loss(self.buf,self.peek_pred(wparams),wparams)
            expected = np.inner(dlos.ravel(),(wparams-self.params).ravel())*m
            i += 1
            if i>16:
                return
        self.params[:] = wparams
        self.ploss = l
        #self.get_loss()
    def refine(self,p=0,t=1,dt=0.001,times=100):
        for time in range(times):
            d,dd = self.gradient(dt)
            self.step(-d+np.random.normal(d)*0.1,d,dd,t)
            if p:
                print(time,self.ploss,end="  \r")
    def stage(self,*a):
        self.get_dat()
        self.get_pred()
        self.get_loss()
        self.refine(*a)
        self.advance()
    def anim(self,times=10,t=1,dt=0.01):
        def anirr(frame,self=self,f=[1],tm=times,dt=dt,t=t):
            if f[0]%tm == 0:
                self.advance()
                self.get_dat()
                self.get_pred()
                self.get_loss()
            f[0] += 1
            self.get_pred()
            d,dd = self.gradient(dt)
            self.step(-d+np.random.normal(d)*0.1,d,dd,t)
            return self.buf.real
        def aniri(frame,self=self):
            return self.buf.imag
        def predr(frame,self=self):
            self.get_pred()
            return self.pred.real
        def predi(frame):
            self.get_pred()
            return self.pred.imag
        def wind(frame,i=[0],tm=times):
            i[0] = (i[0]+1)%tm
            return [1,-1,i[0]/tm*2-1]
        return (anirr,aniri,predr,predi,wind)
            
        
            
        
        
        
        
        
    
        

#closed form l2 pulse wave fitter
def l2pw(buf):
    #returns (phase,period,duty,volume) array
    
    #wave = ((((t/period+phase)%1)<duty)-duty)*volume
    #fourier is sin(πn duty)cos(2πn(phase))

    #loss is
    # ||v - w||^2
    # w = vol*(-duty+


    #csum = np.cumsum(buf)
    pass

    
    

    

def pww(buf,per,duty=0.5,phs=0):
    duty = int(duty*per)
    phs = int(phs*per)
    b = np.concatenate((buf,np.zeros((-len(buf))%per)))
    s = b.reshape((len(b)//per,per))
    ip = np.sum(s[:,phs:duty+phs])+(np.sum(s[:,:duty+phs-per]) if duty+phs>per else 0)
    op = np.sum(s[:,duty+phs:])+(np.sum(s[:,duty+phs-per:phs]) if duty+phs>per else np.sum(s[:,:phs]))
    return ip,op

def pw_component(buf,per,duty=0.5,phs=0):
    ip,op = pww(buf,per,duty,phs)
    e = ip*(1-duty)-op*duty
    tot = (1-duty)*duty*len(buf)
    return e/tot if tot > 0 else 0

#fast? l2 pulse wave fitter
def l2pwf(buf,pcoarse=1,phc=1,dc=1):
    #just go through all the combinations of period (n), duty(period), and phase(period)
    best = 0
    bestw = (0,len(buf),0,0)
    for per in range(2,len(buf)+1,pcoarse):
        b = np.concatenate((buf,np.zeros((-len(buf))%per)))
        s = b.reshape((len(b)//per,per))
        for phs in range(0,per//2,phc):
            for duty in range(0,per,dc):
                ip = np.sum(s[:,phs:duty+phs])+(np.sum(s[:,:duty+phs-per]) if duty+phs>per else 0)
                op = np.sum(s[:,duty+phs:])+(np.sum(s[:,duty+phs-per:phs]) if duty+phs>per else np.sum(s[:,:phs]))
                e = ip*(1-duty/per)-op*duty/per
                if abs(e) > abs(best):
                    best = e
                    bestw = (phs,per,duty,e)

    phs,per,duty,e = bestw
    tot = (((1-duty/per) * duty + (duty/per) * (per-duty)))
    return (phs,per,duty,per/len(buf)*2*e/tot if tot != 0 else 0)


def l2pwf_wb(buf,phs,per,duty,e):
    b = np.zeros(len(buf)+((-len(buf))%per),dtype=complex)
    s = b.reshape((len(b)//per,per))
    b[:] = (1-duty/per)*e
    s[:,duty+phs:] = -duty*e/per
    if duty+phs>per:
        s[:,duty+phs-per:phs] = -duty*e/per
    else:
        s[:,:phs] = -duty*e/per
    buf += b[:len(buf)]
    return buf


def pulseApprox(signal,chunksize=256,*a):
    try:
        while 1:
            buf = np.fromiter(signal,complex,chunksize)
            pw = l2pwf(buf,*a)
            re = l2pwf_wb(np.zeros(chunksize,dtype=complex),*pw)
            for i in range(chunksize):
                yield re[i],buf[i]-re[i]
    except StopIteration:
        pass


    
def pulseRApprox(signal,chunksize=256,times=16,waves=1,pf = lambda p,cs: max(8,int(random.random()*min(2048,cs))),df = lambda p:random.random()*.75+.125, phf = lambda p:random.random()):
    try:
        per = [chunksize//2]*waves
        duty = [0.5]*waves
        phase = [0]*waves
        while 1:
            buf = np.fromiter(signal,complex,chunksize)
            re = np.zeros(chunksize,dtype=complex)
            for w in range(waves):
                e = 0
                for i in range(times):
                    #perpos = [max(2,per//2),per-1,per,per+1,min(chunksize,per*2)]
                    cper = pf(per[w],chunksize)#min(chunksize,max(8,int(random.random()*chunksize)))#(random.random()*.1+.95)*per)))#perpos[int(random.random()*len(perpos))]
                    cduty = df(duty[w])#min(1,max(0,duty+(random.random()-.5)*.5))
                    cphase = phf(phase[w])#(phase+(random.random()-.5)*.125)%1
                
                    ne = pw_component(buf,cper,cduty,cphase)
                    if abs(ne)>abs(e):
                        e = ne
                        per[w] = cper
                        duty[w] = cduty
                        phase[w] = cphase
            
                r = l2pwf_wb(np.zeros(chunksize,dtype=complex),int(phase[w]*per[w]),per[w],int(duty[w]*per[w]),e)
                re += r
                buf -= r
            for i in range(chunksize):
                yield re[i],buf[i]
    except StopIteration:
        pass


                          

        
