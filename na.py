from Clockwork_RNN import *
from NNBlocks import *
from audioIn import audioFile
try:
    ct = audioFile("/Users/paul/Music/other/chiptune/Random chiptune mix 26-dPHWrJfV5e4.mp4")
except:
    ct = None


sn = Clockwork_RNN(inputs=2,clock_periods=[1<<i for i in range(8)],subnet_sizes=[[4]]*8)
sn.scramble(0,0.01)

def audio_clip(g):
    for i in g:
        if abs(i) > 2:
            yield 2*i/abs(i)
        else:
            yield i


def train(net,audio=ct,alpha=0.001,vel=0,epochs=1,prnt=False):
    inp = np.zeros(2,dtype=float)
    for e in range(epochs):
        if prnt:
            print(e,"/",epochs)
        for s in audio:
            pred = net(inp)
            inp[0] = s.real
            inp[1] = s.imag
            dloss = pred-inp
            net.grad_desc(dloss,alpha)
            if prnt:
                print("\x1b[Jloss:",.5*np.sum(dloss*dloss),end="\r")
            net.grad_apply(vel)
            yield pred[0]+1j*pred[1],inp
def train_sr(net,audio=ct,alpha=0.001,vel=0,epochs=1,prnt=False):
    inp = np.zeros(net.shape[0],dtype=float)
    for e in range(epochs):
        if prnt:
            print(e,"/",epochs)
        for s in audio:
            pred = net(inp)
            inp[2:] = inp[:-2]
            inp[0] = s.real
            inp[1] = s.imag
            dloss = pred-inp[0:1]
            net.grad_desc(dloss,alpha)
            if prnt:
                print("\x1b[Jloss:",.5*np.sum(dloss*dloss),end="\r")
            net.grad_apply(vel)
            yield pred[0]+1j*pred[1],inp

def bake_sr(netdesc=(32,32,2),a=0.001,v=0.5,tf = tanh):
    sr = NN(netdesc,tanh)
    sr.scramble()
    g = train_sr(sr,ct,a,v)
    rend_to_file(g,"shift net "+str(netdesc)+" "+tf.name+f"a={a},v={v}"+".f32")


from entropy import *
def train_sr_dev(net,dist=dtanhe,audio=ct,alpha=0.001,vel=0,epochs=1,prnt=False):
    inp = np.zeros(net.shape[0],dtype=float)
    for e in range(epochs):
        if prnt:
            print(e,"/",epochs)
        for s in audio:
            pred = net(inp)
            inp[2:] = inp[:-2]
            inp[0] = s.real
            inp[1] = s.imag
            ent0,grad0 = entropy(inp[0],pred[0],pred[2],dist)
            ent1,grad1 = entropy(inp[1],pred[1],pred[3],dist)
            dloss = np.array([grad0[0],grad1[0],grad0[1],grad1[1]])
            net.grad_desc(dloss,alpha)
            net.grad_apply(vel)
            yield pred,ent0,ent1,inp[0],inp[1]
def rend_dev_to_file(gen,path="net run.f32-8"):
    f = open(path,"wb")
    a = np.array([0.]*8,dtype=np.float32)
    s = 0
    for p,e0,e1,i0,i1 in gen:
        a[0:4] = p[:]
        a[4] = e0
        a[5] = e1
        a[6] = i0
        a[7] = i1
        f.write(a.tobytes())
        s += 1
        if s%1000 == 0:
            print(s,end="\r")
    f.close()

def bake_sr_dev(netdesc=(32,32,4),a=0.001,v=0.5,tf = tanh):
    sr = NN(netdesc,tanh)
    sr.scramble()
    g = train_sr_dev(sr,dtanhe,ct,a,v)
    rend_dev_to_file(g,"shift net with uncertainty"+str(netdesc)+" "+tf.name+f"a={a},v={v}"+".f32-8")

def rend_to_file(gen,path="net run 4x8.f32"):
    f = open(path,"wb")
    a = np.array([0.,0,0,0],dtype=np.float32)
    s = 0
    for v,i in gen:
        a[0] = v.real
        a[1] = v.imag
        a[2:3] = i[2:3]
        f.write(a.tobytes())
        s += 1
        if s%1000 == 0:
            print(s,end="\r")
    f.close()


    
def bake():
    rend_to_file(train(sn))
            
def rend(gen,num=48000,arr=None):
    if isNone(arr):
        arr = []
    for v in gen:
        arr += [v]
        num -= 1
        if num == 0:
            break
    return arr


def graph(func):
    from matplotlib import pyplot as plt
    plt.plot([i/100 for i in range(-2000,2000)],[func(inp=i/100)[0] for i in range(-2000,2000)])
    plt.show(block=0)

def graph_gen(g):
    from matplotlib import pyplot as plt
    y = [i for i in g]
    plt.plot([i for i in range(len(y))],y)
    plt.show(block=0)
def graph_gens(g):
    from matplotlib import pyplot as plt
    ys = np.array([i for i in g]).transpose()
    for y in ys:
        plt.plot([i for i in range(len(y))],y)
    plt.show(block=0)


def steach(n,dats,a=0.001):
    for i in range(1000):
        for d in dats:
            _ = n.grad_desc(n(d[0])-d[1],a)
            _ = n.grad_apply()



def play_f32(f):
    while 1:
        b = f.read(512*16)
        for i in range(512):
            l,r,cl,cr = struct.unpack("4f",b[16*i:16*(i+1)])
            yield l+1j*r
def gen_f32(f):
    while 1:
        b = f.read(512*16)
        for i in range(512):
            yield struct.unpack("4f",b[16*i:16*(i+1)])
def play_es_f32(f):
    for l,r,cl,cr in gen_f32(f):
        yield (cl-l)+1j*(cr-r)
def gen_f32_8(f):
    while 1:
        b = f.read(512*32)
        for i in range(512):
            try:
                yield struct.unpack("8f",b[32*i:32*(i+1)])
            except:
                return

import random
import math
def tanhNoise():
    return math.atanh(random.random()*2-1)
def play_f32_8(f,noise=True):
    while 1:
        b = f.read(512*32)
        for i in range(512):
            l,r,vl,vr,el,er,cl,cr = struct.unpack("8f",b[32*i:32*(i+1)])
            n = tanhNoise()*vl+1j*tanhNoise()*vr
            yield l+1j*r+noise*n
def play_es_f32_8(f,noise=True):
    while 1:
        b = f.read(512*32)
        for i in range(512):
            l,r,vl,vr,el,er,cl,cr = struct.unpack("8f",b[32*i:32*(i+1)])
            n = tanhNoise()*vl+1j*tanhNoise()*vr
            yield cl+1j*cr-(l+1j*r+noise*n)


def graph_f32(f):
    g = gen_f32(f)
    ys = [i for i in g]
    from matplotlib import pyplot as plt
    plt.plot(range(len(ys)),ys)
    plt.show(block=0)

MIX=[None]
    
def graph_f32_8(f,n=96000,mix=MIX):
    g = gen_f32_8(f)
    from matplotlib import pyplot as plt
    import matplotlib as mpl
    from matplotlib.widgets import Slider, Button
    plt.subplots_adjust(bottom=0.2)

    mpl.rcParams['agg.path.chunksize'] = 1000000
    fig, ax = plt.subplots()
    fig.suptitle('Compression Details')
    ys = []
    for i in range(n):
        try:
            ys += [next(g)]
        except:
            break
    x = range(len(ys))
    datfile = f
    f = lambda x: x
    
    ax.plot(x,[f(i[4]) for i in ys],label="Entropy L",color=(0.4,0,0),alpha=0.1)
    ax.plot(x,[f(i[5]) for i in ys],label="Entropy R",color=(0,0,0.4),alpha=0.1)
    gl = [entropy(i[6],i[0],i[2])[1] for i in ys]
    gr = [entropy(i[7],i[1],i[3])[1] for i in ys]
    ax.plot(x,[f(i[0]) for i in gl],label="Gradient P L",color=(0.4,0.6,0),alpha=0.1)
    ax.plot(x,[f(i[0]) for i in gr],label="Gradient P R",color=(0,0.6,0.4),alpha=0.1)
    ax.plot(x,[f(i[1]) for i in gl],label="Gradient D L",color=(0.2,0.6,0),alpha=0.1)
    ax.plot(x,[f(i[1]) for i in gr],label="Gradient D R",color=(0,0.6,0.2),alpha=0.1)


    ratio = []
    log_prob = 0
    se = []
    for i in range(len(ys)):
        pl = ys[i][4]
        pr = ys[i][5]
        e = math.log2(pl)+math.log2(pr)
        se += [-e/2]
        log_prob += e
        ratio += [-log_prob/((i+1)*64)]
        
    ax.plot(x,ratio,label="Compression",color=(0,.6,.6),alpha=0.1)
    ax.plot(x,se,label="Sample Entropy",color=(0,1,1),alpha=0.1)


    ax.plot(x,[f(i[0]-i[2]) for i in ys],color=(.8,.4,.0),alpha=.5)
    ax.plot(x,[f(i[0]+i[2]) for i in ys],color=(.8,.4,.0),alpha=.5)
    ax.plot(x,[f(i[1]-i[3]) for i in ys],color=(.0,.4,.8),alpha=.5)
    ax.plot(x,[f(i[1]+i[3]) for i in ys],color=(.0,.4,.8),alpha=.5)
    ax.plot(x,[f(i[0]) for i in ys],label="Predicted L",color=(.8,.4,0))
    ax.plot(x,[f(i[1]) for i in ys],label="Predicted R",color=(0,.4,.8))
    ax.plot(x,[f(i[6]) for i in ys],linestyle=":",label="Signal L",color=(1,0,0))
    ax.plot(x,[f(i[7]) for i in ys],linestyle=":",label="Signal R",color=(0,0,1))    
        
        
    
    ax.set_ylim((-3,3))
    ax.set_xlim((-1,299))

    #https://stackoverflow.com/questions/31001713/plotting-the-data-with-scrollable-x-time-horizontal-axis-on-linux
    axcolor = 'lightgoldenrodyellow'
    axpos = plt.axes([0.2, 0.01, 0.65, 0.03], facecolor=axcolor)
    
    spos = Slider(axpos, 'Pos', -1, len(ys)+1)
    
    def update(val):
        pos = spos.val
        l,r = ax.get_xlim()
        w = r-l
        ax.set_xlim((pos,pos+w))
        fig.canvas.draw_idle()
        
    spos.on_changed(update)

    def play_curry(l,f=datfile,mix=mix):
        def p(event,f=f,mix=mix):
            try:
                f.seek(0)
                mix[0].scale = 1
                mix[0].out = audio_clip(l(event,f))
            except Exception as e:
                print(e)
        return p
    cbs = []
    i = 0
    for b,l in [("Play Error",lambda e,f: play_es_f32_8(f)),
                ("Play Predict",lambda e,f: play_f32_8(f)),
                ("Play Truth",lambda e,f: (i[6]+1j*i[7] for i in gen_f32_8(f))),
                ]:
        playc = play_curry(l)
        axbtn = plt.axes([0.05+i*.1, 0.9, 0.09, 0.05])
        i += 1
        btn = Button(axbtn, b)
        btn.on_clicked(playc)
        cbs += [playc,btn,axbtn]
    
    plt.show(block=0)
    return update,cbs

#new architecture test
def predictor(hist=32,hidden=32,af=sigm):
    mat = Affine(hist,2)
    output = Concat(mat)
    inp = Input("input")
    mat.connect(inp)

    hl = Affine(hist,hidden)
    hl.scramble()
    f = Func(af)
    f.connect(hl)
    hl.connect(inp)
    hlo = Affine(hidden,2)
    hlo.connect(f)
    fo = Func(expo)
    fo.connect(hlo)
    output.connect(fo)

    return output
    

def run_predictor(p,a=0.01,l=32,d=dtanhe,sig=ct):
    inp = np.zeros(l,dtype=float)
    for v in sig:
        pred = p(input=inp)
        inp[:] = np.roll(inp,2)
        inp[0] = v.real
        inp[1] = v.imag
        e0,g0 = entropy(inp[0],pred[0],pred[2],d)
        e1,g1 = entropy(inp[1],pred[1],pred[3],d)
        grad = np.array([g0[0],g1[0],-g0[1],-g1[1]])  #want to ascend
        #print(grad)
        p.grad_desc(grad,a) 
        p.grad_desc_fin()
        yield pred,e0,e1,inp[0],inp[1]
        
class FancyPredictor:
    def __init__(self,hist=32,nd=(32,),rec=16,af=tanh):
        self.affine = np.zeros((2,hist+1),dtype=float)
        self.affine_gd = np.copy(self.affine)
        self.delayline = np.zeros(hist+1,dtype=float)
        self.go = np.zeros(2,dtype=float)
        self.out = np.zeros(4,dtype=float)
        self.error = np.zeros(2,dtype=float)
        if type(af) == ActFunc:
            af = [af]*len(nd)
        self.cnet = NN((2+rec,)+nd+(2+rec+1,),af+[iden],2)
        self.cnet.scramble()
        #zero out the a row, and the var rows
        self.cnet.weights[-1][0][:] = 0
        self.cnet.weights[-1][1][:] = 0
        self.cnet.weights[-1][2][:] = 0
        self.a = 0
        self.rec = rec
    def __call__(self,**kw):
        l,r = kw['input'][0:2]
        self.error[0] = l-self.out[0]
        self.error[1] = r-self.out[1]
        cres = self.cnet(np.concatenate(
            (self.cnet()[3:],self.error)))

        self.a = sigm(cres[0])
        self.out[2:4] = np.exp(cres[1:3])
        
        self.delayline = np.roll(self.delayline,2)
        self.delayline[0] = l
        self.delayline[1] = r
        self.delayline[-1] = 1

        self.go[:] = self.affine_gd@self.delayline
        self.out[0:2] = (self.affine@self.delayline)+self.go*self.a
        return self.out
    def grad_desc(self,prop,a=0.001):
        prop = np.clip(prop,-1,1)
        self.affine += self.a*self.affine_gd
        self.affine_gd = np.clip(-np.outer(prop[0:2],self.delayline),-0.1,0.1)
        vprop = prop[2:4]*self.out[2:4]
        aprop = np.inner(self.go,prop[0:2])*(1-self.a)*self.a
        firstprop = np.concatenate((np.array([aprop]),vprop,np.zeros(self.rec)))
        
        recprop = self.cnet.grad_desc(firstprop,a)
        recprop = np.concatenate((np.zeros(3),recprop[:-2]))
        recprop = self.cnet.grad_desc(recprop,a,1)
        return recprop
    def grad_desc_fin(self,v=0.5):
        self.cnet.grad_apply(v)
    def grad_apply(self,v=0):
        self.cnet.grad_apply(v)        
    def __str__(self):
        return "FancyPredictor"
        


        
def bake_pred(p=None,sig=("chiptune",ct),l=32,h=32,a=0.01,f=tanh,d=dtanh):
    if p == None:
        p = predictor(l,h,f)
    g = run_predictor(p,a,l,d,sig[1])
    path = f"{sig[0]} -> net:{p} l={l},h={h},a={a},f={f.name},d={d.name}.f32-8"
    f = open(path,"wb")
    i = 0
    for p,e0,e1,i0,i1 in g:
        f.write(struct.pack("8f",p[0],p[1],p[2],p[3],e0,e1,i0,i1))
        if i%100 == 0:
            print(i,p[0],p[1],p[2],p[3],e0,e1,i0,i1,"\x1b[J",end="\r")
        i += 1
    f.close()
    return path


def test_predictor(p,sig = ct,a=0.01,vel=0,d=dtanh):
    #print("Making",shape,"sigm net")                                                                                                                                     
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    ax.set_ylim((-3,3))
    tla = [0]*512
    tl, = ax.plot(range(512),tla)
    tra = [0]*512
    tr, = ax.plot(range(512),tra)
    la = [0]*512
    l, = ax.plot(range(512),la)
    ra = [0]*512
    r, = ax.plot(range(512),ra)
    vla = [0]*512
    vl, = ax.plot(range(512),vla,linestyle=":")
    vra = [0]*512
    vr, = ax.plot(range(512),vra,linestyle=":")

    ga = np.zeros((512,4),dtype=float)
    g_1,g_2,g_3,g_4 = ax.plot(range(512),ga,linewidth=0.5)
        
    
    inp = np.zeros(32,dtype=float)
    sig = iter(sig)
    def anim(i):
        for i in range(512):
            v = next(sig)
            pred = p(input=inp)
            la[i],ra[i],vla[i],vra[i] = pred
            inp[:] = np.roll(inp,2)
            inp[0] = v.real
            inp[1] = v.imag
            tla[i] = v.real
            tra[i] = v.imag
            e0,g0 = entropy(inp[0],pred[0],pred[2],d)
            e1,g1 = entropy(inp[1],pred[1],pred[3],d)
            grad = np.array([g0[0],g1[0],-g0[1],-g1[1]])  #want to ascend
            ga[i] = grad
            #print(grad)
            p.grad_desc(grad,a)
            p.grad_desc_fin(vel)
        print(pred[0],pred[1],pred[2],pred[3],e0,e1,inp[0],inp[1],"\x1b[J",end="\r")
        l.set_ydata(la)
        r.set_ydata(ra)
        tl.set_ydata(tla)
        tr.set_ydata(tra)
        vl.set_ydata(vla)
        vr.set_ydata(vra)
        g_1.set_ydata(ga.transpose()[0])
        g_2.set_ydata(ga.transpose()[1])
        g_3.set_ydata(ga.transpose()[2])
        g_4.set_ydata(ga.transpose()[3])
        return l,r,vl,vr,g_1,g_2,g_3,g_4,
    import matplotlib.animation as animation
    ani = animation.FuncAnimation(fig,anim,interval=1)

    plt.show(block=0)
    return ani,anim,fig,ax,p



import os
f32_8_names = [[i for i in os.listdir() if i[-6:] == '.f32-8']]
def refresh():
    f32_8_names[0] = [i for i in os.listdir() if i[-6:] == '.f32-8']
GC_PROTECT = [None]
def show(i,l=96000):
    n = f32_8_names[0][i]
    print("opening",n)
    GC_PROTECT[0] = graph_f32_8(open(n,"rb"),l)


#10 second "songs"
silence = lambda : (0 for i in range(480000))

import math
nsin = lambda x: (math.sin(2*math.pi*x)/2)*(1+1j)
ntri = lambda x: (abs((2*(x%1))-1)-.5)*(1+1j)
nsqr = lambda x: ((x%1 > .5)-.5)*(1+1j)

tone = lambda f=440,w=nsin: (w(i*f/48000) for i in range(480000))
sweep = lambda fl=440,fh=880,w=nsin: (w(i*(fl+(fh-fl)*i/480000)/48000) for i in range(480000))
scale = lambda f=440,w=nsin: (w(i*(f*2**(((12*i)//480000)/12))/48000) for i in range(480000))

exp = lambda t,l=4800: (t>0)*math.exp(-t/l)
xexp = lambda t,l=4800: (t>0)*(t/l)*math.exp(-t/l)

freq = lambda note=59,base=440: base*2**((note-59)/12)

def play_notes(notes,inst=lambda f: (exp(i)*nsin(i*f/48000) for i in range(12000))):
    key = "awsedftgyhujkolp;']"
    for n in notes:
        if n in key:
            f = freq(47+key.index(n))
        else:
            f = 0
        for i in inst(f):
            yield i
        
        
def twang(p,l=48000):
    b = [random.random()*2-1+1j*(2*random.random()-1) for i in range(p)]
    for i in range(l):
        yield b[i%p]
        b[i%p] = (b[i%p]+b[(i+1)%p])/2

def snare(p=2,l=3000):
    m = 1
    for i in range(l):
        yield (random.random()*2-1+1j*(2*random.random()-1))*(1-i/l)

def kick(f=110,l=6000):
    p = 0
    for i in range(l):
        p += (1-i/l)*f/48000
        yield nsin(p)*(1+1j)

def sequencer(track,inst,spb=6000):
    key = "awsedftgyhujkolp;']"
    step = " "
    stop = "."
    oscs = [None]
    for c in track:
        if c == step or c == stop:
            for i in range(spb+(c==stop)*480000):
                a = 0
                p = oscs
                o = oscs[0]
                while o != None:
                    try:
                        a += next(o[1])
                    except StopIteration:
                        p[0] = o[0]
                    finally:
                        p = o
                    o = o[0]
                yield a
                if c == stop and oscs[0] == None:
                    return
        elif c in key:
            oscs[0] = [oscs[0],inst(key.index(c))]
def gen_add(*g):
    l = None
    for i in g:
        l = [l,i]
    l = [l]
    while l[0] != None:
        a = 0
        p = l
        o = l[0]
        while o != None:
            try:
                a += next(o[1])
            except StopIteration:
                p[0] = o[0]
            finally:
                p = o
            o = o[0]
        yield a
def multitrack(tracks):
    for i in gen_add(*(sequencer(t[0],t[1]) for t in tracks)):
        yield i

    
tests = [("silence",silence),("sine",tone),("triangle",lambda : tone(440,ntri)),("square", lambda : tone(440,nsqr)),#4
         ("low sine",lambda : tone(220)),("low tri",lambda : tone(220,ntri)),("low square", lambda : tone(220,nsqr)),#7
         ("sweep",sweep),("tri sweep", lambda : sweep(440,880,ntri)),("sqr sweep",lambda : sweep(440,880,nsqr)),#10
         ("scale",scale),("tri scale", lambda : scale(440,880,ntri)),("sqr scale",lambda : scale(440,880,nsqr)),#13
         ("melody",lambda : play_notes("adgk kgda aegk kgea",lambda f: (nsin(i*f/48000) for i in range(12000)))),
         ("tri exp melody",lambda : play_notes("adgk kgda aegk kgea",lambda f: (exp(i)*ntri(i*f/48000) for i in range(12000)))),
         ("twang",lambda : twang(218,480000)),
         ("twang melody",lambda : play_notes("adgk kgda aegk kgea",lambda f: twang(int(48000/f),12000) if f != 0 else (0 for i in range(12000)))),
         ("song",lambda :
          multitrack([
              ("k  ;  g  ;  j  l  g  l  h  k  d  k  g  j  d  j  f  h  a  h  d  g  a  g  f  h  a  h  g h j k l j g l gad.",lambda k:(exp(i)*ntri(i*freq(k+59)/48000) for i in range(12000))),
              ("ag    d    gl    j    h;    k    dj    g    fk    h    a g   d k   f a h f k h ; k g  j  l  '  ak.",lambda k:twang(int(48000/freq(k+47)),96000)),
              ("k        g        h        d        f        a        f        g        a.",lambda k:twang(int(48000/freq(k+59)),48000*4)),
              ("; y d a a a a a - - - - - - - ;"*4+".",lambda k:snare(k+1)),
              ("a    k    r    r    "*4+".",lambda k:kick(55*2**(k/4))),
              ]))


         ]
         
         


def quickstart(test=0,*a):
    p = FancyPredictor()
    print(tests[test][0])
    path = bake_pred(p,(tests[test][0],tests[test][1]()))
    refresh()
    show(f32_8_names[0].index(path))
    
    
    
