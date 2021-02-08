

class nogc:
    def __init__(self,*stuff):
        self.stuff = stuff
    def __repr__(self):
        return "nogc(...)"

def graph(func,xm=-10,xM=10,res=1000):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=1, ncols=1)
    t = lambda x: x/(res-1)*(xM-xm)+xm
    ax.plot([t(i) for i in range(res)],[func(t(i)) for i in range(res)])
    plt.show(block=0)

    
def live_graph(dfuncs=()):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    fig, ax = plt.subplots(nrows=1, ncols=1)
    p = [ax.plot(f(-1))[0] for f in dfuncs]
    def update(frame,df=dfuncs,p=p):
        for i in range(len(p)):
            p[i].set_ydata(df[i](frame))
        return p
    ani = FuncAnimation(fig, update)
    plt.show(block=0)
    return plt,nogc(ani,update,p,fig,ax)


def oscope(gen,spf=1024,upsampleTo=4096,ms=.5,a=.1):
    import scipy
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_facecolor("k")
    fig.patch.set_facecolor("xkcd:grey")
    ax.set_ylim((-2,2))
    ax.set_xlim((-2,2))
    buf = [0]*(spf*2)
    outbuf = scipy.signal.resample(buf,upsampleTo*2)[upsampleTo//2:(3*upsampleTo)//2]
    i = spf
    mdat, = plt.plot([i.real for i in outbuf], [i.imag for i in outbuf],"o",color=(0,1,0), ms=ms,alpha=a)
    plt.show(block=False)
    for v in gen:
        yield v
        buf[i] = v
        i = (i+1)%(spf*2)
        if i == 0 or i == spf:
            outbuf = scipy.signal.resample(buf,upsampleTo*2) if upsampleTo != spf else buf
            mdat.set_ydata([outbuf[(j+i)%len(outbuf)].imag for j in range(upsampleTo//2,(3*upsampleTo)//2)])
            mdat.set_xdata([outbuf[(j+i)%len(outbuf)].real for j in range(upsampleTo//2,(3*upsampleTo)//2)])
            fig.canvas.draw_idle()

def upsamp(frame_size=1<<10,to_size=1<<16):
    import numpy as np
    import scipy
    def do(v,i=[0],buf=np.array([0j]*frame_size),o=[np.array([0j]*to_size*2)]):
        buf[i[0]] = v
        i[0] = (i[0]+1)%len(buf)
        if i[0] == 0 or i[0] == frame_size//2:
            ob = scipy.signal.resample(buf,to_size)
            o[0] = np.concatenate((ob,ob))
        return o[0][(i[0]+frame_size//4)*to_size//frame_size:(i[0]+1+frame_size//4)*to_size//frame_size]
    return do
def batchUpsamp(frame_size=1<<10,to_size=1<<16):
    import numpy as np
    import scipy
    def do(v,i=[False],buf=np.array([0j]*frame_size),o=[np.array([0j]*to_size*2)]):
        if i[0]:
            buf[frame_size//2:] = v
        else:
            buf[:frame_size//2] = v
        i[0] = not i[0]
        ob = scipy.signal.resample(buf,to_size)
        o[0] = np.concatenate((ob,ob))
        return o[0][((i[0]*2+1)*frame_size//4)*to_size//frame_size:((i[0]*2+3)*frame_size//4)*to_size//frame_size]
    return do
import numpy as np

def vscope(gen,spf = 1<<10,res=1<<10,d=.6,g=100,ups = 1<<16,colr = np.array([.01,.1,.01],dtype=np.float32)):
    import scipy
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=1, ncols=1)
    #ax.set_facecolor("k")
    fig.patch.set_facecolor("xkcd:grey")
    #ax.set_ylim((-2,2))
    #ax.set_xlim((-2,2))
    fbuf = np.array([[[0,0,0]]*res]*res,dtype=np.float32)
    im = ax.imshow(fbuf)
    plt.show(block=False)
    import math
    ef = math.log(d)
    buf = np.fromiter(gen,complex,spf)
    oldBuf = np.array([0j]*spf)
    fr = np.arange(ups).astype(np.float32)
    gs = np.exp(fr*(-ef/ups))*(g*spf/ups)
    while 1:
        for v in buf:
            yield v
        r = scipy.signal.resample(np.concatenate((oldBuf,buf)),ups*2)[ups//2:3*ups//2]
        x,y = np.clip(((r.real+1)*res/2).astype(int),0,res-1),np.clip(((1-r.imag)*res/2).astype(int),0,res-1)
        fbuf[y,x] += np.outer(gs,colr)
        im.set_data(np.clip(fbuf,0,1))
        fig.canvas.draw_idle()
        fbuf *= d

        oldBuf = buf
        buf = np.fromiter(gen,complex,spf)

def vscope_p(gen,spf = 1<<10,res=1<<10,d=.6,g=100,ups = 1<<16,colr = np.array([.01,.1,.01],dtype=np.float32)):
    import scipy
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=1, ncols=1)
    #ax.set_facecolor("k")
    fig.patch.set_facecolor("xkcd:grey")
    #ax.set_ylim((-2,2))
    #ax.set_xlim((-2,2))
    fbuf = np.array([[[0,0,0]]*res]*res,dtype=np.float32)
    im = ax.imshow(fbuf)
    plt.show(block=False)
    import math
    ef = math.log(d)
    buf = np.fromiter(gen,complex,spf)
    oldBuf = np.array([0j]*spf)
    t = (np.arange(ups)*(res/ups)).astype(int)
    gs = np.ones(ups)*(g*spf/ups)
    while 1:
        for v in buf:
            yield v
        r = scipy.signal.resample(np.concatenate((oldBuf,buf)),ups*2)[ups//2:3*ups//2]
        x,y = np.clip(((r.real+1)*res/4).astype(int),0,res//2-1),np.clip(((1-r.imag)*res/4).astype(int),0,res//2-1)
        fbuf[y+(res//2),t] += np.outer(gs,colr)
        fbuf[x,t] += np.outer(gs,colr)
        im.set_data(np.clip(fbuf,0,1))
        fig.canvas.draw_idle()
        fbuf *= d

        oldBuf = buf
        buf = np.fromiter(gen,complex,spf)

        
def ioscope(gen,spf=1<<11,res=1<<10,d=.6,g=100,resampler = upsamp()):
    import scipy
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(nrows=1, ncols=1)
    #ax.set_facecolor("k")
    fig.patch.set_facecolor("xkcd:grey")
    #ax.set_ylim((-2,2))
    #ax.set_xlim((-2,2))
    fbuf = np.array([[[0,0,0]]*res]*res,dtype=np.float32)
    im = ax.imshow(fbuf)
    plt.show(block=False)
    colr = np.array([.01,.2,.01],dtype=np.float32)
    import math
    ef = math.log(d)
    while 1:
        for i in range(spf):
            vg = next(gen)
            r = resampler(vg)
            x,y = np.clip(((r.real+1)*res/2).astype(int),0,res-1),np.clip(((1-r.imag)*res/2).astype(int),0,res-1)
            fr = np.arange(len(r)).astype(np.float32)
            k = (1-i/spf)*ef
            f = (1-(i + 1/len(r))/spf)*ef - k
            gs = np.exp(fr*f+k)*(g/len(r))
            fbuf[y,x] += np.outer(gs,colr)
            #for ip in range(len(r)):
            #    v = r[ip]
            #    i_f = i + ip/len(r)
            #    x,y = min(res-1,max(0,int((v.real+1)*res/2))),min(res-1,max(0,int((-v.imag+1)*res/2)))
            #    fbuf[y,x] += colr*(g*math.exp((1-i_f/spf)*ef)/len(r))
            yield vg
        #c = clamp(fbuf)
        im.set_data(np.clip(fbuf,0,1))
        fig.canvas.draw_idle()
        fbuf *= d


def design():
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotThings import DraggablePoint
    from matplotlib.widgets import CheckButtons,Textbox
        

#https://stackoverflow.com/questions/21654008/matplotlib-drag-overlapping-points-interactively
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class DraggablePoint:
    lock = None #only one can be animated at a time
    def __init__(self, point,posfunc=lambda x:x,posinv=lambda x: x,animate=False,hooks = [None,None,None]):
        self.point = point
        self.hooks = hooks
        self.press = None
        self.animate=False
        self.background = None
        self.dat = None
        self.map = posfunc
        self.imap = posinv
    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.point.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.point.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.point.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.point.axes: return
        if DraggablePoint.lock is not None: return
        contains, attrd = self.point.contains(event)
        if not contains: return
        self.press = (self.point.center), event.xdata, event.ydata
        DraggablePoint.lock = self

        if self.hooks[0] != None:
            self.hooks[0](self)

        if self.animate:
            # draw everything but the selected rectangle and store the pixel buffer
            canvas = self.point.figure.canvas
            axes = self.point.axes
            self.point.set_animated(True)
            canvas.draw()
            self.background = canvas.copy_from_bbox(self.point.axes.bbox)

            # now redraw just the rectangle
            axes.draw_artist(self.point)

            # and blit just the redrawn area
            canvas.blit(axes.bbox)

    def move(self,x,y):
        self.point.center = self.imap((x,y))
    def pos(self):
        return self.map(self.point.center)

    def on_motion(self, event):
        if DraggablePoint.lock is not self:
            return
        if event.inaxes != self.point.axes: return
        self.point.center, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.point.center = (self.point.center[0]+dx, self.point.center[1]+dy)

        if self.hooks[1] != None:
            self.hooks[1](self)


        if self.animate:
            canvas = self.point.figure.canvas
            axes = self.point.axes
            # restore the background region
            canvas.restore_region(self.background)
            
            # redraw just the current rectangle
            axes.draw_artist(self.point)

            # blit just the redrawn area
            canvas.blit(axes.bbox)


    def on_release(self, event):
        'on release we reset the press data'
        if DraggablePoint.lock is not self:
            return

        self.press = None
        DraggablePoint.lock = None

        if self.hooks[2] != None:
            self.hooks[2](self)

        if self.animate:
            # turn off the rect animation property and reset the background
            self.point.set_animated(False)
            self.background = None
            
            # redraw the full figure
            self.point.figure.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.point.figure.canvas.mpl_disconnect(self.cidpress)
        self.point.figure.canvas.mpl_disconnect(self.cidrelease)
        self.point.figure.canvas.mpl_disconnect(self.cidmotion)



"""        
fig = plt.figure()
ax = fig.add_subplot(111)
drs = []
circles = [patches.Circle((0.32, 0.3), 0.03, fc='r', alpha=0.5),
               patches.Circle((0.3,0.3), 0.03, fc='g', alpha=0.5)]

for circ in circles:
    ax.add_patch(circ)
    dr = DraggablePoint(circ)
    dr.connect()
    drs.append(dr)

plt.show()
"""




    
