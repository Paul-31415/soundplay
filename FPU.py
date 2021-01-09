


import numpy as np

def fpudl(initial=32,damp=0.9,a=0,b=0,dt=0.001,show=True):
    if type(initial) == int:
        initial = np.zeros(initial)
    if type(initial) == list:
        initial = np.array(initial)

    x = initial
    v = np.zeros(x.shape)
    if show:
        import matplotlib.animation as animation    
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
        xp, = ax.plot(range(len(x)),x,'o')
        vp, = ax.plot(range(len(x)),v,'o')
        ax.set_ylim(-1.3,1.3)

    crt = 2**(1/3)
    den = (2*(2-crt))
    c = [1/den,(1-crt)/den,(1-crt)/den,1/den]
    d = [2/den,-2*crt/den,2/den,0]

    if show:
        def anim(i,x=x,v=v):
            xp.set_ydata(x)
            vp.set_ydata(v)
            return xp,vp,
        ani = animation.FuncAnimation(fig,anim,interval=1)
        dgc = (ani,anim)
    else:
        dgc = None
    if show:
        plt.show(block=0)
    def fpu_delay_line(inp,dt=dt,damp=damp,x=x,v=v,a=a,b=b,c=c,d=d,dgc=dgc):
        #step with symplectic integration
        def accel():
            shifted_p = np.roll(x,1)
            shifted_p[0] = inp.real
            shifted_n = np.roll(x,-1)
            shifted_n[-1] = 0
            d_p = shifted_p - x
            d_n = shifted_n - x
            sq = [d_p*d_p,d_n*d_n]
            return d_p+d_n + a*(sq[0]-sq[1]) +\
                b*(sq[0]*d_p+sq[1]*d_n)
        #now can velocity verlet
        for i in range(len(c)):
            x += v*dt*c[i]
            v += accel()*dt*d[i]
        v[-1] *= damp
        out = (lambda n: n+a*n*n+b*n*n*n)(x[-1])
        return out
        
    return fpu_delay_line
def fpu(initial=32,h=1,a=0,b=0,dt=0.001,simrate=0.001,show=True):
    if type(initial) == int:
        initial = np.sin(h*(np.arange(initial)+1)/(initial+1) * np.pi)
    if type(initial) == list:
        initial = np.array(initial)

    x = initial
    v = np.zeros(x.shape)
    if show:
        import matplotlib.animation as animation    
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
        xp, = ax.plot(range(len(x)),x,'o')
        vp, = ax.plot(range(len(x)),v,'o')
        ax.set_ylim(-1.3,1.3)

    crt = 2**(1/3)
    den = (2*(2-crt))
    c = [1/den,(1-crt)/den,(1-crt)/den,1/den]
    d = [2/den,-2*crt/den,2/den,0]
    def update(dt=dt,x=x,v=v,a=a,b=b,c=c,d=d):
        #step with symplectic integration
        def accel():
            shifted_p = np.roll(x,1)
            shifted_p[0] = 0
            shifted_n = np.roll(x,-1)
            shifted_n[-1] = 0
            d_p = shifted_p - x
            d_n = shifted_n - x
            sq = [d_p*d_p,d_n*d_n]
            return d_p+d_n + a*(sq[0]-sq[1]) +\
                b*(sq[0]*d_p+sq[1]*d_n)
        #now can velocity verlet
        for i in range(len(c)):
            x += v*dt*c[i]
            v += accel()*dt*d[i]
    if show:
        def anim(i,x=x,v=v):
            xp.set_ydata(x)
            vp.set_ydata(v)
            return xp,vp,
        ani = animation.FuncAnimation(fig,anim,interval=1)
        dgc = (ani,anim)
    else:
        dgc = None
    def gen(dontgc = dgc,simrate=simrate,dt=dt):
        time = 0
        while 1:
            yield 0
            for i in range(len(x)):
                yield x[i]+1j*v[i]
            time += simrate/2
            while time > dt:
                update()
                time -= dt
            yield 0
            for i in range(len(x)-1,-1,-1):
                yield -x[i]-1j*v[i]
            time += simrate/2
            while time > dt:
                update()
                time -= dt
    if show:
        plt.show(block=0)
    return gen()

def fpuv(initial=32,h=1,a=0,b=0,dt=0.001):
    if type(initial) == int:
        initial = np.sin(h*(np.arange(initial)+1)/(initial+1) * np.pi)
    if type(initial) == list:
        initial = np.array(initial)

    x = initial
    v = np.zeros(x.shape)
    import matplotlib.animation as animation    
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    xp, = ax.plot(range(len(x)),x,'o')
    vp, = ax.plot(range(len(x)),v,'o')
    ax.set_ylim(-1.3,1.3)

    def anim(i,x=x,v=v,dt=dt,a=a,b=b):
        #step with velocity verlet
        def accel():
            shifted_p = np.roll(x,1)
            shifted_p[0] = 0
            shifted_n = np.roll(x,-1)
            shifted_n[-1] = 0
            d_p = shifted_p - x
            d_n = shifted_n - x
            sq = [d_p*d_p,d_n*d_n]
            return d_p+d_n + a*(sq[0]-sq[1]) +\
                b*(sq[0]*d_p+sq[1]*d_n)
        #now can velocity verlet
        a = accel()
        x += v*dt+.5*dt*dt*a
        v += .5*(a+accel())
        
        xp.set_ydata(x)
        vp.set_ydata(v)
        return xp,vp,
        
    ani = animation.FuncAnimation(fig,anim,interval=1)
    plt.show(block=0)
    return ani,anim
