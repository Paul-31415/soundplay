import aifc
import random


            

def monoChannelWrapper(gen):
    for s in gen:
        yield [s]

def saveM(gen,t,name = "save.aiff"):
    saveSample(monoChannelWrapper(extent(gen,t)),name,1,1)
        
def saveSample(gen,name = "out.aiff",channels=2,prec=3,rate=48000,dither=1):
    f = aifc.open(name,'w')
    f.setnchannels(channels)
    f.setsampwidth(prec)
    f.setframerate(rate)
    for s in gen:

        resv = (int(s[i]*(1<<(prec*8 - 2))+random.random()*dither)        for i in range(channels))
        f.writeframes(bytes(((v>>(8*i))%256 for i in range(prec) for v in resv)))
        #print(resv,end='\r')
    f.close()



#some sample generation things

def forceGen(x=0,y=0,vx=-1,vy=1,forceFunc = lambda x,y,vx,vy: (-x,-y), damping = 0.1,dt= 1/96000):
    while 1:
        x += vx*dt
        y += vy*dt
        tmp = forceFunc(x,y,vx,vy)
        vx += tmp[0]*dt
        vy += tmp[1]*dt
        vx *= 1-damping
        vy *= 1-damping
        yield [x,y]

def makeMono(gen,ind = 0):
    for i in gen:
        yield i[ind]



def makeGen(n):
    if type(n) != type((1 for i in ())):
        while 1:
            yield n
    else:
        for i in n:
            yield i

def shift(g,d):
    d = makeGen(d)
    for i in g:
        yield i+next(d)
def step(g,d):
    for i in g:
        d -= 1;
        if d <= 0:
            break
    for i in g:
        yield i
            
def fof(g,l = lambda x: x):
    for i in g:
        yield l(i)
        
def prod(g1,g2):
    g1 = makeGen(g1)
    g2 = makeGen(g2)
    for i in g1:
        yield i*next(g2)
def sum(g1,g2):
    g1 = makeGen(g1)
    g2 = makeGen(g2)
    for i in g1:
        yield i+next(g2)

def integrate(g,k=1,d = 0):
    v =0
    d = makeGen(d)
    k = makeGen(k)
    for i in g:
        v += next(k)*i - next(d)*v
        yield v
        
def derivitive(g):
    prev = 0
    for i in g:
        yield i-prev
        prev = i
                                    
        
def coupledOscilators(xs,ks,k0s,damp=0,dt = 1/96000):
    vs = [0 for i in xs]
    while 1:
        for i in range(len(xs)):
            xs[i] += vs[i]*dt
        for i in range(len(xs)):
            vs[i] = (vs[i]+((xs[i-1]-xs[i])*ks[i]
                            -(xs[i]-xs[(i+1)%len(xs)])*ks[(i+1)%len(xs)]
                            -xs[i]*k0s[i])*dt)*(1-damp)
        yield xs


        
def magnetField(mx,my,mz,mf,g=1):#immitates the magnet pendulum physics
    #https://www.math.hmc.edu/~dyong/math164/2006/win/finalreport.pdf
    def force(x,y,vx,vy):
        fx = -x*g
        fy = -y*g
        for i in range(len(mx)):
            r = ((mx[i]-x)**2+(my[i]-y)**2+mz[i]**2)**(3/2)
            fx += mf[i]*(mx[i]-x)/r
            fy += mf[i]*(my[i]-y)/r
        return (fx,fy)
    return force

def movingMagnetField(mx,my,mz,mf,mov = lambda t,mx,my,mz,mf:(mx,my,mz,mf),dt=1/48000,g=1):#immitates the magnet pendulum physics
    #https://www.math.hmc.edu/~dyong/math164/2006/win/finalreport.pdf
    def force(x,y,vx,vy,t=[0],mx=[mx],my=[my],mz=[mz],mf=[mf]):
        fx = -x*g
        fy = -y*g
        for i in range(len(mx)):
            r = ((mx[0][i]-x)**2+(my[0][i]-y)**2+mz[0][i]**2)**(3/2)
            fx += mf[0][i]*(mx[0][i]-x)/r
            fy += mf[0][i]*(my[0][i]-y)/r
        mx[0],my[0],mz[0],mf[0] = mov(t[0],mx[0],my[0],mz[0],mf[0])
        t[0] += dt
        return (fx,fy)
    return force

def ext(g,n):
    for i in g:
        yield i
        n -= 1
        if n<0:
            return
        
def td(l,dt=1/48000):
    
    def lt(x,y,vx,vy,t=[0]):
        t[0] += dt
        return l(x,y,vx,vy,t[0])
    return lt
              
def lin(s,d=48000):
    s /= d
    x = 0
    while 1:
        yield x
        x += s



def foldMod(v,n):
    return abs((v-n)%(4*n)-2*n)-n

def fm1(v):
    return abs((v-1)%4-2)-1


    
        
