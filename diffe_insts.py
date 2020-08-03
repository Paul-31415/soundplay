import random

def randc():
    return random.random()*1j+random.random()

def brand():
    return (randc()-(.5+.5j))*2


def drum1(pos,vel,damping=.01,noise=1,vnoise=1,dt=480/48000):
    while 1:
        pos += (vel+abs(vel)*vnoise*brand())*dt
        vel -= dt*((pos+brand()*noise)*abs(pos) + vel*damping)
        yield pos


def o(l):
    def r(n):
        return l(n.real)+1j*l(n.imag)
    return r
        

def hihat1(pos,vel,damping = .01, mod = .1,modw=1, dt = 1/48000):
    while 1:
        pos += vel*dt
        vel -= dt*(vel*damping+pos+modw*(o(lambda x: x%mod)(pos*abs(pos))))
        yield pos
