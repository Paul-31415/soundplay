import math
import filters
#stereo ready
class string:
    def __init__(self,mass=1,tension=1,stiffness=0,drag = .01,fidelity=8):
        self.mass = mass
        self.tension = tension
        self.stiffness = stiffness
        self.drag = drag
        self.pos = [0 for i in range(fidelity)]
        self.vel = [0 for i in range(fidelity)]

    def tune(self,f):
        #f = √(t/m)/2l
        #t = 2lf**2*m
        self.tension = ((len(self.pos)*2*f)**2)*self.mass

        
    def force(self,i):
        p = (0 if i == 0 else self.pos[i-1])
        c = self.pos[i]
        n = (0 if i == len(self.pos)-1 else self.pos[i+1])
        #string tension and stiffness
        f = self.tension*(c-p+c-n) + self.stiffness*(c-(p+n)/2)*abs(c-(p+n)/2)
        return -f
        
    def step(self,dt):
        accel = [self.force(i)/self.mass for i in range(len(self.pos))]
        for i in range(len(self.pos)):
            self.pos[i] += self.vel[i]*dt
            self.vel[i] += (accel[i]-self.vel[i]*abs(self.vel[i])*self.drag)*dt

    def gen(self,dt):
        while 1:
            yield self.force(0)+self.force(len(self.pos)-1)
            self.step(dt)

def sin2π(v):
    return math.sin(2*math.pi*v)


    
        
def Inst_string(vol=1,sus=.9999,cut=.1,fdistort=.001,hitA=1,pluck=.1/110,harms=4):
    class inst:
        def __init__(self,channel,note,velocity,vals,sr):
            f = 440*2**((note-69)/12)
            a = vol*velocity/128
            def gen():
                #pluck position factor, 1/f is length of string
                pos = pluck*f 
                #forier form of pulse wave
                amps = [2/(i+1)/math.pi*sin2π((i+1)*pos/2) for i in range(harms)]
                #convert to triangle of string initial conditions
                #       ___
                #       | |  ->    _-¯\    = integrate with specific dc offset and scale
                # ______| |     _-¯    \  
                #
                # in fourier space is just integrating and scale cause we dont care of dc
                # scale = 1/pos
                #integrate f -> 1/f * wave
                amps = [a/pos*amps[i]/(i+1) for i in range(harms)]
                #all the coses become sins
                
                #now for the sound
                linear = [math.e**(1j*2*math.pi*f*(i+1)/sr)*(sus**(i+1)) for i in range(harms)]
                #quadratic = [fdistort
                mag = cut+1
                while mag>cut:
                    tot = 0
                    mag = 0
                    for i in range(harms):
                        amps[i] *= linear[i]
                        tot += amps[i]
                        mag += amps[i].real*amps[i].real+amps[i].imag*amps[i].imag
                    yield tot
        
            self.gen = gen()
        def note_off(self,channel,note,velocity):
            self.gen = (0 for i in range(0))#.close()
        def __next__(self):
            return next(self.gen)
    return inst                                                             



def Inst_pullstring(vol=1,sus=.9999,fsus=1,cut=.1,ftime=.1,pluck=.1/110,harms=4):
    class inst:
        def __init__(self,channel,note,velocity,vals,sr):
            f = 440*2**((note-69)/12)
            a = vol*velocity/128
            def gen():
                #pluck position factor, 1/f is length of string
                pos = pluck*f 
                #forier form of pulse wave
                amps = [2/(i+1)/math.pi*sin2π((i+1)*pos/2) for i in range(harms)]
                #convert to triangle of string initial conditions
                #       ___
                #       | |  ->    _-¯\    = integrate with specific dc offset and scale
                # ______| |     _-¯    \  
                #
                # in fourier space is just integrating and scale cause we dont care of dc
                # scale = 1/pos
                #integrate f -> 1/f * wave
                amps = [a/pos*amps[i]/(i+1) for i in range(harms)]
                #all the coses become sins
                
                #now for the sound
                linear = [math.e**(1j*2*math.pi*f*(i+1)/sr)*(sus**((i+1)/sr))*(fsus**((f-440)/sr)) for i in range(harms)]
                #quadratic = [fdistort

                #want freq to follow f-e**(e**(-t)) or maybe f-e**(-(t^2))
                #or i could just exp_interpolate from 1 to linear over ftime
                lerps = [l**(1/sr/ftime) for l in linear]
                lerpvs = [1 for l in linear]
                
                t = 0
                mag = cut+1
                while mag>cut:
                    tot = 0
                    mag = 0
                    if t > 1:
                        for i in range(harms):
                            amps[i] *= linear[i]
                            tot += amps[i]
                            mag += amps[i].real*amps[i].real+amps[i].imag*amps[i].imag
                    else:
                        for i in range(harms):
                            amps[i] *= lerpvs[i]
                            lerpvs[i] *= lerps[i]
                            tot += amps[i]*t
                            mag += amps[i].real*amps[i].real+amps[i].imag*amps[i].imag
                        t += 1/sr/ftime
                        
                    yield tot
        
            self.gen = gen()
        def note_off(self,channel,note,velocity):
            self.gen = (0 for i in range(0))#.close()
        def __next__(self):
            return next(self.gen)
    return inst                                                             












def pullstring(f,r=1,p=.5):
    #mimics holding a loose string at a pos lightly and then rapidly tensioning it.
    #    .             .
    #.__/ \__. -> ._________.
    # 3:07 Foria - Break Away [NCS Release]
    
    pass
