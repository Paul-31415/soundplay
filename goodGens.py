import math

class sample:
    def __init__(self,a=[],r=1):
        self.dat = a
        self.rate = r
    def __getitem__(self,i):#no interpolation
        return self.dat[int(i*self.rate)]
    def __len__(self):
        return len(self.dat)/self.rate
class linterpSample(sample):
    def __getitem__(self,i):
        if math.ceil(i*self.rate) < len(self.dat):
            a = i*self.rate-math.floor(i*self.rate)
            return self.dat[math.floor(i*self.rate)]*(1-a)+self.dat[math.ceil(i*self.rate)]*a
        return self.dat[-1]
table_rate = 256
table_size = table_rate**2
sampled_step = sample([0 for i in range(table_size)],table_rate)
for i in range(1,len(sampled_step.dat)):
    sampled_step.dat[i] = sampled_step.dat[i-1]+math.sin(i*math.pi/table_rate)/(i*math.pi/table_rate)/table_rate
def getSampledStep(s):
    if s < 0:
        return -getSampledStep(-s)
    if s >= sampled_step.__len__():
        return 1
    else:
        return sampled_step[s]
sinTable = sample([math.sin(i*math.pi/table_size/2) for i in range(table_size+1)],table_size*4)
def sin2π(x):
    x = x%1
    if x > .5:
        return -sin2π(x-.5)
    if x > .25:
        return sinTable[.5-x]
    return sinTable[x]

    
class square:
    def __init__(self,f,w=.5,p=0,sr=48000):
        self.phase = p
        self.sample_frequency = f/sr
        self.sample_rate = sr
        self.delta = 1
        self.width = w
        self.method_switch = 1/2
    def freq(self,f=None):
        if f == None:
            return self.sample_frequency*self.sample_rate*self.delta
        else:
            self.sample_frequency = f/self.sample_rate/self.delta
    def hi(self,d=1):
        self.sample_frequency *= self.delta/d
        self.delta = d
    def __next__(self):
        p = self.phase
        self.phase = (self.phase + self.sample_frequency*self.delta)%1

        if self.sample_frequency > self.method_switch:
            #forier form (for high frequency waves):
            f = self.sample_frequency
            i = 1
            tot = 0
            while f <= .5:
                tot += 2/i/math.pi*sin2π(i*self.width/2)*sin2π(p)
                i += 1
                f += self.sample_frequency
            return tot
        else:
            #sinc-filter table form
            #phase in samples
            sp = p/self.sample_frequency
            return getSampledStep(sp)\
                  +getSampledStep(self.width/self.sample_frequency-sp)
                  #return -getSampledStep(sp-self.width/self.sample_frequency)
                   #    -getSampledStep(1/self.sample_frequency-sp)
    def __iter__(self):
        while 1:
            yield next(self)


    

