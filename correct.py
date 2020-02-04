import math

def sinc(x,f=1):
    if x != 0:
        return math.sin(x*math.pi*2*f)/(x*math.pi)
    return 1
def si(x,epsilon=1/(1<<24)):
    #want x st minterm < epsilon
    #          e^(3-x) < ep
    #          x     > 3-ln(ep)
    #minterm = math.exp(3-x) #x!/(x**x) ln of minterm can be approximated by 3-x
    if abs(x)<3-math.log(epsilon):
        tot = 0
        term = x
        x2 = x*x
        i = 1
        while abs(term)>epsilon:
            tot += term
            i += 2
            term *= -x2*(i-2)/(i-1)/i/i
        return tot
    else:
        tot = math.pi/2
        x2 = x*x
        cterm = -math.cos(x)/x
        sterm = -math.sin(x)/x2
        i = 1
        while (abs(cterm)+abs(sterm))>epsilon and i < x:
            tot += cterm+sterm
            i += 1
            cterm *= -(i*(i-1))/x2
            sterm *= -(i*(i+1))/x2
            i += 1
        return tot

def nsi(x,e=1/(1<<24)):
    return si(x*math.pi)/math.pi
"""
def 


def windowedSincFilter(g,f,w=1024,window=):
    b = [0]*w
    i = 0
    while 1:
        for
"""


class squareWave:
    def __init__(self,f,dw=.5,sr=48000):
        self.sample_period = sr/f
        self.sample_rate=sr
        self.position = 0.5
        self.duty_width=dw
        self.precision = 1/(1<<24)
        self.delta = 1
        
    def freq(self,f=None):
        if f == None:
            return self.sample_rate/self.sample_period
        else:
            self.sample_period = self.sample_rate/f
        
    def __next__(self):
        #standard: ¯_  
        p = self.position
        self.position = (self.position+self.delta)%self.sample_period
        #wave = sinc•square
        #     = integ(sinc(x-phase)*square(x))
        #     = .5 * for all i: (si(topEnd_i)-si(topStart_i))-(si(bottomEnd_i)-si(bottomStart_i))
        #     so: topEnd_i = bottomStart_i and bottomEnd_i = topStart_(i+1)
        #     = .5 * for all i: si(topEnd_i)-si(topStart_i)-si(topStart_(i+1))+si(topEnd_i)
        #     = for all i: ^(i)-v(i)

        res = 0
        #end when delta < prec,
        # delta < 1/x
        # so when 1/x < prec
        #         1/(i*sp) < prec
        #         i > 1/(prec*sp)
        for i in range(int(4/(self.precision*self.sample_period))):
            res += nsi(i*self.sample_period-p,self.precision)-nsi((i+self.duty_width)*self.sample_period-p,self.precision)+nsi((-1-i)*self.sample_period-p,self.precision)-nsi((-1-i+self.duty_width)*self.sample_period-p,self.precision)
        return res

        
        
    
