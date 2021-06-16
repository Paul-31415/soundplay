import numpy as np

def mand(z,c):
    return z*z+c
def burn(z,c):
    az = abs(z.real)+1j*abs(z.imag)
    return az*az+c

def mag2(z):
    return z.real*z.real+z.imag*z.imag

class FracIter:
    def __init__(self,func=mand,bailout=4):
        self.func = func
        self.bailout = bailout
    def plot(self,m=4,b=-2-2j,res=128,it=64):
        its = np.zeros((res,res),dtype=int)
        for x in range(res):
            for y in range(res):
                c = ((x+1j*y)/res)*m+b
                z = 0
                for i in range(it):
                    if mag2(z)>self.bailout:
                        break
                    z = self.func(z,c)
                else:
                    i = -1
                its[y,x] = i
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(its)
        plt.show(block=0)
    def it(self,z,c=None):
        if c is None:
            c = z
        while mag2(z) < self.bailout:
            yield z
            z = self.func(z,c)
    def cit(self,ci,z=0):
        for c in ci:
            z = self.func(z,c)
            if mag2(z) > self.bailout:
                z = 0
            yield z
    def iaplot(self,m=4,b=-2-2j,res=128,it=64):
        #interactive plot
        its = np.zeros((res,res),dtype=int)
        for x in range(res):
            for y in range(res):
                c = ((x+1j*y)/res)*m+b
                z = 0
                for i in range(it):
                    if mag2(z)>self.bailout:
                        break
                    z = self.func(z,c)
                else:
                    i = -1
                its[y,x] = i
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(its)

        c = 0
        z = 0
        p = [z,c]
        
        def onclick(event,p=p,res=res,m=m,b=b):
            c = ((event.xdata+1j*event.ydata)/res)*m+b
            p[1] = c
            #print(c)
            if event.dblclick:
                p[0] = 0 
        
        def it(p=p,ngc = onclick):
            while 1:
                if mag2(p[0])>self.bailout:
                    p[0] = 0
                yield p[0]
                p[0] = self.func(*p)
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        
        plt.show(block=0)
        return it()

        


        
#TODO:
# chaos and hyperchaos
# (eg: lorenz 3d attractor and 4d hyperchaotic version (which should sound like raindrops))
