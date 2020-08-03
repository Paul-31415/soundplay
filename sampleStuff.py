




class Sample: # as in audio sample, not single measurement
    def __init__(self,samp):
        self.a = samp
    def __getitem__(self,i):
        return self.a[i]
    def __iter__(self):
        return (i for i in self.a)

class Resample2:
    def __init__(self,samp,filt = ):
        self.samplings = {0:samp}
    def res(self,r):
        if r > max(
        
        
        
    def __getitem__(self,i,r=0):
        if r in self.samplings:
            return self.samplings[r][i]
        else:
            self.res(r)
            return self.samplings[r][i]
    def __iter__(self,r=0):
        if r in self.samplings:
            return (i for i in self.samplings[r])
        else:
            self.res(r)
            return (i for i in self.samplings[r])
        
