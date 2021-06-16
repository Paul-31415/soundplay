
import sklearn.neighbors as sk
import scipy as sp
import numpy as np
#take a template of ffts
def make_template(song,size=8192,space = np.abs):
    mask = 1-np.cos(2*np.pi*np.arange(size)/size)/2
    buf = np.zeros(size,dtype=complex)
    i = 0
    for v in song:
        buf[i] = v
        i += 1
        if i == size:
            masked = buf*mask
            buf[:size//2] = buf[size//2:]
            i = size//2

            dat = space(sp.fft.fft(masked))
            yield (dat,masked)

def dupneg_temp(g):
    for v in g:
        yield v
        yield (-v[0],-v[1])
            
def make_template_pts(song,size=8192):
    mask = 1-np.cos(2*np.pi*np.arange(size)/size)/2
    buf = np.zeros(size,dtype=complex)
    i = 0
    for v in song:
        buf[i] = v
        i += 1
        if i == size:
            masked = buf*mask
            buf[:size//2] = buf[size//2:]
            i = size//2
            if np.sqrt(np.sum(masked.real**2)+np.sum(masked.imag**2)) != 0:
                yield masked
            

def decomp_nearest(dists,inds,kn,*a):
    mat = np.linalg.pinv(kn.points[inds[0]].T)
    rmat = kn.payloads[inds[0]].T
    return rmat@(mat@pt)
            
class knn_tf:
    def __init__(self,dset,size=8192,space=np.abs,neighbors=1,combine=lambda d,i,s,o,p:s.payloads[i[0][0]]):
        self.space = space
        self.size = size
        if not (dset is None):
            dset = [i for i in dset]
            self.points = np.array([i[0] for i in dset])
            self.payloads = np.array([i[1] for i in dset])
            self.model = sk.NearestNeighbors(n_neighbors=neighbors,algorithm="auto").fit(self.points)
        self.combine = combine
    def copydat(self,o):
        self.points = o.points
        self.payloads = o.payloads
        self.model = o.model
        self.size = o.size
    def setspace(self,s,neighbors=1):
        self.space = s
        self.points = np.array([self.space(sp.fft.fft(v)) for v in self.payloads])
        self.model = sk.NearestNeighbors(n_neighbors=neighbors,algorithm="auto").fit(self.points)                    
    def regen(self,neighbors):
        self.model.set_params(n_neighbors=neighbors)
    def lookup(self,tv,v):
        dists,inds = self.model.kneighbors(tv.reshape(1, -1))
        return self.combine(dists,inds,self,tv,v)

    def gen(self,g,times=1,error_prop_a=0,error_prop_b=0):
        size = self.size
        mask = 1-np.cos(2*np.pi*np.arange(size)/size)/2
        buf = np.zeros(size,dtype=complex)
        i = 0
        res = np.zeros(size,dtype=complex)
        error = self.points[0]*0
        for v in g:
            buf[i] = v
            i += 1
            if i == size:
                masked = buf*mask
                buf[:size//2] = buf[size//2:]
                i = size//2
                res[:size//2] = res[size//2:]
                res[size//2:] = 0
                
                for t in range(times):
                    dat = self.space(sp.fft.fft(masked))
                    p = self.lookup(dat+error,masked)
                    error *= error_prop_a
                    error += (dat - self.space(sp.fft.fft(p)))*error_prop_b
                    res += p
                    masked -= p
            yield res[i-(size//2)]


class knnn_tf:
    def __init__(self,dset,size=8192,space=lambda x:np.abs(sp.fft.fft(x))):
        self.space = space
        self.size = size
        if not (dset is None):
            try:
                dset = [i for i in dset]
                self.points = np.array([i[0] for i in dset])
                self.payloads = np.array([i[1] for i in dset])
                mags = np.sqrt((self.points**2).sum(-1)).reshape(-1,1)
                self.points /= mags
                self.payloads /= mags
                self.model = sk.NearestNeighbors(n_neighbors=1,algorithm="auto").fit(self.points)
            except:
                self.space = dset.space
                self.copydat(dset)
    def copydat(self,o):
        self.points = o.points
        self.payloads = o.payloads
        self.model = o.model
        self.size = o.size
    def setdat(self,d):
        self.payloads = list(d)
        self.setspace(self.space)
    def setspace(self,s):
        self.space = s
        self.points = np.array([self.space(v) for v in self.payloads])
        mags = np.sqrt((self.points**2).sum(-1)).reshape(-1,1)
        self.points /= mags
        self.payloads /= mags
        self.model = sk.NearestNeighbors(n_neighbors=1,algorithm="auto").fit(self.points)
    def lookup(self,tv):
        dists,inds = self.model.kneighbors(tv.reshape(1, -1))
        return inds[0][0]

    def gen(self,g,times=1,error_prop_a=0,error_prop_b=0):
        size = self.size
        mask = 1-np.cos(2*np.pi*np.arange(size)/size)/2
        buf = np.zeros(size,dtype=complex)
        i = 0
        res = np.zeros(size,dtype=complex)
        error = self.points[0]*0
        for v in g:
            buf[i] = v
            i += 1
            if i == size:
                masked = buf*mask
                buf[:size//2] = buf[size//2:]
                i = size//2
                res[:size//2] = res[size//2:]
                res[size//2:] = 0

                error *= error_prop_a
                goal = self.space(masked)
                resid = goal+error
                for t in range(times):
                    ind = self.lookup(resid/np.sqrt(np.sum(resid**2)))
                    r = self.points[ind]
                    p = self.payloads[ind]
                    a = np.dot(resid,r)
                    res += p*a
                    resid -= r*a
                error += (resid - goal)*error_prop_b
            yield res[i-(size//2)]
    def batch(self,g,times=1,pr=1):
        if pr:
            print("loading data",end="\r")
        if type(g) is not np.ndarray:
            dat = np.array(list(g))
        else:
            dat = g
        if pr:
            print("loaded data ",end="\r")
        size = self.size
        mask = 1-np.cos(2*np.pi*np.arange(size)/size)/2
        rows = len(dat)//size
        rows += rows-1
        if pr:
            print(f"rows:{rows}, applying mask        ",end="\r")
        mdat = np.zeros((rows,size),dtype=complex)
        mdat[::2] = dat[:len(mdat[::2])*size].reshape(-1,size,order='C')
        mdat[1::2] = dat[size//2:size//2+len(mdat[1::2])*size].reshape(-1,size,order='C')
        mdat *= mask
        if pr:
            print(f"applying space                    ",end="\r")
        resid = self.space(mdat.T).T
        res = np.zeros(mdat.shape,dtype=complex)
        for i in range(times):
            if pr:
                print(f"main loop:{i}   norming                ",end="\r")
            #normalize
            mags = np.sqrt(np.sum(resid**2,axis=1).reshape(-1,1))
            mags[mags==0]=1
            if pr:
                print(f"main loop:{i}   knning              ",end="\r")
            dists,inds = self.model.kneighbors(resid/mags)
            if pr:
                print(f"main loop:{i}   accumulating         ",end="\r")
            rs = self.points[inds[:,0]]
            ps = self.payloads[inds[:,0]]
            ayys = np.einsum("ij,ij->i",rs,resid).reshape(-1,1)
            res += ps*ayys
            resid -= rs*ayys
        ores = np.zeros((len(res)+1)*size//2,dtype=complex)
        p1 = res[::2].ravel()
        p2 = res[1::2].ravel()
        ores[:len(p1)] += p1
        ores[size//2:len(p2)+size//2] += p2
        if pr:
            print("                                      ",end="\r")
        return ores
        
        
            
def pitch_bucket_matrix(size=4096,factor=2**(1/24),low=4):
    a = [size//2,(size//2)/factor]
    while a[-2]-a[-1] > low:
        a += [a[-1]/factor]
    mat = tri_interps(a[::-1],size//2)
    return np.concatenate((mat,mat[:,::-1]),axis=1)
    
def tri_interps(a,size=4096):
    mat = []
    r = np.arange(size)
    def tri(l,m,h):
        t = (r<m)*((r-l)/(m-l))+(r>=m)*((r-h)/(m-h))
        return (t>0)*t
    mat = [(r<a[0])+(r>=a[0])*((r-a[1])/(a[0]-a[1]))]
    mat[0] *= (mat[0]>0)
    for i in range(1,len(a)-1):
        mat += [tri(a[i-1],a[i],a[i+1])]
    mat += [(r>=a[-1])+(r<a[-1])*((r-a[-2])/(a[-1]-a[-2]))]
    mat[-1] *= (mat[-1]>0)
    return np.array(mat)
def tri(l,m,h,s):
    r = np.arange(s)
    t = (r<m)*((r-l)/(m-l))+(r>=m)*((r-h)/(m-h))
    return (t>0)*t



    

            
def mipmap_nearest(g,n=12,factor=2**(1/12),f=1):
    buf = []
    for v in g:
        buf += [v]
    for i in range(n):
        x = 0
        while x < len(buf):
            yield buf[int(x)]
            x += f
        f *= factor

    
class submanifold:
    def __init__(self,basis):
        self.basis = list(basis)
        self.matrix = np.array([i[0] for i in self.basis]).T
        self.pinv = np.linalg.pinv(self.matrix)
    def decompose(self,v):
        return self.pinv@v
    
    def gen(self,g):
        size = len(self.basis[0][0])
        mask = 1-np.cos(2*np.pi*np.arange(size)/size)/2
        buf = np.zeros(size,dtype=complex)
        i = 0
        res = np.zeros(size,dtype=complex)
        for v in g:
            buf[i] = v
            i += 1
            if i == size:
                masked = buf*mask
                buf[:size//2] = buf[size//2:]
                i = size//2
                res[:size//2] = res[size//2:]
                res[size//2:] = 0
                
                dat = np.abs(sp.fft.fft(masked))
                p = self.decompose(dat)
                for i in range(len(p)):
                    res += p[i]*self.basis[i][1]
            yield res[i-(size//2)]




class decomp_tf:
    def __init__(self,dset,size=8192,space=lambda x:np.abs(sp.fft.fft(v)),neighbors=1):
        self.space = space
        self.size = size
        if not (dset is None):
            dset = [i for i in dset]
            self.points = np.array([i[0] for i in dset])
            self.payloads = np.array([i[1] for i in dset])
    def copydat(self,o):
        self.points = o.points
        self.payloads = o.payloads
        self.size = o.size
    def setspace(self,s):
        self.space = s
        self.points = np.array([self.space(v) for v in self.payloads])
    def lookup(self,pt):
        #find the point that removes the most energy from pt
        #a = np.min(pt*|self.points|/self.points,axis=1)
        i = np.argmax(a)
        return (i,a[i])
        

    def gen(self,g,times=1,error_prop_a=0,error_prop_b=0):
        size = self.size
        mask = 1-np.cos(2*np.pi*np.arange(size)/size)/2
        buf = np.zeros(size,dtype=complex)
        i = 0
        res = np.zeros(size,dtype=complex)
        #error = self.points[0]*0
        for v in g:
            buf[i] = v
            i += 1
            if i == size:
                masked = buf*mask
                buf[:size//2] = buf[size//2:]
                i = size//2
                res[:size//2] = res[size//2:]
                res[size//2:] = 0
                pt = self.space(masked)
                for t in range(times):
                    ind,a = self.lookup(pt)
                    rpt = self.points[ind]*a
                    pt -= rpt
                    res += self.payloads[ind]*a
            yield res[i-(size//2)]
