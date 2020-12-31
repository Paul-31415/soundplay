import struct

def pad(b,s):
    return b+b'\x00'*(s-len(b))


class ArrayFile:
    def __init__(self,f,o=0):
        if type(f) == str:
            f = open(f,"rb")
        self.file = f
        self.offset = o
    def __len__(self):
        self.file.seek(0,2)
        return self.file.tell()-self.offset
    def __getitem__(self,i):
        if type(i) == slice:
            start, stop, step = i.indices(len(self))
            assert step != 0
            if step > 0:
                if stop<start:
                    return b''
                self.file.seek(start+self.offset)
                return self.file.read(stop-start)[::step]
            else:
                if stop>start:
                    return b''
                self.file.seek(stop+self.offset)
                return self.file.read(start-stop)[::step]
        self.file.seek(i+(i>=0)*self.offset,i<0)
        return self.file.read(1)[0]
    def unpack(self,structDef,i=0):
        s = struct.calcsize(structDef)
        if i < 0 and self.offset+i >= 0:
            return self.shifted(-self.offset).unpack(structDef,self.offset+i)
        return struct.unpack(structDef,pad(self[i:i+s],s))
    def shifted(self,s=0):
        assert s <= len(self)
        return ArrayFile(self.file,self.offset+s)


class Lazy:
    def __init__(self,func,*args):
        self.f = func
        self.a = args
        if not callable(func):
            if len(self.a) != 0:
                raise Exception("Gave a value to Lazy with > 0 args.",func,args)
            self.c = self.f
            self.v = True
        else:
            self.c = None
            self.v = False
    def eval(self):
        if not self.v:
            try:
                self.c = self.f(*self.a)
            except Exception as e:
                print("Error with func:",self.f,"args:",self.a)
                raise e
            self.v = True
        return self.c
    def __repr__(self):
        if self.v:
            return "Lazy<ran>("+repr(self.c)+")"
        return "Lazy("+repr(self.f)+repr(self.a)+")"
    def __call__(self,*args):
        return Lazy(lambda a,b: val(a)(val(b)),self,args)
    def __eq__(self,o):
        return Lazy(lambda a,b: val(a)==val(b),self,o)
    def __neq__(self,o):
        return Lazy(lambda a,b: val(a)!=val(b),self,o)
    def __lt__(self,o):
        return Lazy(lambda a,b: val(a)<val(b),self,o)
    def __le__(self,o):
        return Lazy(lambda a,b: val(a)<=val(b),self,o)
    def __gt__(self,o):
        return Lazy(lambda a,b: val(a)>val(b),self,o)
    def __ge__(self,o):
        return Lazy(lambda a,b: val(a)>=val(b),self,o)
    def __add__(self,o):
        return Lazy(lambda a,b: val(a)+val(b),self,o)
    def __radd__(self,o):
        return Lazy(lambda a,b: val(a)+val(b),o,self)
    def __and__(self,o):
        return Lazy(lambda a,b: val(a) and val(b),self,o)
    def __or__(self,o):
        return Lazy(lambda a,b: val(a) or val(b),self,o)
    def __contains__(self,k):
        return Lazy(lambda a,b: val(a) in val(b),k,self)
    
def val(v):
    head = v
    while issubclass(type(v),Lazy):
        v = v.eval()
    while issubclass(type(head),Lazy):
        #writeback the vals to shorten the linked list for next time
        head.c,head = v,head.c
    return v
    
#testfiles
dinf = ArrayFile("/Users/paul/Music/other/chiptune/Deathro/BotB 21733 Deathro - I Never Forget.ftm")
prism = ArrayFile("/Users/paul/Music/other/chiptune/Deathro/The Prism's Eye FINAL 11-30-2015.ftm")


class Parser:
    def __init__(self):
        assert 0
    def bytelen(self):
        return 0
    def parse(self):
        return self

def pcurry(f,*a):
    def r(a,b=a):
        return f(a,*b)
    return r

def FTMBlockHeaderParse(f):
    name,version,length = f.unpack("<16sII")
    name = name.rstrip(b'\x00').decode()
    if name == "END":
        return (None,0,0)
    return (name,24,length)
    
class LinkedDict(Parser):
    def __init__(self,f,entryParser=FTMBlockHeaderParse,cache_addrs=True):
        self.f = f
        self.p = entryParser
        self.cache = cache_addrs
        if cache_addrs:
            self.addrs = dict()
            self.nextAddr = 0
            self.cache_length = None
    def _valify(self,k,addr):
        return self.f.shifted(addr)
    def _raw_iter(self):
        if self.cache:
            o = self.nextAddr
        else:
            o = 0
        key = 0
        while key != None:
            key,hlen,delta = self.p(self.f.shifted(o))
            if self.cache:
                self.addrs[key] = o+hlen
                self.nextAddr = o+hlen+delta
            if key != None:
                yield key,o+hlen
            o += hlen+delta
        if self.cache:
            self.cache_length = o
        yield None,o
    def _cache_or_raw_iter(self):
        if self.cache:
            for k,v in self.addrs.items():
                yield k,v
        for k,v in self._raw_iter():
            yield k,v
    def pairs(self):
        for k,v in self._cache_or_raw_iter():
            if k != None:
                yield k,self._valify(k,v)
    def __iter__(self):
        for k,v in self.pairs():
            yield v
    def keys(self):
        for k,v in self.pairs():
            yield k
    def __len__(self):            
        i = -1
        for k,v in self._cache_or_raw_iter():
            i += 1
        if self.cache:
            return len(self.addrs)
        return i
        
    def keylist(self):
        return [k for k in self.keys()]
    def bytelen(self):
        if self.cache:
            if self.cache_length != None:
                return self.cache_length
        o = 0
        for k,v in self._raw_iter():
            o = v
        return o
    def __getitem__(self,key):
        if self.cache:
            if key in self.addrs:
                return self._valify(key,self.addrs[key])
        for k,v in self._raw_iter():
            if k==None:
                return None
            if key==k:
                return self._valify(k,v)
    
class TypedLinkedDict(LinkedDict):
    def __init__(self,f,typeKeys,entryParser=FTMBlockHeaderParse,cache_addrs=True,cache_vals=True):
        super().__init__(f,entryParser,cache_addrs)
        self.types = typeKeys
        self.cache_v = cache_vals
        if cache_vals:
            self.vals = dict()
    def _valify(self,k,addr):
        return self.types[k](super()._valify(k,addr),self)
    def __getitem__(self,k):
        if self.cache_v:
            if not (k in self.vals):
                f = super().__getitem__(k)
                if f == None:
                    return f
                self.vals[k] = f
                return f
            return self.vals[k]
        return super().__getitem__(k)
class LazyTypedLinkedDict(LinkedDict):
    def __init__(self,f,typeKeys,entryParser=FTMBlockHeaderParse,cache_addrs=True,cache_vals=True):
        super().__init__(f,entryParser,cache_addrs)
        self.types = typeKeys
        self.cache_v = cache_vals
        if cache_vals:
            self.vals = dict()
    def _valify(self,k,addr):
        return Lazy(self.types[k],super()._valify(k,addr),self)
    def __getitem__(self,k):
        if self.cache_v:
            if not (k in self.vals):
                f = super().__getitem__(k)
                if f == None:
                    return f
                self.vals[k] = f
                return f
            return self.vals[k]
        return super().__getitem__(k)


    
class LazyHomogenousLengthedDict(LazyTypedLinkedDict):
    def __init__(self,f,l,t,eparser,cache_addrs=True,cache_vals=True):
        super().__init__(f,t,eparser,cache_addrs,cache_vals)
        if cache_addrs:
            self.totalCached = 0
        self.l = l
    def __len__(self):
        return self.l
    def _valify(self,k,addr):
        return Lazy(self.types,self.f.shifted(addr),self)
    def _raw_iter(self):
        n = self.l
        if self.cache:
            o = self.nextAddr
            n -=self.totalCached
        else:
            o = 0
        for i in range(n):
            key,hlen,delta = self.p(self.f.shifted(val(o)))
            if self.cache:
                self.addrs[key] = val(o+hlen)
            if key != None:
                yield key,val(o+hlen)
            if delta == 'unknown':
                if self.cache_v:
                    if not key in self.vals:
                        v = self.types(self.f.shifted(val(o+hlen)),self)
                        self.vals[key] = v
                    else:
                        v = self.vals[key]
                else:
                    v = self.types(self.f.shifted(val(o+hlen)),self)
                delta = v.bytelen()
            if self.cache:
                self.nextAddr = val(o+hlen+delta)
                self.totalCached += 1
            o += val(hlen+delta)
        if self.cache:
            self.cache_length = o
        yield None,o
    def __repr__(self):
        return "LazyHomogenousLengthedDict[l="+str(self.l)+"]("+(repr(self.addrs.keys())+(',...' if len(self.addrs)<self.l else '') if self.cache else "uncached")+")"
class HomogenousLengthedDict(LazyHomogenousLengthedDict):
    def __init__(self,f,l,t,eparser,cache_addrs=True,cache_vals=True):
        super().__init__(f,l,t,eparser,cache_addrs,cache_vals)
    def __getitem__(self,k):
        return val(super().__getitem__(k))
    def __repr__(self):
        return "HomogenousLengthedDict[l="+str(self.l)+"]("+(repr(self.addrs.keys())+(',...' if len(self.addrs)<self.l else '') if self.cache else "uncached")+")"
    
def bytelen(p):
    try:
        return p.bytelen()
    except AttributeError:
        return 0
    #if issubclass(type(p),Parser):
    #    return p.bytelen()
    #return 0
def parse(p):
    v = val(p)
    try:
        return p.parse()
    except AttributeError:
        return p
    
class ConditionalParam(Lazy):
    def __init__(self,cond,valu,default=None):
        #def funk(c,v,d):
        #    if val(c):
        #        print("cond true",v)
        #        return v
        #    print("cond false",v)
        #    return d
        super().__init__(lambda c,v,d: v if val(c) else d,
                         cond,valu,default)
        self.cond = cond
        self.val = valu
        self.default = default
    def bytelen(self):
        return bytelen(val(self))
    def __repr__(self):
        return "condparam"+repr((self.cond,self.val,self.default))
    def parse(self):
        return val(self).parse()

class ParserList(Parser):
    def __init__(self,post,*pars):
        self.par = pars
        self.post = post
    def bytelen(self):
        return sum((bytelen(p) for p in self.par))
    def val(self):
        return self.post(self.par)
    
class LazyOrderedDict(Parser):
    def __init__(self,f,names,ps):
        self.f = f
        parsers = []
        o = f
        for p in ps:
            v = Lazy(p,o)
            parsers += [v]
            o = Lazy(lambda f,p,i: val(f).shifted(bytelen(val(p[i]))),o,parsers,len(parsers)-1)
        self.par = parsers
        self.vals = {names[i]:i for i in range(len(names))}
    def __getitem__(self,k):
        return parse(val(self.par[self.vals[k]]))
    def keys(self):
        return self.vals.keys()
    def __repr__(self):
        return "LazyOrderedDict("+repr(self.keys())+")"
    def bytelen(self):
        return sum((bytelen(val(p)) for p in self.par))
    def setfield(self,k,v):
        self.par[self.vals[k]] = v
    def getfield(self,k):
        return self.par[self.vals[k]]
        
class Struct(Parser):
    def __init__(self,f,desc,post = lambda x:x[0],cache=False):
        self.f = f
        self.desc = desc
        self.post = post
        self.cache = cache
        if cache:
            self.cache_valid = False
            self.cache_value = None
    def bytelen(self):
        return struct.calcsize(self.desc)
    def parse(self):
        if self.cache:
            if not self.cache_valid:
                self.cache_value = self.post(self.f.unpack(self.desc))
                self.cache_valid = True
            return self.cache_value
        return self.post(self.f.unpack(self.desc))
    def __repr__(self):
        return "Struct("+self.desc+" @"+str(self.f.offset)+")"
    def __contains__(self,k):
        return k in self.parse()
def params(f,_):
    def chipsFromMask(m):
        cs = ["VRC6","VRC7","FDS","MMC5","N163","5B"]
        r = ["2A03"]
        for i in range(len(cs)):
            if m&1:
                r += [cs[i]]
            m >>= 1
        return set(r)
    ver = Lazy(f.unpack('<I',-8)[0])
    chips = None
    
    r = LazyOrderedDict(f,('speed','chips','channels','PAL','engine speed','vibrato','row highlight','2nd highlight','N163 channels','speed/tempo split'),
                        (lambda f: ConditionalParam(ver==1,Struct(val(f),"<I"),None),
                         lambda f: ConditionalParam(ver>=2,Struct(val(f),"<B",lambda x: chipsFromMask(x[0]),True),set("2A03")),
                         lambda f: Struct(val(f),"<I"),
                         lambda f: Struct(val(f),"<I"),
                         lambda f: Struct(val(f),"<I"),
                         lambda f: ConditionalParam(ver>=3,Struct(val(f),"<I"),None),
                         lambda f: ConditionalParam(ver>=4,Struct(val(f),"<I"),None),
                         lambda f: ConditionalParam(ver>=4,Struct(val(f),"<I"),None),
                         lambda f: f,#placeholder
                         lambda f: ConditionalParam(ver>=6,Struct(val(f),"<I"),None)))
    r.setfield("N163 channels",Lazy(lambda f,chips: ConditionalParam((ver>=5).__and__(chips.__contains__("N163")),Struct(val(f),"<I"),None),
                                    r.getfield("N163 channels").a[0],Lazy(lambda s: val(s).parse(),r.getfield("chips"))))
    return r

def info(f,_):
    ver = Lazy(f.unpack('<I',-8)[0])
    return LazyOrderedDict(f,('Title','Author','Copyright'),
                           [lambda f: Struct(val(f),"<32s",lambda x: x[0].rstrip(b'\x00').decode())]*3)



def header(f,m):
    ver = Lazy(f.unpack('<I',-8)[0])
    r = LazyOrderedDict(f,('#tracks','tracks','effects'),
                        (lambda f: ConditionalParam(ver>=2,Struct(val(f),"<B",lambda x: x[0]+1),None),
                         lambda f:f,#placeholder
                         lambda f:f))
    num_tracks = Lazy(lambda s: val(s).parse(),r.getfield('#tracks'))
    r.setfield('tracks',Lazy(lambda f,num: ConditionalParam(ver>=3,StructArray(val(f),num,NullTerminatedString),None),
                             r.getfield('tracks').a[0],num_tracks))

    #f,l,t,eparser,cache_addrs=True,cache_vals=True
    r.setfield('effects',Lazy(lambda f,mod,num: LazyHomogenousLengthedDict(val(f),mod['PARAMS']['channels'],
                                                                           lambda g,*_: StructArray(val(g),num,lambda i:Struct(val(i),"<B",lambda x: x[0]+1)),
                                                                           lambda p: (p.unpack("<B")[0],1,num)
                                                                           )
                              ,r.getfield('effects').a[0],m,num_tracks))
    
    
    return r

def instruments(f,m):
    ver = f.unpack('<I',-8)[0]
    #r = LazyOrderedDict(f,('#instruments','instruments'),
    def inst_2a03(f,v):
        f = val(f)
        r = LazyOrderedDict(f,('#sequences','volume','arpeggio','pitch','hi-pitch','duty / noise','notes'),
                            (lambda f: Struct(val(f),"<I"),
                             lambda f: f,#placeholder
                             lambda f: f,#placeholder
                             lambda f: f,#placeholder
                             lambda f: f,#placeholder
                             lambda f: f,#placeholder
                             lambda f: FixedLengthStructArray(val(f),[72,96][val(v>1)],
                                                   lambda g:LazyOrderedDict(g,('index','pitch','delta'),
                                                                            (lambda h:Struct(val(h),"<B"),
                                                                             lambda h:Struct(val(h),"<B"),
                                                                             lambda h:ConditionalParam(v>=6,Struct(val(h),"<B"),None)))
                             )
                            ))
        seqs = Lazy(lambda s:s['#sequences'],r)
        sd = ('volume','arpeggio','pitch','hi-pitch','duty / noise')
        for i in range(len(sd)):
            r.setfield(sd[i],Lazy(lambda f,num,k: ConditionalParam(num>k,LazyOrderedDict(f,("used?","index"),
                                                                                         (lambda g:Struct(val(g),"<B"),
                                                                                          lambda g:Struct(val(g),"<B"))),None)
                                  ,r.getfield(sd[i]).a[0],seqs,i)
                       )
        return r
    def inst_vrc6(f,v):
        f = val(f)
        
        r = LazyOrderedDict(f,('#sequences','volume','arpeggio','pitch','hi-pitch','pulse width'),
                            (lambda f: Struct(val(f),"<I"),
                             lambda f: f,#placeholder
                             lambda f: f,#placeholder
                             lambda f: f,#placeholder
                             lambda f: f,#placeholder
                             lambda f: f,#placeholder
                            ))
        seqs = Lazy(lambda s:s['#sequences'],r)
        sd = ('volume','arpeggio','pitch','hi-pitch','pulse width')
        for i in range(len(sd)):
            r.setfield(sd[i],Lazy(lambda f,num,k: ConditionalParam(num>k,LazyOrderedDict(f,("used?","index"),
                                                                                         (lambda g:Struct(val(g),"<B"),
                                                                                          lambda g:Struct(val(g),"<B"))),None)
                                  ,r.getfield(sd[i]).a[0],seqs,i)
                       )
        return ConditionalParam(v>=2,r,None)
    def inst_vrc7(f,v):
        f = val(f)
        r = LazyOrderedDict(f,("patch number","MML"),
                            (lambda f: Struct(val(f),"<I"),
                             lambda f: Struct(val(f),"<8s")))
        return ConditionalParam(v>=2,r,None)
    def inst_fds(f,v):
        f = val(f)
        r = LazyOrderedDict(f,('wave','mod','mod rate','mod depth','mod delay',
                               'volume','arpeggio','pitch'),
                            (lambda f: Struct(val(f),"<64B",lambda x: x),
                             lambda f: Struct(val(f),"<32B",lambda x: x),
                             lambda f: Struct(val(f),"<i"),
                             lambda f: Struct(val(f),"<i"),
                             lambda f: Struct(val(f),"<i"),
                             lambda f: f,#placeholder
                             lambda f: f,#placeholder
                             lambda f: f,#placeholder
                            ))
        sd = ('volume','arpeggio','pitch')
        def modify(lod):
            lod.setfield("content",Lazy(lambda f,l:FixedLengthStructArray(val(f),l['_len'],lambda g:Struct(val(g),"<B")),
                                        lod.getfield("content").a[0],lod))
            return lod
        for i in range(len(sd)):
            r.setfield(sd[i],Lazy(lambda f,m:m(LazyOrderedDict(f,("_len","loop point","release point","arpeggio type","content"),
                                                               (lambda g:Struct(val(g),"<B"),
                                                                lambda g:Struct(val(g),"<i"),
                                                                lambda g:Struct(val(g),"<i"),
                                                                lambda g:Struct(val(g),"<i"),
                                                                lambda g:g #placeholder
                                                               )))
                                  ,r.getfield(sd[i]).a[0],modify)
            )
        return ConditionalParam(v>=3,r,None)
    def inst_n163(f,v):
        f = val(f)
        r = LazyOrderedDict(f,('#sequences','volume','arpeggio','pitch','hi-pitch','wave',
                               '_wave size','wave position','#waves','waves'),
                            (lambda f: Struct(val(f),"<I"),
                             lambda f: f,#placeholder
                             lambda f: f,#placeholder
                             lambda f: f,#placeholder
                             lambda f: f,#placeholder
                             lambda f: f,#placeholder
                             lambda f: Struct(val(f),"<i"),
                             lambda f: Struct(val(f),"<i"),
                             lambda f: Struct(val(f),"<i"),
                             lambda f: f,#placeholder
                            ))
        r.setfield('waves',Lazy(lambda f,s: FixedLengthStructArray(val(f),s['#waves'],lambda g,t=s:FixedLengthStructArray(val(g),t['_wave size'],lambda h:Struct(val(h),"<B"))),
                                r.getfield('waves').a[0],r))
        seqs = Lazy(lambda s:s['#sequences'],r)
        sd = ('volume','arpeggio','pitch','hi-pitch','wave')
        for i in range(len(sd)):
            r.setfield(sd[i],Lazy(lambda f,num,k: ConditionalParam(num>k,LazyOrderedDict(f,("used?","index"),
                                                                                         (lambda g:Struct(val(g),"<B"),
                                                                                          lambda g:Struct(val(g),"<B"))),None)
                                  ,r.getfield(sd[i]).a[0],seqs,i)
            )
        return ConditionalParam(v>=2,r,None)
    
    r = LazyOrderedDict(f,('#instruments',
                           'instruments'),
                        (lambda f: Struct(val(f),"<I"),
                         lambda f: f,#placeholder
                        ))
    def modify(lod):
        ct = Lazy(lambda s: s['chiptype'],lod)
        lod.setfield('definition',Lazy(lambda f,c,im,v:
                                       im[val(c)](f,v)
                                       ,lod.getfield('definition').a[0],
                                       ct,{"2A03":inst_2a03,
                                           "VRC6":inst_vrc6,
                                           "VRC7":inst_vrc7,
                                           "FDS":inst_fds,
                                           "N163":inst_n163},
                                       ver)
        )
        return lod
    r.setfield('instruments',Lazy(lambda f,s: 
                                  HomogenousLengthedDict(val(f),s['#instruments'],
                                                         lambda g,*_: modify(LazyOrderedDict(val(g),("chiptype","definition","name"),
                                                                                             (lambda h:Struct(val(h),'<B',lambda x: ["2A03","VRC6","VRC7","FDS","N163"][x[0]-1]),
                                                                                              lambda h: h,
                                                                                              lambda h: LengthPrefixedString(val(h)))
                                                         )),
                                                         lambda p: (p.unpack("<I")[0],4,"unknown")
                                  )
                                  ,r.getfield('instruments').a[0],r)
    )
    return r

def sequences(f,m,tps = ['Volume','Arpeggio','Pitch','Hi-pitch','Duty / Noise']):
    ver = f.unpack('<I',-8)[0]
    numSeqs =  f.unpack("<I")[0]
    def modify(r):
        r.setfield('content',Lazy(lambda g,l:ConditionalParam(ver>=3,FixedLengthStructArray(val(g),l['_len seq'],
                                                                                            lambda h: Struct(val(h),"<B"))
        )
                                  ,r.getfield('content').a[0],r))
        return r
    r = HomogenousLengthedDict(f.shifted(4),numSeqs,
                               lambda f,_: modify(LazyOrderedDict(val(f),('rle content','_len seq','loop point','release point','arpeggio type','content'),
                                                                  (lambda g: ConditionalParam((ver>=1).__and__(ver<=2),length_prefixed_array(val(g),'<B',lambda h:LazyOrderedDict(val(h),('val','run length'),[lambda i: Struct(val(i),'<b')]*2),FixedLengthStructArray)),
                                                                   lambda g: ConditionalParam(ver>=3,Struct(val(g),"<B")),
                                                                   lambda g: ConditionalParam(ver>=3,Struct(val(g),"<i")),
                                                                   lambda g: ConditionalParam(ver==4,Struct(val(g),"<i")),
                                                                   lambda g: ConditionalParam(ver==4,Struct(val(g),"<i")),
                                                                   lambda g: g,#placeholder
                                                                )
                               )),
                               lambda p: (p.unpack("<I")[0] if ver == 1 else (lambda s: (s[0],tps[s[1]]))(p.unpack("<II")),4+4*(ver>=2),"unknown")
    )

        
    if ver >= 5:
        s = Lazy(lambda r,n: FixedLengthStructArray(f.shifted(r.bytelen()+4),val(n),lambda f:
                                                    LazyOrderedDict(val(f),("release point","arpeggio type"),
                                                                    (lambda g: Struct(val(g),"<i"),
                                                                     lambda g: Struct(val(g),"<i"))))
                 ,r,numSeqs)
        return {"sequences":r,"sequence attrs":s}
    else:
        return r


def ftm(f):
    return TypedLinkedDict(f.shifted(18+4),{"PARAMS":params,"INFO":info,"HEADER":header
                                            ,"INSTRUMENTS":instruments,"SEQUENCES":sequences
                                            ,"SEQUENCES_VRC6":lambda a,b: sequences(a,b,['Volume','Arpeggio','Pitch','Hi-pitch','Pulse width'])
                                            ,})
    
    

class VersionedSimpleStruct:
    def __init__(self,descs,version=lambda f: f.unpack('<I',-8)[0]):
        self.descs = descs
        self.version = version
    def __call__(self,f):
        v = self.version(f)
        s,n = self.descs[self.version]
        s = f.unpack(s,0)
        r = {n[i]:s[i] for i in range(len(n))}
        return r
    
class StructArray(Parser):
    def __init__(self,f,l,t):
        self.f = f
        self.l = l
        self.t = t
        self.vals = None
    def parse(self):
        if self.vals == None:
            parsers = []
            o = self.f
            for p in range(val(self.l)):
                v = Lazy(self.t,o)
                parsers += [v]
                o = Lazy(lambda f,p,i: val(f).shifted(bytelen(val(p[i]))),o,parsers,len(parsers)-1)
            self.vals = parsers
        return self
    def __len__(self):
        return val(self.l)
    def __getitem__(self,k):
        self.parse()
        return parse(val(self.vals[k]))
    def __repr__(self):
        return "StructArray["+repr(self.l)+"]("+repr(self.t)+")"
    def bytelen(self):
        self.parse()
        return sum(bytelen(val(p)) for p in self.vals)
class FixedLengthStructArray(StructArray):
    def __init__(self,f,l,t):
        super().__init__(f,l,t)
        self.el_len = None
    def parse(self):
        if self.vals == None:
            if val(self.l) == 0:
                return self
            o = self.f
            parsers = [Lazy(self.t,o)]
            self.el_len = Lazy(lambda e:val(e).bytelen(),parsers[0])
            for p in range(1,val(self.l)):
                o = Lazy(lambda f,el,n: val(f).shifted(val(el)*n),self.f,self.el_len,p)
                v = Lazy(self.t,o)
                parsers += [v]
            self.vals = parsers
        return self
    def bytelen(self):
        if val(self.l) == 0:
            return 0
        if self.el_len == None:
            self.el_len = self.t(self.f).bytelen()
        return val(self.el_len)*val(self.l)
    def __repr__(self):
        return "FLStructArray["+repr(self.l)+",(x"+repr(self.el_len)+")]("+repr(self.t)+")"
def length_prefixed_array(f,l,t,tp):
    ld = l
    l = Lazy(lambda f,l: f.unpack(val(l))[0],f,l)
    return tp(f.shifted(struct.calcsize(val(ld))),val(l),t)
    
def nts(f):
    o = 0
    r = ""
    v = f[o]
    while v:
        r += chr(v)
        o += 1
        v = f[o]
    return r,o+1


class ProgrammaticParser:
    def __init__(self,f,parser=nts,cache_len=True,cache_val=False):
        self.f = f
        self.p = parser
        self.cache_a = cache_len
        self.cache_v = cache_val
        if self.cache_a:
            self.length = None
        if self.cache_v:
            self.val = None
    def __repr__(self):
        return "ProgrammaticParser("+repr(self.p)+")"
    def bytelen(self):
        if self.cache_a:
            if self.length == None:
                self.value()
            return self.length
        v,l = self.p(self.f)
        return l
    def value(self):
        if self.cache_v:
            if self.val != None:
                return self.val
            v,l = self.p(self.f)
            self.val = v
            if self.cache_a:
                self.length = l
            return v
        v,l = self.p(self.f)
        if self.cache_a:
            self.length = l
        return v
            
class NullTerminatedString:
    def __init__(self,f,cache=True):
        self.cache = cache
        self.f = f
        if cache:
            self.length = None
    def __len__(self):
        return self.bytelen()-1
    def bytelen(self):
        if self.cache:
            if self.length == None:
                self.parse()
            return self.length
        return len(self.value())+1
    def parse(self):
        if self.cache:
            if self.length != None:
                return f[:self.length-1].decode()
            res,l = nts(self.f)
            self.length = l
            return res
        return nts(self.f)
class LengthPrefixedString(Parser):
    def __init__(self,f,lc="<I"):
        self.f = f
        self.lc = lc
    def __len__(self):
        return self.f.unpack(self.lc)[0]
    def bytelen(self):
        return struct.calcsize(self.lc)+self.f.unpack(self.lc)[0]
    def parse(self):
        l = len(self)
        o = struct.calcsize(self.lc)
        return self.f[o:o+l].decode()
    



        



"""


some things

null terminated string = null | (non-null, n.t.s) ;
int_n                  = byte * n ;

FamiTracker Module     = 'FamiTracker Module', int4 version, {block}, 'END' ;
block                  = block header, block body ;
block header           = string16 type, int4 format, int4 len ;
block body             = cond(header):
                            case type=='PARAMS':
                               int4  (speed if FTM.version == 0.2.2 else tempo)  if format == 1
                               byte  chips  if format >= 2
                               int4  channels
                               int4  pal
                               int4  engine speed
                               int4  vibrato mode  if format >= 3
                               int4  row highlight  if format >= 4
                               int4  2nd highlight  if format >= 4
                               int4  N163 channels  if N163 in chips
                            case type=='INFO':
                               bytes32  title
                               bytes32  author
                               bytes32  copyright
                            case type=='HEADER':
                               byte  numTracks              if format >= 2
                               nts[numTracks]  track names  if format >= 3
                               [FTM.PARAMS.channels]:
                                 byte  channel id
                                 byte[numTracks]  extra effect columns
                            case type=="INSTRUMENTS":
                               int4  numInstruments
                               [numInstruments]
                                 int4  index
                                 byte enum  chip type  (one of 2A03 VRC6 VRC7 FDS N163)
                                 instrument definition  definition
                                 int4  nameLen
                                 byte[nameLen]  name
                                 ***
instrument definition =   switch  chip type
                            case 2A03:
                               numNotes = 72 if format == 1 else 96
                               int4  numSequenceTypes
                               [numSequenceTypes]:
                                 byte  used?
                                 byte  index
                               [numNotes]:
                                 byte  DPCM index (00 for no)
                                 byte  default pitch
                                 byte  initial delta counter (FF for off)  if format >= 6
                            case VRC6:
                               int4 numSequenceTypes
                               [numSequenceTypes]:
                                 byte  used?  if format >= 2
                                 byte  index  if format >= 2
                            case VRC7:
                               int4  patch number  if format >= 2
                               bytes8  MML patch string  if format >= 2
                            case FDS:
                               if format >= 3:
                                   byte[64]  wave
                                   byte[32]  mod
                                   int4  mod rate
                                   int4  mod depth
                                   int4  mod delay
                                   [sequences = 3]:
                                     byte  length
                                     int4  loop point
                                     int4  release point
                                     int4  arpeggio type
                                     byte[length]  content
                            case N163:
                               if format >= 2:
                                   int4  numDefined
                                   [numDefined]:
                                     byte  used?
                                     byte  index
                                   int4  wave size
                                   int4  wave pos
                                   int4  num waves
                                   [num waves]:
                                     byte[wave size]  content
                        ***
                            case type == "SEQUENCES":
                              
 

"""
