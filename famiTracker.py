


class ConstField:
    def __init__(self,v):
        self.val = v
    def parse(self,f):
        v = f.read(len(self.val))
        assert v == self.val
class Field:
    def __init__(self,l,d=lambda x:x):
        self.val = None
        self.len = l
        self.post = d
    def parse(self,f):
        self.val = self.post(f.read(self.len))


def const(v):
    def r(f,v=v):
        assert f.read(len(v)) == v
    return r

def bint(s=4,e="little"):
    def r(f,s=s,e=e):
        return int.from_bytes(f.read(s),e)
    return r
def bsint(s=4,e="little"):
    def r(f,s=s,e=e):
        v = int.from_bytes(f.read(s),e)
    return r
def nts(f):
    s = b""
    v = f.read(1)
    while v[0] != 0:
        s += v
        v = f.read(1)
    return s
def zeros(n):
    def r(f,n=n):
        for i in range(n):
            assert f.read(1) == b'\x00'
    return r
def zps(size):
    def r(f,s=size):
        o = f.read(s)
        i = o.index(0)
        assert max(o[i:]) == 0
        return o[:i]
    return r

def chipsFromMask(m):
    cs = ["VRC6","VRC7","FDS","MMC5","N163","5B"]
    r = []
    for i in range(len(cs)):
        if m&1:
            r += [cs[i]]
        m >>= 1
    return set(r)

def chipChannels(chips):
    r = [0,1,2,3,4]
    if "VRC6" in chips:
        r += [5,6,7]
    if "MMC5" in chips:
        r += [8,9]
    if "N163" in chips:
        r += [11,12,13,14,15,16,17,18]
    if "FDS" in chips:
        r += [19]
    if "VRC7" in chips:
        r += [20,21,22,23,24,25]
    return r


uint32 = bint()
str16 = zps(16)

import struct
def pad(b,s):
    return b+bytes(s-len(b))
def unpackFile(structDef,f):
    s = struct.calcsize(structDef)
    return struct.unpack(structDef,pad(f.read(s),s))


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
        return struct.unpack(structDef,pad(self[i:i+s],s))
    def shifted(self,s=0):
        return ArrayFile(self.file,self.offset+s)

class FieldDef:
    def __init__(self,name,sdef,include = lambda vers,ns:True, post=lambda x:x):
        self.type = sdef
        if type(self.type) == list:
            print(name,self.type)
            assert 0
        self.name = name
        self.include = include
        self.post = post
    def primitive(self):
        return type(self.type) == str
    def array(self):
        return type(self.type) == StructArrayDef
    def len(self):
        return struct.calcsize(self.type)
    def __repr__(self):
        return "<FieldDef "+self.name+" at "+hex(id(self))+">"
    
class StructDef:
    def __init__(self,fields):
        self.fields = fields
        #self.fieldmap = {fields[i].name:i for i in range(len(fields))}
        
        self.fieldmap = {f.name:f for f in self.fields}
    def field(self,k):
        return self.fields[self.fieldmap[k]]
    def cache(self):
        return [dict(),dict()]
    def lazyrepr(self,c):
        fields_addrs,field_cache = c
        return repr(self)+","+repr(field_cache)
    def lazycontains(self,s,k):
        return s.field(k) != None
    def lazy_stepfield(self,slf,field,o):
        fields_addrs,field_cache = slf.cache
        if field.include(slf.version,slf):
            if field.name in fields_addrs:
                s,o = fields_addrs[field.name]
                o += s.len()
                return o
            s = LazyStruct(slf.dat.shifted(o),field.type,slf.version,slf)
            fields_addrs[field.name] = (s,o)
            o += s.len()
        return o
    def calcsize(self,slf):
        o = 0
        for field in self.fields:
            o = self.lazy_stepfield(slf,field,o)
            #self.fieldnames = self.fieldnames.union(set([field.name]))
        return o
    def lazy_getitem(self,slf,k):
        fields_addrs,field_cache = slf.cache
        if not (k in field_cache):
            f = self.fieldmap[k]
            field_cache[k] = f.post(slf.field(k).val())
        return field_cache[k]
    def lazy__len__(self,slf):
        return len(self.fields)
    def lazy_keys(self,slf):
        slf.len()
        return slf.cache[0].keys()
    def lazy_field(self,slf,k,default):
        fields_addrs,field_cache = slf.cache
        if not (k in fields_addrs):
            o = 0 
            for field in self.fields:
                o = self.lazy_stepfield(slf,field,o)
                if field.name == k:
                    break
        if not (k in fields_addrs):
            return default
        return fields_addrs[k][0]
    
class ProgramaticStructDef:
    def __init__(self,parser):
        self.parse = parser
    def cache(self):
        return None
    def lazyrepr(self,val_cache):
        if val_cache == None:
            return repr(self)
        return repr(val_cache)
    def calcsize(self,slf):
        if slf.cache == None:
            slf.cache,s = self.parse(slf.dat)
        return s
def ntsParser(dat):
    o = 0
    while dat[o]:
        o += 1
    return dat[:o].decode(),o+1
nullTerminatedString = ProgramaticStructDef(ntsParser)
        
class StructArrayDef:
    def __init__(self,struct,count=lambda vers,ns:1,post = lambda x:x):
        self.defn = struct
        self.post = post
        self.count = count
    def cache(self):
        return [None,None,[0,0]]
    def lazyrepr(self,c):
        arr_cache,arr_val_cache,arr_pos = c
        arrs = ""
        if arr_cache != None:
            for i in range(len(arr_cache)):
                if arr_cache[i] != None:
                    if arr_val_cache[i] != None:
                        arrs += repr(arr_val_cache[i])
                    else:
                        arrs += repr(arr_cache[i])
                arrs += ','
        else:
            arrs += repr(self)
        return "["+arrs+"]"
    def lazycontains(self,s,k):
        return 0<=k<len(s)
    def calcsize(self,slf):
        s = 0
        for i in range(len(slf)):
            s += slf.field(i).len()
        return s
    def lazy_array_work(self,slf,i=None):
        arr_cache,arr_val_cache,arr_pos = slf.cache
        if arr_cache == None:
            n = slf.type.count(slf.version,slf)
            arr_cache = slf.cache[0] = [None]*n
            arr_val_cache = slf.cache[1] = [None]*n
        if i == None:
            return
        if arr_cache[i] == None:
            o = arr_pos[1]
            for x in range(arr_pos[0],i+1):
                if arr_cache[x] != None:
                    o += arr_cache[x].len()
                    continue
                s = LazyStruct(slf.dat.shifted(o),slf.type.defn,slf.version,slf)
                o += s.len()
                arr_cache[x] = s
            if arr_pos[0]<i+1:
                arr_pos[0] = i+1
                arr_pos[1] = o
    def lazy_getitem(self,slf,k):
        arr_cache,arr_val_cache,arr_pos = slf.cache
        assert type(k) == int
        v = arr_val_cache[k%len(slf)]
        if v == None:
            v = arr_val_cache[k%len(slf)] = self.post(slf.field(k%len(slf)).val())
        return v
    def lazy__len__(self,slf):
        self.lazy_array_work(slf,None)
        return len(slf.cache[0])
    def lazy_keys(self,slf):
        return range(len(slf))

    def lazy_field(self,slf,k,default=None):
        self.lazy_array_work(slf,k%len(slf))
        return slf.cache[0][k%len(slf)]

class StructSwitch:
    def __init__(self,sf):
        self.type = sf
    def do(self,slf):
        if slf.cache == None:
            slf.cache = LazyStruct(slf.dat,self.type(slf),slf.version,slf)
    def cache(self):
        return None
    def lazyrepr(self,c):
        return repr(self) if c == None else repr(c)
    def lazycontains(self,slf,k):
        self.do(slf)
        return k in slf.cache
    def calcsize(self,slf):
        self.do(slf)
        return slf.cache.len()
    def lazy_getitem(self,slf,k):
        self.do(slf)
        return slf.cache[k]
    def lazy__len__(self,slf):
        self.do(slf)
        return len(slf.cache)
    def lazy_keys(self,slf):
        self.do(slf)
        return slf.cache.keys()
    def lazy_field(self,slf,k,default=None):
        self.do(slf)
        return slf.cache.field(k,default)






    
def defparser(desc):
    if type(desc) == list:
        return StructDef([fieldparser(e) for e in desc])
    
def fieldparser(desc):
    name = desc[0]
    tp = desc[1]
    post = lambda x: x
    if type(tp) != str:
        tp = defparser(tp)
    else:
        post = lambda x:x[0] if len(x) == 1 else x
    ver = lambda v,n:True
    if len(desc) > 2:
        if type(desc[2]) == int:
            ver = lambda v,n,l=desc[2]: v >= l
        elif type(desc[2]) == tuple:
            ver = lambda v,n,l=desc[2]: l[0] <= v <= l[1]
        else:
            ver = desc[2]
        if len(desc) > 3:
            post = desc[3]
    if type(name) == str:
        return FieldDef(name,tp,ver,post)
    elif type(name) == list:
        adef = name
        name = adef[0]
        num = adef[1]
        if type(num) == str:
            num = lambda v,n,l=num: n.parent[l]
        elif type(num) == int:
            num = lambda v,n,l=num: l
        elif type(num) == tuple:
            def t(v,n,l=num):
                for i in range(l[0]):
                    n = n.parent
                for k in l[1:]:
                    n = n[k]
                return n
            num = t
        apost = (lambda x: x) if len(adef) < 3 else adef[3]
        return FieldDef(name,StructArrayDef(tp,num,apost),ver,post)
            
    

    





    
    
    
class LazyStruct:
    def __init__(self,d,defn,v,p):
        self.type = defn #sd
        self.dat = d
        self.version = v
        self.parent = p
        self.lenCache = None
        if type(self.type) == str:
            self.lenCache = struct.calcsize(self.type)
        else:
            self.cache = self.type.cache()
    def __repr__(self):
        if type(self.type) == str:
            return "Lazy("+self.type+",@"+str(self.dat.offset)+")"
        else:
            return "Lazy("+self.type.lazyrepr(self.cache)+",@"+str(self.dat.offset)+")"                
    def __contains__(self,k):
        return self.type.lazycontains(self,k)
    def __iter__(self):
        for k in self.keys():
            yield self[k]
    def __getitem__(self,k):
        return self.type.lazy_getitem(self,k)
    def __len__(self):
        return self.type.lazy__len__(self)
    def keys(self):
        return self.type.lazy_keys(self)
    def val(self):
        if type(self.type) == str:
            return self.dat.unpack(self.type)
        elif type(self.type) == ProgramaticStructDef:
            if self.val_cache == None:
                self.val_cache,self.lenCache = self.type.parse(self.dat)
            return self.val_cache
        return self
    def len(self):
        if self.lenCache == None:
            if type(self.type) == str:
                self.lenCache = struct.calcsize(self.type)
            else:
                self.lenCache = self.type.calcsize(self)
        return self.lenCache
    def field(self,k,default=None):
        return self.type.lazy_field(self,k,default)

    
class FTMBlock(LazyStruct):
    def __init__(self,d,defn,p):
        #d,defn,v,p
        name,version,length = d.unpack('<16sII',0)
        super().__init__(d.shifted(24),defn,version,p)
        self.header = d
        self.name = name.rstrip(b'\x00').decode()
        self.trueLength = length
    def __repr__(self):
        return "<FTMBlock "+self.name+" at "+hex(id(self))+">"



def debugPrintTower(p):
    print(p)
    try:
        debugPrintTower(p.parent)
    except:
        pass
    return p
def debugPrint(p):
    print(p)
    return p

first = lambda x: x[0]
trues = lambda *x: True
FTMParamsDef = StructDef([FieldDef("speed/tempo","<I",lambda v,n: v==1,first),
                          FieldDef("chips","<B",lambda v,n: v>=2,lambda x:chipsFromMask(x[0])),
                          FieldDef("channels","<I",trues,first),
                          FieldDef("pal","<I",trues,first),
                          FieldDef("engine speed","<I",trues,first),
                          FieldDef("new vibrato","<I",lambda v,n: v>=3,first),
                          FieldDef("row highlight","<I",lambda v,n: v>=4,first),
                          FieldDef("2nd highlight","<I",lambda v,n: v>=4,first),
                          FieldDef("N163 channels","<I",lambda v,ns: v>=5 and "N163" in ns['chips'],first),
                          FieldDef("speed/tempo split point","<I",lambda v,n: v>=6,first)])
FTMInfoDef = StructDef([FieldDef("title","<32s",trues,lambda x:x[0].rstrip(b'\x00').decode()),
                        FieldDef("author","<32s",trues,lambda x:x[0].rstrip(b'\x00').decode()),
                        FieldDef("copyright","<32s",trues,lambda x:x[0].rstrip(b'\x00').decode())])
FTMHeaderDef = StructDef([FieldDef("_tracks","<B",lambda v,n: v>=2,lambda x:x[0]+1),
                        FieldDef("tracks",StructArrayDef(nullTerminatedString,lambda v,ns:ns.parent['_tracks']),lambda v,n: v>=3),
                        FieldDef("channel effects cols",
                                 StructArrayDef(
                                     StructDef([
                                         FieldDef("id","<B",trues,first),
                                         FieldDef(
                                             "numEffects",
                                             StructArrayDef(
                                                 "<B",
                                                 lambda v,n:n.parent.parent.parent['_tracks'],
                                                 lambda x: x[0]),                                                
                                         )]),
                                     lambda v,n: n.parent.parent['PARAMS']['channels'],
                                     ),                                 
                                 trues,lambda x: {e["id"]:e["numEffects"] for e in x})])


FTM_2A03_init = defparser([
    ('_numSeqs','<I'),
    (["seqs",'_numSeqs'],[('used?','<B'),('index','<B')]),
    (["notes",lambda v,n: [96,72][v==1]],[('index','<B'),('pitch','<B'),('delta','<B',6)])
])
FTM_VRC6_init = defparser([
    ('_numSeqs','<I',2),
    (['seqs','_numSeqs'],[('used?','<B'),('index','<B')],2)
])
FTM_VRC7_init = defparser([
    ('patch','<I',2),
    ('MML','<8s',2)
])
FTM_FDS_init = defparser([
    ('wave','<64s',3),
    ('mod','<32s',3),
    ('mod rate','<i',3),
    ('mod depth','<i',3),
    ('mod delay','<i',3),
    (['seqs',3],[('_len','<B'),('loop point','<i'),('release point','<i'),('arpeggio type','<i'),
                 (['content','_len'],[('v','<B')])],3)
])
FTM_N163_init = defparser([
    ('_numSeqs','<I',2),
    (['seqs','_numSeqs'],[('used?','<B'),('index','<B')],2),
    ('wave size','<I',2),
    ('wave position','<I',2),
    ('_numWaves','<I',2),
    (['waves','_numWaves'],[(['content',(2,'wave size')],[('v','<B')])],2)
])

FTMInstrDefinitions = [FTM_2A03_init,FTM_VRC6_init,FTM_VRC7_init,FTM_FDS_init,FTM_N163_init]

FTMInstrumentsDef = StructDef([
    FieldDef("_instruments","<I",trues,first),
    FieldDef("instruments",
             StructArrayDef(
                 StructDef([
                     FieldDef("index","<I",trues, first),
                     FieldDef("chip type","<B",trues, first),
                     FieldDef("inst",
                              StructSwitch(
                                  lambda ns: StructDef(FTMInstrDefinitions[ns.parent["chip type"]-1].fields+[
                                      FieldDef("_name len","<I",trues,first),
                                      FieldDef("name",
                                               StructArrayDef(StructDef([FieldDef('v','<B',trues,first)]))
                                               ,trues,lambda x: "".join((chr(e['v']) for e in x)))
                                  ]),
                              )
                              ,trues)]),
                 lambda v,n: n.parent.parent['PARAMS']['channels'],
             ),
             trues,lambda x: {e['index']:e['inst'] for e in x})
])
                 
                     

class FTM:
    field_types = {"PARAMS":FTMParamsDef,
                   "INFO":FTMInfoDef,
                   "HEADER":FTMHeaderDef,
                   "INSTRUMENTS":FTMInstrumentsDef,
                   }
    def __init__(self,f):
        self.dat = ArrayFile(f)
        self.addrs = dict()
        self.fields = dict()
        magic,self.format = self.dat.unpack('<18sI',0)
        assert magic == b'FamiTracker Module'
    def readHeader(self,i):
        r = self.dat.unpack('<16sII',i)
        return r[0].rstrip(b'\x00').decode(),r[1],r[2]
    def __getitem__(self,k):
        return self.field(k)
    def field(self,k):
        if not k in self.fields:
            a = self.field_addr(k)
            if a == None:
                return a
            self.fields[k] = FTMBlock(self.dat.shifted(a),FTM.field_types[k],self)
        return self.fields[k]
    def field_addr(self,k):
        if k in self.addrs:
            return self.addrs[k]
        if len(self.addrs):
            p = max(self.addrs.values())
        else:
            p = 22
        while 1:
            h,m,s = self.readHeader(p)
            self.addrs[h] = p
            if h == k:
                return p
            p += s+24
            if h == "END":
                return None


class FTMParams:
    def __init__(self,d):
        self.dat = d
        self.version, = d.unpack('<I',16)
        self.addrs = dict()
        o = 24
        if self.version == 1:
            self.addrs['speed/tempo'] = o
            o += 4
            self.chips = set()
        else:
            self.addrs['chips'] = o
            o += 1
            self.chips = chipsFromMask(self.dat[self.addrs['chips']])
        self.addrs['channels'] = o
        o += 4
        self.addrs['pal'] = o
        o += 4
        self.addrs['engine speed'] = o
        o += 4
        if self.version >= 3:
            self.addrs['new vibrato'] = o
            o += 4
            if self.version >= 4:
                self.addrs['row highlight'] = o
                o += 4
                self.addrs['2nd highlight'] = o
                o += 4
                if self.version >= 5:
                    if "N163" in self.chips:
                        self.addrs['N163 channels'] = o
                        o += 4
                    if self.version >= 6:
                        self.addrs['speed/tempo split'] = o
    def field_addr(self,k):
        if k in self.addrs:
            return self.addrs[k]
        return None





#http://www.famitracker.com/wiki/index.php?title=FamiTracker_module
#https://wiki.nesdev.com/w/index.php/NSF

def famiTracker(f):
    magic = b'FamiTracker Module'
    assert f.read(len(magic)) == magic
    li = bint()
    si = bint(2)
    sect = zps(16)
    version = li(f)
    if version == 0x0440:
        assert sect(f) == b"PARAMS"
        mystery = li(f)
        skip = li(f)
        chipmask = f.read(1)[0]
        chips = chipsFromMask(chipmask)
        num_channels = li(f)
        pal = li(f)
        custom_speed = li(f)
        vibrato = li(f)
        row_highlight_dist = li(f)
        highlight_2_dist = li(f)
        if "N163" in chips:
            N163_num_channels = li(f)
        speed_tempo_split_point = li(f)
    assert sect(f) == b'INFO'
    mystery = li(f)
    skip = li(f)
    title = zps(32)(f)
    author = zps(32)(f)
    copywrite = zps(32)(f)

    assert sect(f) == b'HEADER'
    mystery = li(f)
    skip = li(f)
    num_tracks = f.read(1)[0]+1
    tracknames = [nts(f) for i in range(num_tracks)]
    channels = chipChannels(chips)
    effects_cols = [{c:0 for c in channels} for i in range(num_tracks)]
    for i in range(len(channels)):
        c = f.read(1)[0]
        for t in range(num_tracks):
            effects_cols[t][c] = f.read(1)[0]+1
    assert sect(f) == b'INSTRUMENTS'
    mystery = li(f)
    skip = li(f)

    num_instruments = li(f)
    instruments = dict()
    for i in range(num_instruments):
        ii = li(f)
        ctype = ["2A03","VRC6","VRC7","FDS","N163"][f.read(1)[0]-1]
        inst = {"type":ctype}
        if ctype == "2A03":
            seq_types = ["volume","arpeggio","pitch","hi-pitch","duty / noise"]
            num_seqs = li(f)
            seqs = dict()
            for i in range(num_seqs):
                seqs[seq_types[i]] = {"used":f.read(1)[0],"index":f.read(1)[0]}
            notes = []
            for n in range(96):
                ind = f.read(1)[0]
                pitch = f.read(1)[0]
                delta = f.read(1)[0]
                notes += [{"index":ind,"pitch":pitch&0x7f,"initial delta":delta,"loop":pitch>>7}]
            defn = (seqs,notes)
        elif ctype == "VRC6":
            seq_types = ["volume","arpeggio","pitch","hi-pitch","pulse width"]
            num_seqs = li(f)
            seqs = dict()
            for i in range(num_seqs):
                seqs[seq_types[i]] = {"used":f.read(1)[0],"index":f.read(1)[0]}
            defn = (seqs,)
        elif ctype == "VRC7":
            patchn = li(f)
            mmlrep = f.read(8)
            defn = (patchn,mmlrep)
        elif ctype == "FDS":
            wave = f.read(64)
            mod = f.read(32)
            modrate = li(f)
            moddepth = li(f)
            moddelay = li(f)
            seqs = dict()
            for k in ("volume","arpeggio","pitch"):
                seq = {"length":f.read(1)[0],"loop":li(f),
                       "release":li(f),"arpeggio":li(f)}
                seq["content"] = f.read(seq["length"])
                seqs[k] = seq
            defn = (wave,mod,modrate,moddepth,moddelay,seqs)
        elif ctype == "N163":
            seq_types = ["volume","arpeggio","pitch","hi-pitch","wave"]
            num_seqs = li(f)
            seqs = dict()
            for i in range(num_seqs):
                seqs[seq_types[i]] = {"used":f.read(1)[0],"index":f.read(1)[0]}
            wave_size = li(f)
            wave_pos = li(f)
            num_waves = li(f)
            waves = [f.read(wave_size) for i in num_waves]
            defn = (seqs,wave_pos,waves)
        nameLen = li(f)
        name = f.read(nameLen)
        inst["definition"] = defn
        inst["name"] = name
        instruments[ctype] = inst

    assert sect(f) == b'SEQUENCES'
    mystery = li(f)
    skip = li(f)

    num_seqs = li(f)
    seqs = dict()
    inds = []
    for s in range(num_seqs):
        ind = li(f)
        inds += [ind]
        stype = ["volume","arpeggio","pitch","hi-pitch","duty / noise"][li(f)]
        #runs = f.read(1)[0]
        #seqrle = [(f.read(1)[0],f.read(1)[0]) for i in range(runs)]
        seqlength = f.read(1)[0]
        loop_point = li(f)
        #release_point = li(f)
        #arpeg_type = li(f)
        content = f.read(seqlength)
        seqs[ind] = [stype,loop_point,content]
    for ind in inds:
        release_point = li(f)
        arpeg_type = li(f)
        seqs[ind] += [release_point,arpeg_type]

    assert sect(f) == b'FRAMES'
    mystery = li(f)
    skip = li(f)

    frames = dict()
    numpatterns = [0]*len(tracknames)
    ti = 0
    for t in tracknames:
        numf = li(f)
        fspeed = li(f)
        ftempo = li(f)
        nrowspf = li(f)
        pattern = f.read(numf*len(channels))
        numpatterns[ti] = max(max(pattern),numpatterns[ti])
        ti += 1
        frames[t] = (fspeed,ftempo,nrowspf,pattern)
    
    assert sect(f) == b'PATTERNS'
    mystery = li(f)
    skip = li(f)
    end = f.tell()+skip

    
    patterns = [dict() for i in range(len(tracknames))]
    for i in range((sum(numpatterns)+len(numpatterns))*len(channels)):
        if f.tell() >= end:
            break
        song_index = li(f)
        channel_order = li(f)
        chn = channels[channel_order]
        pattern_index = li(f)
        num_rows = li(f)
        rows = dict()
        for r in range(num_rows):
            row_index = li(f)
            note = f.read(1)[0]
            octave = f.read(1)[0]
            inst = f.read(1)[0]
            vol = f.read(1)[0]
            effects = [(f.read(1)[0],f.read(1)[0]) for i in range(effects_cols[song_index][chn])]
            rows[row_index] = {'note':note,'octave':octave,'instrument id':inst,'volume':vol,'effects':effects}

        patterns[song_index][pattern_index] = {"channel":chn,"rows":rows}



    
    assert sect(f) == b'DPCM SAMPLES'
    mystery = li(f)
    skip = li(f)

    num_dpcm_samples = f.read(1)[0]
    dpcm_samples = dict()
    for i in range(num_dpcm_samples):
        index = f.read(1)[0]
        namelen = li(f)
        name = f.read(namelen)
        slen = li(f)
        content = f.read(slen)
        dpcm_samples[index] = (name,content)



    if "N163" in chips:
        assert sect(f) == b'SEQUENCES_N163'
        mystery = li(f)
        skip = li(f)

        numNseqs = li(f)
        seqs_n163 = dict()
        for s in range(numNseqs):
            si = li(f)
            ty = ["volume","arpeggio","pitch","hi-pitch","wave"][li(f)]
            l = f.read(1)[0]
            looppt = li(f)
            relpt = li(f)
            arpegt = li(f)
            content = f.read(l)
            seqs_n163[si] = (ty,looppt,relpt,arpegt,content)
    
    assert sect(f) == b'COMMENTS'
    mystery = li(f)
    skip = li(f)

    shown = li(f)
    comment = nts(f)

    if shown:
        print(comment)
    
    if "VRC6" in chips:
        assert sect(f) == b'SEQUENCES_VRC6'
        mystery = li(f)
        skip = li(f)
        
        num_seqs = li(f)
        seqs_vrc6 = dict()
        inds = []
        for s in range(num_seqs):
            ind = li(f)
            inds += [ind]
            stype = ["volume","arpeggio","pitch","hi-pitch","pulse width"][li(f)]
            #runs = f.read(1)[0]
            #seqrle = [(f.read(1)[0],f.read(1)[0]) for i in range(runs)]
            seqlength = f.read(1)[0]
            loop_point = li(f)
            #release_point = li(f)
            #arpeg_type = li(f)
            content = f.read(seqlength)
            seqs_vrc6[ind] = [stype,loop_point,content]
        for ind in inds:
            release_point = li(f)
            arpeg_type = li(f)
            seqs_vrc6[ind] += [release_point,arpeg_type]

    assert f.read(3) == b'END'
    return locals()    
