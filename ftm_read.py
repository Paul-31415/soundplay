import struct
#use like this:
"""
from a import *
import ftm_read as fr
def savetee(g):
    global res
    for v in g:
        res += [v]
        yield v

def sumf(v):
 t = v[0]
 for r in v[1:]:
  t += r
 return t

(None,reload(fr))[0]
ft = fr.ftm(fr.f.read())
ftu = fr.ftm(fr.ftu.read())
sts = fr.ftm(fr.sts.read())
mix.out = (res:=[],(sumf([r[0]*0,r[1],r[2]*30*0,r[3]*0,r[4]*30*0,r[5]*0,r[6]*0,r[7]*0]) for r in savetee((lambda a:([next(v) for v in a[1:]] for i in a[0]))([ (i*(1+1j) for i in v) for v in fr.tsts(ftu,800,b=32.7) ]))))[1]
"""
f = open("/Users/paul/Music/other/chiptune/nsf/The Prism's Eye FINAL 11-30-2015.ftm",'rb')
ftu = open("/Users/paul/Music/other/chiptune/nsf/FamiTracker Modules/Face_the_Unknown_(Jayster).ftm","rb")
sts = open("/Users/paul/Music/other/chiptune/nsf/FamiTracker Modules/Sunrise_to_Sunset_(Jayster).ftm","rb")
#notes:
#it appears that the DPCM is 1 bit dpcm LSB first


def tng(ft,c=0):
    for fr in ft.frames[0]:
        p = fr[c]
        pat = ft.patterns[0][p]
        if c in pat:
            pat = pat[c]
        else:
            continue
        for r in range(ft.frames.songs[0]['rows per frame']):
            yield pat[r]
import math
def test(ft,c=0,a=22,d=800,nf = math.sin,ro=None):
    vol = 0
    note = 13
    phase = 0
    rvol = 1
    octave = 0
    ro = ft.frames.songs[0]['rows per frame'] if ro == None else ro
    dp = 0
    for fr in ft.frames[0]:
        p = fr[c]
        pat = ft.patterns[0][p]
        if c in pat:
            pat = pat[c]
        else:
            for r in range(ro):
                for i in range(d):
                    rvol = rvol*.999
                    yield nf(phase)*rvol
                    phase += dp
            continue
        for r in range(ro):
            nd = pat[r]
            if nd['volume'] != None:
                vol = nd['volume']
            if nd['note'] != 0:
                note = nd['note']
                octave = nd['octave']
            dp = a*math.pi*2/48000  * 2**(note/12+octave)
            evol = vol if note <= 12 else 0
            for i in range(d):
                rvol = rvol*.9+vol/150
                yield nf(phase)*rvol
                phase += dp


def testri(ft,c=0,a=22,d=800,nf = math.sin):
    vol = 0
    note = 13
    phase = 0
    rvol = 1
    octave = 0
    dp = 0
    for row in ft.row_gen():
        nd = row[c]
        if nd['volume'] != None:
            vol = nd['volume']
        if nd['note'] != 0:
            note = nd['note']
            octave = nd['octave']
        dp = a*math.pi*2/48000  * 2**(note/12+octave)
        evol = vol if note <= 12 else 0
        for i in range(d):
            rvol = rvol*.9+vol/150
            yield nf(phase)*rvol
            phase += dp

CHIP_SPEED = 1789773                
RATE_PERIOD_TABLE = [428, 380, 340, 320, 286, 254, 226, 214, 190, 160, 142, 128, 106,  84,  72,  54]

def dpcmTest(ft,inst=0,d=800,cs=CHIP_SPEED/48000,ro=None):
    c = 4
    notes = ft.instruments[inst].dpcm_notes
    sa = ft.dpcm_samples
    note = -1
    ro = ft.frames.songs[0]['rows per frame'] if ro == None else ro
    dpcm = 64
    rate = 0
    loop = 0
    t = 0
    gen = (i for i in range(0))
    for fr in ft.frames[0]:
        p = fr[c]
        pat = ft.patterns[0][p]
        if c in pat:
            pat = pat[c]
        else:
            for r in range(ro):
                for i in range(d):
                    yield dpcm/64-1
            continue
        for r in range(ro):
            nd = pat[r]
            if nd['note'] != 0:
                if nd['note'] > 12:
                    note = -1
                elif nd['note'] <= 12:
                    note = nd['note']-1 + nd['octave']*12
                else:
                    note = -1
                if note >= 0:
                    s,r,i = notes[note]
                    if i != 255:
                        dpcm = i
                    if s != 0:
                        gen = sa[s-1]
                        loop = r&0x80
                        rate = cs/[428, 380, 340, 320, 286, 254, 226, 214, 190, 160, 142, 128, 106,  84,  72,  54][r&0xf]
                else:
                    gen = (i for i in range(0))
            if nd['instrument'] != 0x40:
                notes = ft.instruments[nd['instrument']].dpcm_notes
            for i in range(d):
                yield dpcm/64-1
                t += rate
                while t > 1:
                    t -= 1
                    try:
                        a = next(gen)
                        dpcm += (a*4-2) if 0 <= dpcm+(a*4-2)< 128 else 0
                    except:
                        if loop != 0:
                            s,r,i = notes[note]
                            gen = sa[s]
         



class chip_2a03:
    def __init__(self):
        self.dpcm = 64
        self.dpcm_rate = 0
        self.dpcm_time = 0
        self.dpcm_gen = None
        self.dpcm_obj = None
    def step(self,state):
        c = state['state']['2a03']
        self.dpcm_time += self.dpcm_rate
        while self.dpcm_time > 1:
            self.dpcm_time -= 1
            try:
                a = next(self.dpcm_gen)
                self.dpcm += (a*4-2) if 0 <= self.dpcm+(a*4-2)< 128 else 0
            except:
                if self.dpcm_obj != None:
                    if loop != 0:
                        self.dpcm_gen = c['DPCM'].instrument.dpcm(c['DPCM'].note_num())
                    
def t_2a03_pulse(ch,d=800,t=0,b=3,f=lambda x,w:  x*8 < [1,2,4,6][w%4]):
    while 1:
        vol = ch.out_volume if ch.play else 0
        note = ch.out_note
        w = ch.duty
        for i in range(d):
            dt = b/48000 * (2**(note/12))
            t += dt
            t %= 1
            yield f(t,w)*vol
def t_2a03_tri(ch,d=800,b=3,f=lambda x:abs(x-.5)*2-.5):
    t = 0
    while 1:
        vol = ch.out_volume if ch.play else 0
        note = ch.out_note
        for i in range(d):
            dt = b/48000 * (2**(note/12))
            t += dt
            t %= 1
            yield f(t)#*vol
def t_2a03_noise(ch,d=800,s=1):
    t = 0
    while 1:
        vol = ch.out_volume if ch.play else 0
        note = int(ch.out_note)
        w = ch.duty
        for i in range(d):
            t += CHIP_SPEED/48000/RATE_PERIOD_TABLE[note&0xf]
            out = s&1
            n = 1
            while t > 1:
                t -= 1
                if ((s^(s>>(1+5*(w&1))))&1):
                    s = s>>1|(1<<14)
                else:
                    s >>= 1
                out += s&1
                n += 1
            yield (s&1)/n*vol
def t_2a03_dpcm(ch,d=800):
    v = 64
    obj = ch.instrument.dpcm(int(ch.out_note))
    t = 0
    while 1:
        if ch.note_start:
            obj = ch.instrument.dpcm(int(ch.out_note))
            ch.note_start = 0
            #obj = {'rate':rate&0xf,'divisor':RATE_PERIOD_TABLE[rate&0xf],'loop':0!=rate&0x80,'init':init,'id':which,
            #            'samples':self.module.dpcm_samples,'sample':self.module.dpcm_samples[which]}
            if obj['init'] is not None:
                t = obj['init']
        for i in range(d):
            t += CHIP_SPEED/48000/obj['divisor']
            out = v
            n = 1
            while t > 1:
                t -= 1
                try:
                    a = next(obj['sample'])
                    v += (a*4-2) if 0 <= v+(a*4-2)< 128 else 0
                except:
                    if obj['loop']:
                        obj['sample'] = obj['samples'][obj['id']]
                out += v
                n += 1
            yield out/n/64-1

def t_vrc6_pulse(ch,d=800,b=3, f=lambda x,w:  x*16 < w+1):
    t = 0
    while 1:
        vol = ch.out_volume if ch.play else 0
        note = ch.out_note
        w = ch.duty
        for i in range(d):
            dt = b/48000 * (2**(note/12))
            t += dt
            t %= 1
            yield f(t,w)*vol
def t_vrc6_saw(ch,d=800,b=3):
    t = 0
    while 1:
        vol = ch.out_volume if ch.play else 0
        note = ch.out_note
        w = ch.duty
        for i in range(d):
            dt = b/48000 * (2**(note/12))
            t += dt
            t %= 1
            yield t*vol

def t_fds(ch,d=800,b=3):
    t = 0
    wave = [32]
    while 1:
        vol = ch.out_volume if ch.play else 0
        note = ch.out_note
        inst = ch.instrument
        try:
            wave = inst.wave
        except:
            wave = [32]
        for i in range(d):
            dt = b/48000 * (2**(note/12))
            t += dt/2
            t %= 1
            yield (wave[int(t*len(wave))]/64-.5)*vol
    
def tsts(ft,c=800,s=0,b=22,pf=lambda x,w:  x*8 < [1,2,4,6][w%4],g=None):
    g = ft.states_gen(s) if g is None else g
    s = next(g)
    def advancer():
        for i in range(c):
            yield 0
        for v in g:
            for i in range(c):
                yield 0
    r = [advancer(),t_2a03_pulse(s['state']['2A03']['pulse 1'],c,0,b,pf),t_2a03_pulse(s['state']['2A03']['pulse 2'],c,0,b,pf),
         t_2a03_tri(s['state']['2A03']['triangle'],c,b),
         t_2a03_noise(s['state']['2A03']['noise'],c),
         t_2a03_dpcm(s['state']['2A03']['DPCM'],c)]

    #cname = ['VRC6','MMC5','N163','FDS','VRC7','5B']
    #cnum = [3,2,self.params.n_n163_channels if self.params.n_n163_channels is not None else 0,1,6,3]
    #ccnames = [['pulse 1','pulse 2','sawtooth'],['pulse 1','pulse 2'],[f'Namco {i+1}' for i in range(cnum[2])],None,[f'fm {i+1}' for i in range(6)],['square 1','square 2','square 3']]
    if 'VRC6' in ft.params.chips:
        r += [t_vrc6_pulse(s['state']['VRC6']['pulse 1'],c,b),
              t_vrc6_pulse(s['state']['VRC6']['pulse 2'],c,b),
              t_vrc6_saw(s['state']['VRC6']['sawtooth'],c,b)]
    if 'FDS' in ft.params.chips:
        r += [t_fds(s['state']['FDS'],c,b)]





    return r




def v_bar(v):
    bar = " ▁▂▃▄▅▆▇█"
    return bar[round(v*8)]
def note_str(n):
    i = round(n)
    r = n-i
    return NOTES[i%12]+f"{i//12}{v_bar(r+.5)}"
def h_bar(n,l=8):
    bar = " ▏▎▍▌▋▊▉█"
    v = round(n*8*l)
    f = v//8
    r = bar[-1]*f+(bar[v%8] if f < l else "")
    return r + (bar[0]*(l-len(r)))

def center_str(s,l=8):
    if len(s) > l:
        b = len(s)//2-l//2
        return s[b:b+l]
    else:
        b = l-len(s)
        return (" "*(b//2))+s+(" "*(b-b//2))
    
def test_print(ft,steps=64,s=0,wpc=16):
    g = ft.states_gen(s)
    s = next(g)
    s = next(g)
    chs = []
    legend = []
    sublegend = []
    ks = ['2A03','VRC6','MMC5','N163','FDS','VRC7','5B']
    for k in ks:
        if k in s['state']:
            if k == 'FDS':
                chs += [s['state'][k]]
                legend += [(k,len(chs[-1].raw_effects))]
                sublegend += [("",len(chs[-1].raw_effects))]
            else:
                for c in s['state'][k]:
                    chs += [s['state'][k][c]]
                    legend += [(k,len(chs[-1].raw_effects))]
                    sublegend += [(c,len(chs[-1].raw_effects))]
    #print legend
    print("    "+"|".join((center_str(v,wpc)+"    "*e for v,e in legend)))
    print("    "+"|".join((center_str(v,wpc)+"    "*e for v,e in sublegend)))
    v = wpc//2
    nv = wpc - v-1
    rc = -1
    def fmt_cmd(c):
        return c[0] + hex(0x100|c[1])[3:] if type(c[0]) is str else "---"
    get = lambda : [" *"[c.note_start] + center_str(note_str(c.out_note),nv)+center_str(str(c.duty),2)+h_bar(c.out_volume/15,v-2) +
                    "".join((" "+ fmt_cmd(cmd) for cmd in c.raw_effects))
                    for c in chs]
    prev =  get()
    for c in chs:
        c.note_start = 0
    for i in range(steps):
        clr = "\033[0m\033[2m" if chs[0].step else ("\033[0m" if rc%4 else "\033[0m\033[1m")
        pre = clr+hex((rc&0xff)|0x100)[3:]+"."+str(chs[0].step)
        _ = next(g)
        new = get()
        for i in range(len(new)):
            if new[i][0] == "*":
                prev[i] = "\033[4m" + prev[i]
        print(pre+("\033[0m\033[2m┊"+clr).join(prev))
        prev = new
        for c in chs:
            c.note_start = 0
        rc += chs[0].step == 0
    print("\033[0m")
        
def sign_extend(n,s=7):
    if n&(1<<s):
        return n|(-1<<s)
    return n&((1<<s)-1)
def sign_split(n,largest_pos,b):
    m = 1<<b
    n &= b-1
    if n > largest_pos:
        n -= m
    return n
class reader:
    def __init__(self,b,i=0):
        #if type(b) is file:
        #    b = b.read()
        self.b = b
        self.i = i
    def copy(self):
        return reader(self.b,self.i)
    def read(self,p=None):
        if p is None:
            self.i += 1
            return self.b[self.i-1]
        d = struct.calcsize(p)
        r = struct.unpack(p,self.b[self.i:self.i+d])
        self.i += d
        return r
    def read_bytes(self,n):
        r = self.b[self.i:self.i+n]
        self.i += n
        return r
    def peek(self,p=None):
        if p is None:
            return self.b[self.i]
        d = struct.calcsize(p)
        return struct.unpack(p,self.b[self.i:self.i+d])
    def peek_bytes(self,n):
        return self.b[self.i:self.i+n]
    def skip(self,n):
        if type(n) is str:
            self.i += struct.calcsize(n)
        else:
            self.i += n
        return self
    def unskip(self,n):
        if type(n) is str:
            self.i -= struct.calcsize(n)
        else:
            self.i -= n
        return self
    def __len__(self):
        return len(self.b)-self.i
    
class params:
    def __init__(self,r,f):
        self.r = r.copy()
        header = r.read("<16sII")
        name,ver,size = header
        name = name.rstrip(b'\0')
        assert name == b'PARAMS'

        def chipsFromMask(m):
            cs = ["VRC6","VRC7","FDS","MMC5","N163","5B"]
            r = []
            for i in range(len(cs)):
                if m&1:
                    r += [cs[i]]
                m >>= 1
            return set(r)

        
        self.ver = ver
        if ver == 1:
            self.tempo,\
                self.n_channels,\
                self.PAL,\
                self.overclock = r.read("<IIII")
            self.chips = set()
            self.new_vibrato = \
                self.row_highlight_dist = \
                self.second_highlight_dist = \
                self.n_n163_channels = \
                self.speed_tempo_split = None
        else:
            self.tempo = None
            self.chips = chipsFromMask(r.read())
            self.n_channels,\
                self.PAL,\
                self.overclock,\
                self.new_vibrato,\
                self.row_highlight_dist,\
                self.second_highlight_dist,\
                self.n_n163_channels,\
                self.speed_tempo_split,*rest = *r.read("<III"+'I'*[0,1,3,3+("N163" in self.chips),4+("N163" in self.chips)][ver-2]),*([None]*10)
            if ver == 6 and "N163" not in self.chips:
                self.n_n163_channels,self.speed_tempo_split = self.speed_tempo_split,self.n_n163_channels
    def __repr__(self):
        return f"params(ver:{self.ver},chips:{self.chips},channels:{self.n_channels})"
class info:
    def __init__(self,r,f):
        self.r = r.copy()
        header = r.read("<16sII")
        name,ver,size = header
        name = name.rstrip(b'\0')
        assert name == b'INFO'
        self.ver = ver
        assert ver == 1
        title,author,cr = r.read("<32s32s32s")
        self.title     = title.rstrip(b'\0')
        self.author    = author.rstrip(b'\0')
        self.copyright = cr.rstrip(b'\0')
    def __repr__(self):
        return f"info({self.title} by {self.author}, ©{self.copyright})"
            
            
class header:
    def __init__(self,r,f):
        self.r = r.copy()
        header = r.read("<16sII")
        name,ver,size = header
        start = r.i
        name = name.rstrip(b'\0')
        assert name == b'HEADER'
        self.ver = ver
        if ver >= 2:
            self.n_tracks = r.read()+1
        else:
            self.n_tracks = 1
        if ver >= 3:
            self.track_names = []
            for i in range(self.n_tracks):
                self.track_names += [[]]
                while r.peek() != 0:
                    self.track_names[-1] += [r.read()]
                r.read()
                self.track_names[-1] = bytes(self.track_names[-1])
        else:
            self.track_names = [None]*self.n_tracks
        left = start+size-r.i
        nchannels = left//(1+self.n_tracks)
        d = r.read(f'<{left}b')
        self.track_effect_cols = [dict() for i in range(self.n_tracks)]
        self.track_effect_cols_arr = [[] for i in range(self.n_tracks)]
        self.total_effect_cols = 0
        for i in range(nchannels):
            o = i*(1+self.n_tracks)
            for j in range(self.n_tracks):
                self.track_effect_cols_arr[j] += [d[o+j+1]+1]
                self.track_effect_cols[j][d[o]] = d[o+j+1]+1
                self.total_effect_cols += d[o+j+1]+1
    def __repr__(self):
        return f"header({self.track_names},{self.track_effect_cols})"


class sequencing_inst_state:
    def __init__(self,seqs):
        self.sequences = seqs
        self.si = {k:0 for k in self.sequences.keys()}
        self.st = {k:0 for k in self.sequences.keys()}
    def __getitem__(self,k):
        if type(k) is tuple:
            if k[0] in self.si:
                c = self.sequences[k[0]]['c']
                i = self.si[k[0]]
                d = k[1]
            else:
                return k[1]
        elif k in self.si:
            c = self.sequences[k]['c']
            i = self.si[k]
            d = None
        else:
            return None
        if i < 0 or i >= len(c):
            return d
        return c[i]
    def keys(self):
        return self.si.keys()
    def tick(self):
        for k in self.si.keys():
            seq = self.sequences[k]
            rp = seq['release point']
            lp = seq['loop point']
            l = len(seq['c'])
            if l: self.st[k] += seq['c'][self.si[k]]
            #if loop is same time or after release, it loops after release
            lt = rp if rp > lp else l
            if lp != -1 and self.si[k] == lt:
                self.si[k] = lp
            elif self.si[k] == rp:
                pass
            elif self.si[k]+1 != l:
                self.si[k] += 1
    def release(self):
        for k in self.si.keys():
            seq = self.sequences[k]
            if self.si[k] < seq['release point']:
                self.si[k] = seq['release point']
    def __call__(self,attrs):
        x = attrs['x'] if 'x' in attrs else 0
        y = attrs['y'] if 'y' in attrs else 0
        fd = {'volume':{'absolute': lambda i,s,c: i*s/15},
              'arpeggio':
              {'absolute': lambda i,s,c: i+s,
               'fixed': lambda i,s,c: s,
               'relative': lambda i,s,c:i+s+c,
               'scheme': lambda i,s,c:i+sign_split(s,36,6)+[0,x,y,-y][s>>6]},
              'pitch': {'absolute': lambda i,s,c: i+s+c,
                        'fixed': lambda i,s,c: s,},
              'hi_pitch':{'absolute': lambda i,s,c: i+s+c},
              'duty':{'absolute': lambda i,s,c: s}}
        for k in attrs.keys():
            if (m:=self[k]) != None:
                seq = self.sequences[k]
                f = fd[k][seq['arp type']]
                attrs[k] = f(attrs[k],m,self.st[k])
        return attrs

class inst_2A03:
    seq_keys = ['volume','arpeggio','pitch','hi_pitch','duty']
    def __init__(self,r,ver):
        self.name = ''
        self.ver = ver
        ndef = r.read('<I')[0]
        self.seqs = [r.read('<BB') for i in range(ndef)]
        self.dpcm_notes = [r.read('<BB'+'B'*(ver >= 6)) for i in range(72 if ver == 1 else 96)]
    def __repr__(self):
        return f"inst_2A03({self.name})"
    def __call__(self,*a):
        class inst_2A03_state(sequencing_inst_state):
            def __init__(self,i,m):
                self.arp_tot_offset = 0
                self.inst = i
                self.module = m
                super().__init__({inst_2A03.seq_keys[i]:self.module.sequences[self.inst.seqs[i][1]][i] for i in range(len(i.seqs)) if self.inst.seqs[i][0]})
        
            def dpcm(self,note):
                which,rate,init = self.inst.dpcm_notes[note]
                if init == 255: init = None;
                return {'rate':rate&0xf,'divisor':RATE_PERIOD_TABLE[rate&0xf],'loop':0!=rate&0x80,'init':init,'id':which,
                        'samples':self.module.dpcm_samples,'sample':self.module.dpcm_samples[which]}
            def __repr__(self):
                return f"[2A03] {repr(self.inst.name)[2:-1]} ({self.si})"
        return inst_2A03_state(self,*a)
class inst_VRC6:
    def __init__(self,r,ver):
        self.name = ''
        self.ver = ver
        ndef = r.read('<I')[0]
        self.seqs = [r.read('<BB') for i in range(ndef)]
    def __repr__(self):
        return f"inst_VRC6({self.name})"
    def __call__(self,*a):
        class inst_VRC6_state(sequencing_inst_state):
            def __init__(self,i,m):
                self.arp_tot_offset = 0
                self.inst = i
                self.module = m
                super().__init__({inst_2A03.seq_keys[i]:
                                  self.module.sequences_vrc6[
                                      self.inst.seqs[i][1]
                                  ][i] for i in range(len(i.seqs))
                                  if self.inst.seqs[i][0]})
            def __repr__(self):
                return f"[VRC6] {repr(self.inst.name)[2:-1]} ({self.si})"
        return inst_VRC6_state(self,*a)

class inst_VRC7:
    def __init__(self,r,ver):
        self.name = ''
        self.ver = ver
        self.patch_number = r.read('<I')[0]
        self.custom_patch = r.read('<8B')
    def __repr__(self):
        return f"inst_VRC7({self.name})"
    def __call__(self,*a):
        return {'patch number':self.patch_number,'patch':self.custom_patch}
class inst_FDS:
    def __init__(self,r,ver):
        self.name = ""
        self.ver = ver
        self.wave = [r.read() for i in range(64)]
        self.mod = [r.read() for i in range(32)]
        self.mod_rate,self.mod_depth,self.mod_delay = r.read('<III')
        seqs = []
        for i in range(3):
            l = r.read()
            lp,rp,at = r.read('<iii')
            seq = [r.read() for i in range(l)]
            seqs += [{'loop point':lp,'release point':rp,'arp type':['absolute','fixed','relative','scheme'][at],'c':seq}]
        self.vol_seq,self.arp_seq,self.pitch_seq = seqs
        self.seqs = seqs
    def __repr__(self):
        return f"inst_FDS({self.name})"
    def __call__(self,*a):
        class inst_FDS_state(sequencing_inst_state):
            def __init__(self,i,m):
                self.arp_tot_offset = 0
                self.inst = i
                self.module = m
                seqs = ['volume','arpeggio','pitch']
                super().__init__({seqs[ind]:i.seqs[ind] for ind in range(3)})
                self.wave = self.inst.wave
            def __repr__(self):
                return f"[FDS] {repr(self.inst.name)[2:-1]} ({self.si})"
        return inst_FDS_state(self,*a)
class inst_N163:
    def __init__(self,r,ver):
        self.name = ""
        self.ver = ver
        ndef = r.read('<I')[0]
        self.seqs = [r.read('<BB') for i in range(ndef)]
        ws,self.wave_position,nw = r.read('<III')
        self.waves = [[r.read() for i in range(ws)] for j in range(nw)]
    def __repr__(self):
        return f"inst_N163({self.name})"
    def __call__(self,*a):
        raise "notimplemented"
            
class instruments:
    insts = [None,inst_2A03,inst_VRC6,inst_VRC7,inst_FDS,inst_N163]
    def __init__(self,r,f):
        self.r = r.copy()
        header = r.read("<16sII")
        name,ver,size = header
        start = r.i
        name = name.rstrip(b'\0')
        assert name == b'INSTRUMENTS'
        self.ver = ver
        num = r.read('<I')[0]
        self.raw_instruments = []
        self.instruments = dict()
        self.ninstruments = dict()
        for i in range(num):
            ind,t = r.read('<IB')
            inst = instruments.insts[t](r,ver)
            nl = r.read('<I')[0]
            name = bytes([r.read() for i in range(nl)])
            inst.name = name
            self.instruments[ind] = (name,inst)
            self.ninstruments[name] = (ind,inst)
    def __getitem__(self,i):
        return self.instruments[i][1]

class sequences:
    def __init__(self,r,f):
        self.r = r.copy()
        header = r.read("<16sII")
        name,ver,size = header
        start = r.i
        name = name.rstrip(b'\0')
        #assert name == b'SEQUENCES'
        self.name = name
        nseqs = r.read("<I")[0]
        seqs = []
        seqd = dict()
        ats = ['absolute','fixed','relative','scheme']
        for i in range(nseqs):
            seqi = r.read("<I")[0]
            seqt = r.read("<I")[0] if ver >= 2 else None
            seq_nruns = r.read("<B")[0] if ver <= 2 else None
            seq_runs = [r.read("<h")[0] for i in range(seq_nruns)]  if ver < 2 else None
            seq_len = r.read("<B")[0] if ver >= 3 else 0
            seq_loopPt = r.read("<i")[0] if ver >= 3 else None
            seq_relPt = r.read("<i")[0] if ver == 4 else None
            seq_arpType = r.read("<i")[0] if ver == 4 else None
            seq_content = [r.read("<B")[0] for i in range(seq_len)] if ver >= 3 else None
            seq = {'id':seqi,'type':seqt,'rle':seq_runs,'loop point':seq_loopPt,'release point':seq_relPt,'arp type':ats[seq_arpType] if ver == 4 else None,'c':seq_content}
            seqs += [(seqi,seq)]
            if seqi not in seqd:
                seqd[seqi] = dict()
            seqd[seqi][seqt] = seq
        if ver >= 5:
            for i in range(nseqs):
                seqs[i][1]['release point'],seqs[i][1]['arp type'] = r.read("<ii")
                seqs[i][1]['arp type']  = ats[seqs[i][1]['arp type'] ]
                
        self.seqd=seqd
        self.seqs = seqs
    def __getitem__(self,i):
        return self.seqd[i]
class frames:
    def __init__(self,r,f):
        nc = f[b'PARAMS'].n_channels
        self.r = r.copy()
        header = r.read("<16sII")
        name,ver,size = header
        start = r.i
        name = name.rstrip(b'\0')
        assert name == b'FRAMES'
        songs = []
        while r.i-start < size:
            nf = r.read('<I')[0]
            ss = r.read('<I')[0] if ver >= 3 else 6
            st = r.read('<I')[0] if ver >= 2 else 150
            rpf = r.read('<I')[0] if ver >= 2 else 64
            nch = r.read('<I')[0] if ver == 1 else nc
            c = [[r.read('<B')[0] for i in range(nc)] for j in range(nf)]
            songs += [{'speed':ss,'tempo':st,'rows per frame':rpf,'frames':c}]
        self.songs = songs
    def __getitem__(self,k):
        return self.songs[k]['frames']

effect_dict_li = {'0': 10, '1': 16, '2': 17, '3': 6, '4': 11, '7': 12, 'A': 22, 'B': 2, 'C': 4, 'D': 3, 'E': 5, 'F': 1, 'G': 14, 'H': 8, 'I': 9, 'J': 28, 'L': 33, 'M': 25, 'O': 34, 'P': 13, 'Q': 20, 'R': 21, 'S': 23, 'T': 35, 'V': 18, 'W': 29, 'X': 24, 'Y': 19, 'Z': 15}
effect_dict_il = {10: '0', 16: '1', 17: '2', 6: '3', 11: '4', 12: '7', 22: 'A', 2: 'B', 4: 'C', 3: 'D', 5: 'E', 1: 'F', 14: 'G', 8: 'H', 9: 'I', 28: 'J', 33: 'L', 25: 'M', 34: 'O', 13: 'P', 20: 'Q', 21: 'R', 23: 'S', 35: 'T', 18: 'V', 29: 'W', 24: 'X', 19: 'Y', 15: 'Z'}
    
class pattern:
    def __init__(self,c,n,e,v):
        self.c = c
        self.e = e
        self.v = v
        r = reader(c)
        self.rows = dict()
        for i in range(n):
            ind = r.read('<B') if v == 1 else r.read('<I')[0]
            note,octave,instrument,volume = r.read('<BBBB')
            if v == 1 and volume == 0 or v != 1 and volume == 0x10:
                volume = None
            elif v == 1:
                volume -= 1
            effects = [r.read('<BB') for j in range(e)]
            self.rows[ind] = {'note':note,'octave':octave,'instrument':instrument,'volume':volume,'neffects':effects,'effects':[(self.effect_to_l(e[0]),e[1]) for e in effects]}
    def __getitem__(self,i):
        if i in self.rows:
            return self.rows[i]
        return {'note':0,'octave':0,'instrument':0x40,'volume':None,'effects':[(self.effect_to_l(0),0)]*self.e,'neffects':[(0,0)]*self.e}
    def effect_to_l(self,e):
        #  0   1   2  3   4   7   A  B  C  D  E  F   G  H  I   J   L   M   O   P   Q   R   S   T   V   W   X   Y   Z
        #[10, 16, 17, 6, 11, 12, 22, 2, 4, 3, 5, 1, 14, 8, 9, 28, 33, 25, 34, 13, 20, 21, 23, 35, 18, 29, 24, 19, 15]
        if e in effect_dict_il:
            return effect_dict_il[e]
        return e

    
class patterns:
    def __init__(self,r,f):
        self.r = r.copy()
        header = r.read("<16sII")
        name,ver,size = header
        start = r.i
        name = name.rstrip(b'\0')
        assert name == b'PATTERNS'
        rpp = r.read("<I")[0] if ver == 1 else None
        patterns = dict()
        while r.i-start < size:
            si = r.read("<i")[0] if ver >= 2 else 0
            co,pi,dr = r.read("<III")
            ec = f[b'HEADER'].track_effect_cols_arr[si][co]
            cols = 7 if ver == 1 else 8+2*ec
            content = r.read_bytes(dr*cols)
            if si not in patterns:
                patterns[si] = dict()
            if pi not in patterns[si]:
                patterns[si][pi] = dict()
            patterns[si][pi][co] = pattern(content,dr,ec,ver)
        self.patterns = patterns
    def __getitem__(self,k):
        return self.patterns[k]
class dpcm_samples:
    def __init__(self,r,f):
        self.r = r.copy()
        header = r.read("<16sII")
        name,ver,size = header
        start = r.i
        name = name.rstrip(b'\0')
        assert name == b'DPCM SAMPLES'
        ns = r.read("<B")[0]
        s = []
        sname = dict()
        snum = dict()
        for i in range(ns):
            ind = r.read("<B")[0]
            nl = r.read("<I")[0]
            name = r.read(f"<{nl}s")[0]
            ls = r.read("<I")[0]
            c = [r.read("<B")[0] for i in range(ls)]
            s += [(ind,name,c)]
            snum[ind] = (name,c)
            sname[name] = (ind,c)

        self.samples = s
        self.name_samples = sname
        self.nsamples = snum
    def __getitem__(self,k):
        try:
            if type(k) is bytes:
                s = self.name_samples[k][1]
            else:
                s = self.nsamples[k][1]
        except:
            return
        for i in s:
            for b in range(8):
                yield i&1
                i >>= 1
    
class sequences_n163:
    def __init__(self,r,f):
        self.r = r.copy()
        header = r.read("<16sII")
        name,ver,size = header
        start = r.i
        name = name.rstrip(b'\0')
        ns = r.read('<I')
        seqs = dict()
        for i in range(ns):
            si,st,l,lp,rp,at = r.read('<IIBiii')
            sc = r.read_bytes(l)
            seqs[si] = (st,lp,rp,at,sc)
        self.seqs = seqs

class comments:
    def __init__(self,r,f):
        self.r = r.copy()
        header = r.read("<16sII")
        name,ver,size = header
        start = r.i
        name = name.rstrip(b'\0')

        self.show = r.read(b'<I')
        com = []
        while r.i-start < size:
            s = []
            while r.i-start < size and r.peek() != 0:
                s += [r.read()]
            r.read()
            com += [bytes(s)]
        self.c = com
    
    
names = {
    b'PARAMS':params,
    b'INFO':info,
    b'HEADER':header,
    b'INSTRUMENTS':instruments,
    b'SEQUENCES':sequences,
    b'FRAMES':frames,
    b'PATTERNS':patterns,
    b'DPCM SAMPLES':dpcm_samples,
    b'SEQUENCES_VRC6':sequences,
    b'SEQUENCES_N163':sequences_n163,
    b'COMMENTS':comments,
    }
NOTES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
class ftm: #todo: blocks have interblock dependencies, so pass self to the constructors and only construct in a call to getitem (cache it)
    def __init__(self,buf,v=0):
        self.dat = buf
        r = reader(buf)
        #reading time
        assert r.read('<18s')[0] == b'FamiTracker Module'
        self.raw_version = r.read("<I")[0]
        #grab block headers
        self.raw_blocks = []
        self.blocks = dict()
        self.block_locs = dict()
        while 1:
            if r.peek("<3s")[0] == b'END':
                break
            header = r.read("<16sII")
            name,ver,size = header
            name = name.rstrip(b'\0')
            self.raw_blocks += [(name,ver,size,r.copy().unskip("<16sII"))]
            self.block_locs[name] = r.copy().unskip("<16sII")
            if v:
                print("found block",name,"v",ver,",len",size)
            r.skip(size)

        self.init()
    def __getitem__(self,k):
        if k not in self.blocks:
            if k not in self.block_locs:
                return None
            if k in names:
                self.blocks[k] = names[k](self.block_locs[k].copy(),self)
            else:
                r = self.block_locs[k].copy()
                header = r.read("<16sII")
                name,ver,size = header
                self.blocks[k] = (name,ver,size,bytes((r.read() for i in range(size))))    
        return self.blocks[k]
    def init(self):
        self.params = self[b'PARAMS']
        self.info = self[b'INFO']
        self.header = self[b'HEADER']
        self.instruments = self[b'INSTRUMENTS']
        self.sequences = self[b'SEQUENCES']
        self.frames = self[b'FRAMES']
        self.patterns = self[b'PATTERNS']
        self.dpcm_samples = self[b'DPCM SAMPLES']
        self.sequences_vrc6 = self[b'SEQUENCES_VRC6']
        self.sequences_n163 = self[b'SEQUENCES_N163']
        self.comments = self[b'COMMENTS']

    def row_gen(self,song=0):
        frame = 0
        row = 0

        ro = self.frames.songs[song]['rows per frame']
        pats = self.patterns[song]

        #vols = [0]*self.params.n_channels
        empty_row = {'note':0,'octave':0,'instrument':0x40,'volume':None,'effects':[],'neffects':[]}
        while 1:
            fr = self.frames[song][frame]
            r = [pats[fr[i]][i][row] if i in pats[fr[i]] else empty_row for i in range(len(fr))]
            yield r
            #check for jump commands
            for c in (c for v in r for c in v['effects']):
                if c[0] == 'B':
                    frame = c[1]
                    break
                if c[0] == 'C':
                    return
                if c[0] == 'D':
                    row = c[1]
                    frame += 1
                    break
            else:
                row += 1
                if row >= ro:
                    row = 0
                    frame += 1
                    if frame >= len(self.frames[song]):
                        frame = 0
    def state_gen(self,song=0):
        if type(song) is int:
            rg = self.row_gen(song)
        else:
            rg = song
        speed = self.frames.songs[song]['speed']
        tempo = self.frames.songs[song]['tempo']
        
        channels = self.params.n_channels
        class channel_state:
            def __init__(self,m,c):
                self.note_start = 0
                self.channel = c
                self.module = m
                self.volume = 0
                self.note = 'cut'
                self.last_note = 0
                self.octave = 0
                self.instrument_ind = 0
                self.instrument = self.module.instruments[0](self.module)
            def u(self,v):
                inst = v['instrument']
                if inst != 0x40:
                    self.instrument_ind = inst
                    self.instrument = self.module.instruments[inst](self.module)
                n = v['note']
                if n == 0:
                    pass
                elif n <= 12:
                    if type(self.note) is int: self.last_note = self.note;
                    self.note = n-1
                    self.note_start += 1
                elif n == 13:
                    self.note = 'cut'
                elif n == 14:
                    self.note = 'release'
                    if self.instrument is not None:
                        self.instrument.release()
                    #self.instrument.release()
                if n != 0:
                    self.octave = v['octave']
                if v['volume'] is not None:
                    self.volume = v['volume']
                for cmd,arg in v['effects']:
                    pass
            def note_num(self):
                return (self.note if type(self.note) is int else self.last_note)+self.octave*12
            def tick(self):
                if self.instrument is not None:
                    self.instrument.tick()
            def __repr__(self):
                no = f're{NOTES[self.last_note]}' if self.note == 'release' else self.note if type(self.note) is str else NOTES[self.note]
                return f"ch{self.channel}({no}-{self.octave} {hex(self.volume)[2:]} {self.instrument})"
                
        states = [channel_state(self,i) for i in range(channels)]
        stateObj = dict()
        stateObj['2A03'] = {['pulse 1','pulse 2','triangle','noise','DPCM'][i]:states[i] for i in range(5)}
        cname = ['VRC6','MMC5','N163','FDS','VRC7','5B']
        cnum = [3,2,self.params.n_n163_channels if self.params.n_n163_channels is not None else 0,1,6,3]
        ccnames = [['pulse 1','pulse 2','sawtooth'],['pulse 1','pulse 2'],[f'Namco {i+1}' for i in range(cnum[2])],None,[f'fm {i+1}' for i in range(6)],['square 1','square 2','square 3']]
        o = 5
        for i in range(5):
            if cname[i] in self.params.chips:
                if ccnames[i] == None:
                    stateObj[cname[i]] = states[o]
                    o += 1
                else:
                    stateObj[cname[i]] = {ccnames[i][j]:states[o+j] for j in range(cnum[i])}
                    o += cnum[i]
        
        for row in rg:
            for i in range(channels):
                states[i].u(row[i])
                for cmd,arg in row[i]['effects']:
                    if cmd == 'F':
                        if arg >= self.params.speed_tempo_split:
                            tempo = arg
                        else:
                            speed = arg
            for tick in range(speed):
                yield {'state':stateObj,'tempo':tempo}
                for s in states:
                    s.tick()
                

    def states_gen(self,song=0,EFFECT_SCALES = None):
        if EFFECT_SCALES == None:
            EFFECT_SCALES = {'A':1/15,'4':1/16,'1':1/256,'3':1/256,'P':1/256,'Q':8/256,'7':1}
        if type(song) is int:
            rg = self.row_gen(song)
        else:
            rg = song
        speed = self.frames.songs[song]['speed']
        tempo = self.frames.songs[song]['tempo']
        
        channels = self.params.n_channels
        class channel_state:
            def __init__(self,m,c):
                self.channel = c
                self.module = m
                self.instrument_ind = 0
                self.instrument = self.module.instruments[0](self.module)

                self.raw_note = 0
                self.raw_volume = 0
                self.raw_effects = []
                self.raw_play = False
                self.trigger = False
                self.note_start = 0
                self.step = 0
                
                self.out_note = 0
                self.out_volume = 0

                self.row_delay_counter = 0
                self.row = None
                
                self.note_cut_delay = 1<<16

                self.duty = 0
                self.play = False
                self.note = 0
                self.volume = 0

                self.arp_x = 0
                self.arp_y = 0
                self.arp_i = 0

                self.vib_depth = 0
                self.vib_rate = 0
                self.vib_t = 0

                self.pitch_slide_speed = 0

                self.portamento_speed = 0

                self.note_slide_dest = None
                self.note_slide_speed = 0

                self.tremolo_speed = 0
                self.tremolo_depth = 0
                self.tremolo_t = 0

                self.volume_slide_speed = 0
            def inst_note(self,n=None):
                if n == None:
                    n = self.note
                self.instrument
            def u(self,v):
                self.row = v
                self.row_delay_counter = 0
                for cmd,arg in v['effects']:
                    if cmd == 'G':
                        self.row_delay_counter = arg
            def do_row(self,v):
                inst = v['instrument']
                self.step = 0
                if inst != 0x40:
                    self.instrument_ind = inst
                    self.instrument = self.module.instruments[inst](self.module)
                n = v['note']
                if n == 0:
                    pass
                elif n <= 12:
                    self.raw_note = n-1 + v['octave']*12
                    if self.portamento_speed == 0:
                        self.instrument = self.module.instruments[self.instrument_ind](self.module)
                        self.trigger = True
                        self.note_start += 1
                    self.raw_play = True
                    self.note_slide_dest = None
                elif n == 13:
                    self.raw_play = False
                elif n == 14:
                    self.instrument.release()
                if v['volume'] is not None:
                    self.raw_volume = v['volume']
                    self.volume = self.raw_volume

                self.note_cut_delay = 1<<16
                self.raw_effects = v['effects']
                #if self.channel == 1:
                #    print(self.raw_effects)
                for cmd,arg in v['effects']:
                    arg &= 0xff
                    if type(cmd) is not str:
                        continue
                    #if self.channel == 1:
                    #    print(cmd,hex(arg))
                    x,y = arg>>4,arg&0xf
                    if cmd == "0":
                        self.arp_x = x
                        self.arp_y = y
                        self.arp_i = 0
                    elif cmd in '12':
                        self.pitch_slide_speed = arg * (2*(cmd=="1")-1)
                        self.portamento_speed = 0
                    elif cmd == '3':
                        self.portamento_speed = arg
                        self.pitch_slide_speed = 0
                    elif cmd == '4':
                        self.vib_rate = x
                        self.vib_depth = y if x else 0
                    elif cmd == '7':
                        self.tremolo_speed = x
                        self.tremolo_depth = y if x else 0
                    elif cmd == 'A':
                        self.volume_slide_speed = x-y
                    elif cmd == 'E':
                        pass #self.raw_volume = arg
                    #elif cmd == "G":
                    #    self.row_delay_counter = arg
                    elif cmd == 'P':
                        self.note += (arg-0x80)*EFFECT_SCALES['P']
                    elif cmd in 'QR':
                        self.note_slide_speed += x*2+1
                        if self.note_slide_dest is None: self.note_slide_dest = self.raw_note
                        if (cmd=="Q"):
                            self.note_slide_dest += y
                        else:
                            self.note_slide_dest -= y
                        self.portamento_speed = 0
                        self.pitch_slide_speed = 0
                        #if self.channel == 1:
                        #    print(cmd,x,y,":",self.raw_note,self.note_slide_dest,self.note_slide_speed)
                    elif cmd == "S":
                        self.note_cut_delay = arg
                    elif cmd == "V":
                        self.duty = arg
                    
            def note_num(self):
                return (self.note if type(self.note) is int else self.last_note)+self.octave*12
            def tick(self):
                self.step += 1
                self.instrument.tick()
                if self.row_delay_counter == 0:
                    self.do_row(self.row)
                self.row_delay_counter -= 1
                self.play = self.raw_play and self.note_cut_delay > 0
                self.note_cut_delay -= 1
                
                #apply instrument
                r = self.instrument({'arpeggio':self.raw_note,'x':self.arp_x,'y':self.arp_y,
                                     'volume':self.raw_volume,
                                     'duty':self.duty,
                                     #'pitch':0,
                                     #'hi_pitch':0,
                                     })
                ks = self.instrument.keys()
                if 'arpeggio' in ks:
                    self.note = r['arpeggio']
                elif self.arp_x or self.arp_y or self.trigger:
                    d = [0,self.arp_x,self.arp_y][self.arp_i]
                    if self.arp_i and d==0:
                        self.arp_i = 0
                    else:
                        self.arp_i = (self.arp_i + 1) % 3
                    self.note = self.raw_note+d
                self.trigger = False
                if 'volume' in ks:
                    self.volume = r['volume']
                else:
                    self.volume += self.volume_slide_speed*EFFECT_SCALES['A']
                if 'duty' in ks:
                    self.duty = r['duty']
                    
                    
                self.vib_t += self.vib_rate/64
                
                self.note += self.pitch_slide_speed*EFFECT_SCALES['1']
                
                
                if self.note_slide_dest != None:
                    ss = self.note_slide_speed*EFFECT_SCALES['Q']
                    #if self.channel == 1:
                    #    print(self.note,"->",self.note_slide_dest,ss)
                    if abs(self.note-self.note_slide_dest) < ss:
                        self.note = self.note_slide_dest
                        self.note_slide_dest = None
                        self.note_slide_speed = 0
                    elif self.note_slide_dest > self.note:
                        self.note += ss
                        #self.note_slide_dest -= ss
                    else:
                        self.note -= ss
                        #self.note_slide_dest += ss
                else:
                    self.note_slide_speed = 0
                    
                if self.portamento_speed == 0:
                    self.out_note = self.note + math.sin(self.vib_t*2*math.pi)*self.vib_depth*EFFECT_SCALES['4']
                else:
                    s = abs(self.portamento_speed*EFFECT_SCALES['3'])
                    if self.out_note < self.note:
                        self.out_note += s
                        if self.out_note > self.note:
                            self.out_note = self.note
                    else:
                        self.out_note -= s
                        if self.out_note < self.note:
                            self.out_note = self.note
                    
                    
                self.tremolo_t += self.tremolo_speed/64
                self.out_volume = self.volume + math.sin(self.tremolo_t*2*math.pi)*self.tremolo_depth*EFFECT_SCALES['7']
                if self.out_volume < 0:
                    self.out_volume = 0
                if self.out_volume > 15:
                    self.out_volume = 15
            def __repr__(self):
                bar = " ▁▂▃▄▅▆▇█"
                raw_note_str = NOTES[self.raw_note%12]+f"-{self.raw_note//12}"
                d = self.out_note-round(self.out_note)
                vol = str(self.out_volume)[:5]
                actual_note_str = NOTES[round(self.out_note)%12]+f"-{round(self.out_note)//12}{(bar+'█')[int((d+.5)*len(bar))]}"
                return f"ch{self.channel}({raw_note_str} {hex(self.raw_volume)[2:]} d:{self.duty}|{actual_note_str} {vol})"
                
        states = [channel_state(self,i) for i in range(channels)]
        stateObj = dict()
        stateObj['2A03'] = {['pulse 1','pulse 2','triangle','noise','DPCM'][i]:states[i] for i in range(5)}
        cname = ['VRC6','MMC5','N163','FDS','VRC7','5B']
        cnum = [3,2,self.params.n_n163_channels if self.params.n_n163_channels is not None else 0,1,6,3]
        ccnames = [['pulse 1','pulse 2','sawtooth'],['pulse 1','pulse 2'],[f'Namco {i+1}' for i in range(cnum[2])],None,[f'fm {i+1}' for i in range(6)],['square 1','square 2','square 3']]
        o = 5
        for i in range(5):
            if cname[i] in self.params.chips:
                if ccnames[i] == None:
                    stateObj[cname[i]] = states[o]
                    o += 1
                else:
                    stateObj[cname[i]] = {ccnames[i][j]:states[o+j] for j in range(cnum[i])}
                    o += cnum[i]
        
        for row in rg:
            for i in range(channels):
                states[i].u(row[i])
                for cmd,arg in row[i]['effects']:
                    if cmd == 'F':
                        if arg >= self.params.speed_tempo_split:
                            tempo = arg
                        else:
                            speed = arg
            for tick in range(speed):
                yield {'state':stateObj,'tempo':tempo}
                for s in states:
                    s.tick()
                








#easy func
def sumf(v):       
    t = v[0]          
    for r in v[1:]:   
        t += r           
    return t
def dotf(r,v=[],default=1):
    t = 0
    for i in range(len(r)):
        if i >= len(v):
            t += default*r[i]
        else:
            t += v[i]*r[i]
    return t
def play(fname,mix=[1/15,1/15,2,1/15,2],dm=1/15):
    ft = ftm(open(fname,"rb").read())
    yield from (dotf(r,mix,dm) for r in (lambda a:([next(v) for v in a[1:]] for i in a[0]))([ (i*(1+1j) for i in v) for v in tsts(ft,800,b=32.7) ]))

