

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
    cs = ["VRC6","VRC7","FDS","MMC5","N163","Sunsoft 5B"]
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
