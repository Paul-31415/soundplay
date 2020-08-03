

def keysToPitches(s,offset=-9,spaces=False):
    notes = "awsedftgyhujkolp;']"
    capnotes = "AWSEDFTGYHUJKOLP:\"}"
    n = []
    for c in s:
        if c in notes:
            n += [(notes.index(c)+offset,False)]
        elif c in capnotes:
            n += [(capnotes.index(c)+offset,True)]
        elif c == "z":
            offset -= 12
        elif c == "x":
            offset += 12
        elif c == "Z":
            offset -= 1
        elif c == "X":
            offset += 1
        elif c == "1":
            offset -= 1/4
        elif c == "2":
            offset += 1/4
        elif c == "!":
            offset -= 1/16
        elif c == "@":
            offset += 1/16
        elif c == "(":
            yield .5
        elif c == ")":
            yield 2
        if c == " " or not spaces:
            yield n
            n = []
        elif c == "-":
            yield n
    yield n
def tri(t):
    return (abs(((t-.25)%1)-.5)-.25)*(4+4j)

def pitchToFreq(p,sr=48000):
    return 440*(2**(p/12))

def notesPlay(s,spaces=1,t=.25,lagato=1,nf=tri,sr=48000):
    g = keysToPitches(s,-9,spaces)
    oscs = []
    for ns in g:
        if type(ns) == type([]):
            if lagato:
                oscs = (oscs+[0 for n in ns])[:len(ns)]
            else:
                oscs = [0 for n in ns]
            rates = [pitchToFreq(n[0])/sr for n in ns]
            for s in range(int(t*sr)):
                r = sum((nf(i) for i in oscs))
                yield r
                for i in range(len(oscs)):
                    oscs[i] += rates[i]
        else:
            t *= ns
def splay(ss,s=1,t=.25,l=1,nf=tri,sr=48000):
    gens = [notesPlay(n,s,t,l,nf,sr) for n in ss]
    for v in gens[0]:
        for i in range(1,len(gens)):
            v += next(gens[i])
        yield v
        
m = "a a k  g    t f e f e f g zu u xk  g    t f e s e s a s e zy y xk  zy y xk  zyxk  zyxk  zyxk  zyxk jzgxgs-- jgslzjx---- p- l- k- l- a a k "
eh = "aeg--- zgul--- fyk-g-y-j kp  kp  glj  gjl"
c = "jzgxgs-- jgslzjx----g-p--l--k--l- agekxaegk-u-y-g-f-e-s-az zaxagek-u-y-g-f-e-s-a sfy-e-f-g-y-u-k-l sfyjl[  sfyjl[  efyjp[  efyjp[  aegzagzaaaaxxk"

cd = "(a s d f g ( )g ( )g ( )g h g )  (g h g f d ( )d ( )d ( )f d a )  "
ce = "x=zgx- d s (a--  )a s- a s a s- d- zhx- d s (a--  )a s- a s d s- a- zgx- d s (a--  )a s- a s a s- d- g- f d (a--  )a (s--  )s d- s- a-"
cc = "z=)))gk; gjl hk; fhk gk; gjl hk; fhk ((("
