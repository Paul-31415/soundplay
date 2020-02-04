xs = [8591]

def sbc(a,b):
    return (a&0xff)+((-b-((a>>8)&1))&0xff)
def adc(a,b):
    return (a&0xff)+(b&0xff)+((a>>8)&1)
def rla(a):
    return ((a<<1)|((a>>8)&1))
def rra(a):
    return (((a&0x1ff)>>1)|((a&1)*0x100))
def rrca(a):
    return (((a&0xff)>>1)|((a&1)*0x180));
def rlca(a):
    return ((a<<1)|((a>>7)&1));
def add(a,b):
    return (a&0xff)+(b&0xff)
def sub(a,b):
    return (a&0xff)+((-b)&0xff)
def cp(a,o):
    return (a&0xff)|(sub(a,o)&0x100)

def r16(x=8591):
    s = [0,0]
    while 1:
        yield s[0]
        r0 = s[0]
        r1 = s[1]
        r1 = sub(rrca((r0^0xff)-1),r1)
        r0 = (adc(cp(r1,x&0xff),r0)^(x>>8))
        s[0] = r0&0xff
        s[1] = r1&0xff


"""


def,rand(seed):
   def,rot(n):
,,,,,,,,return,(n,>>,1),|,(n,<<,7)
,,,,v,=,[seed,&,255,,(seed,<<,8),&,255,,seed,<<,16]
,,,,while,True:
,,,,,,,,
,,,,,,,,v[0],=,(v[0],+,1),&,255
,,,,,,,,v[1],^=,(v[0],&,v[2],+,1),&,255
,,,,,,,,v[2],^=,rot(v[0],+,v[1]),&,255
,,,,,,,,v[0],^=,(v[1],+,v[2],-,1),&,255
,,,,,,,,yield,(v[0]/255)*2,-,1



"""
