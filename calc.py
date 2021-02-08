from prng import rrca8,rand8,rand16,rand24,vrand16



def ohf(g,nb=1,grey=1,rev=1):
    ar = [i for i in range(1<<nb)]
    if rev == 1:
        for i in range(len(ar)):
            ar[i] = eval("0b"+bin((1<<nb)+ar[i])[:2:-1])
    if grey == 1:
        for i in range(len(ar)):
            ar[i] ^= ar[i]>>1
    for i in range(len(ar)):
        ar[i] <<= 8-nb
    for i in g:
        for j in ar:
            yield i^j

def from8(v):
    return ((int(v)%256)-127.5)/127.5
def to8(v):
    return int(v*127.5+127.5)%256

