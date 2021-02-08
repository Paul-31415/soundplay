def rrca8(v):
    return ((v>>1)&0xff)|((v<<7)&0xff)
def rand8():
    v = 0
    while 1:
        v ^= rrca8(v-37)
        yield v
def rand16():
    v = [0,0]
    while 1:
        v[1] = (rrca8((v[0]^0xff)-1)+v[1])&0xff
        v[0] = ((v[1]-v[0])^162)&0xff
        yield v[1]

def vrand16():
    v = [0,0]
    while 1:
        v[1] = (rrca8((v[0]^0xff)-1)+v[1])&0xff
        v[0] = ((v[1]-v[0])^162)&0xff
        yield v[0]-v[1]
        
def rand24():
    v = [0,0,0]
    while 1:
        v[2] = (v[2]+v[0])&0xff
        v[1] = ((rrca8((v[0]^0xff)-1)+v[1])^v[2])&0xff
        v[0] = ((v[1]-v[0])^124)&0xff
        yield v[2]



def rand():
    for i in rand24():
        yield i/256
