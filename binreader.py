

def reverse_byte(b):
    #12345678
    m = ((b>>1)^b)&0x55
    b ^= m*3
    #21436587
    m = ((b>>2)^b)&0x33
    b ^= m*5
    #43218765
    return ((b&0xf)<<4)|(b>>4)
class BinReader:
    def __init__(self,f,backtrack = 16):
        if type(f) == str:
            self.file = open(f,'rb')
        else:
            self.file = f
        self.lastByte = 0
        self.bitOffset = 8
        self.bitLittleEndian = True
        self.byteLittleEndian = True
    def readBit(self):
        if self.bitOffset == 8:
            self.bitOffset = 0
            self.lastByte = self.file.read(1)[0]
            if not self.bitLittleEndian:
                self.lastByte = reverse_byte(self.lastByte)
        #now gib bit
        r = self.lastByte >> self.bitOffset
        self.bitOffset += 1
        return r&1
    def readBits(self,b=8):
        r = 0
        o = 0
        while b:
            s = min(b,8-self.bitOffset)
            if self.byteLittleEndian:
                r |= ((self.lastByte>>self.bitOffset)&((1<<s)-1))<<o
                o += s
            else:
                r = (r<<s) | ((self.lastByte>>self.bitOffset)&((1<<s)-1))
            b -= s
            self.bitOffset += s
            if self.bitOffset == 8:
                self.bitOffset = 0
                self.lastByte = self.file.read(1)[0]
                if not self.bitLittleEndian:
                    self.lastByte = reverse_byte(self.lastByte)
        return r
    def readBytes(self,b = 4):
        if b == 0:
            return bytearray(0)
        start = bytearray(0)
        if self.bitOffset == 0:
            start = bytearray(1)
            start[0] = self.lastByte if self.bitLittleEndian else reverse_byte(self.lastByte)
        rest = bytearray(b-len(start))
        self.file.readinto(rest)
        self.bitOffset = 8
        if self.byteLittleEndian:
            return start+rest
        return (start+rest)[::-1]
    def readUInt(self,b=32):
        return self.readBits(b)
    def readInt(self,b=32):
        m = 1<<(b-1)
        r = self.readBits(b)
        return (r&(m-1))-(r&m)
    def seek(self,bitPos=0):
        self.file.seek(bitPos//8)
        self.bitOffset = bitPos%8
        if self.bitOffset:
            self.lastByte = self.file.read(1)[0]
            if not self.bitLittleEndian:
                self.lastByte = reverse_byte(self.lastByte)
        else:
            self.bitOffset = 8
    def tell(self):
        return self.file.tell()*8-8+self.bitOffset
