
import struct

def pad(b,s):
    return b+bytes(s-len(b))

class ArrayFile:
    def __init__(self,f,o=0):
        if type(f) is str:
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

    def __repr__(self):
        return "array"+repr(self.file)




class OCC:
    def __init__(self,arraylike):
        if type(arraylike) is str:
            arraylike = ArrayFile(arraylike)
        self.dat = arraylike
    def __repr__(self):
        try:
            r = repr(self.dat)
            if len(r)>40:
                r = r[:40-3]+'...'
            return "<0CC-ftm reader of "+repr(self.dat)+", format="+hex(self.format)+">"
        except:
            try:
                r = repr(self.dat)
                if len(r)>40:
                    r = r[:40-3]+'...'
                return "<0CC-ftm reader of "+repr(self.dat)+">"
            except:
                return "<0CC-ftm reader>"
    def __call__(self):
        magic,self.format = self.dat.unpack('<18sI',0)
        if magic != b"FamiTracker Module":
            print("warning: magic number doesn't match!")
            print("expected: b'FamiTracker Module'")
            print("got:",magic)
            assert False
        self.blocks = []
        self.block_dict = {}
        a = self.dat.shifted(18+4)
        while 1:
            if a[:3] == b'END':
                break
            name,ver,length = a.unpack("<16sII")
            self.blocks += [(name,ver,a)]
            self.block_dict[name.rstrip(b'\0')] = a
            a = a.shifted(length+16+4+4)

        self.params = OCCParams(self.block_dict[b'PARAMS'])
        self.info = OCCInfo(self.block_dict[b'INFO'])
        self.header = OCCHeader(self.block_dict[b'HEADER'])
class OCCParams:
    def __init__(self,arraylike):
        if type(arraylike) is str:
            arraylike = ArrayFile(arraylike)
        self.dat = arraylike
    def __repr__(self):
        try:
            return "<PARAMS v="+str(self.version)+">"
        except:
            try:
                return "<PARAMS at "+str(self.dat.offset)+">"
            except:
                return "<PARAMS>"
            
    def __call__(self):
        name,self.version,length = self.dat.unpack("<16sII")


class OCCInfo:
    def __init__(self,arraylike):
        if type(arraylike) is str:
            arraylike = ArrayFile(arraylike)
        self.dat = arraylike
    def __repr__(self):
        try:
            return "<INFO v="+str(self.version)+", "+str(self.title)+" by "+str(self.author)+",©"+str(self.copyright)+">"
        except:
            try:
                return "<INFO v="+str(self.version)+">"
            except:
                try:
                    return "<INFO at "+str(self.dat.offset)+">"
                except:
                    return "<INFO>"
            
    def __call__(self):
        name,self.version,length = self.dat.unpack("<16sII")
        t,a,c = self.dat.unpack("<32s32s32s",16+4+4)
        self.title = t.rstrip(b'\0')
        self.author = a.rstrip(b'\0')
        self.copyright = c.rstrip(b'\0')

class OCCHeader:
    def __init__(self,arraylike):
        if type(arraylike) is str:
            arraylike = ArrayFile(arraylike)
        self.dat = arraylike
    def __repr__(self):
        try:
            return "<HEADER v="+str(self.version)+">"
        except:
            try:
                return "<HEADER at "+str(self.dat.offset)+">"
            except:
                return "<HEADER>"
            
    def __call__(self):
        name,self.version,length = self.dat.unpack("<16sII")

