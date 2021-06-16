import math
import struct
class halffloat:
    has_inf_and_nan = True
    emask = 0x1f
    eprec = 5
    ebias = 0xf
    mprec = 10
    mmask = 0x3ff
    mask = 0xffff
    def __init__(self,v,literal=0):
        if literal:
            assert type(v) is int
            self.v = v&self.mask
        elif type(v) is halffloat:
            self.v = v.v
        else:
            self.v = self.from_float(float(v))

    @staticmethod
    def construct(*a):
        return halffloat(*a)
    @staticmethod
    def same_type(o):
        return type(o) is halffloat
    def arg_is_sub(self,o):
        return issubclass(type(o),type(self))
    def arg_is_sup(self,o):
        return issubclass(type(self),type(o))
    def sign(self):
        return self.v>>(self.eprec+self.mprec)
    def exponent(self):
        return (self.v>>self.mprec)&self.emask
    def mantissa(self):
        return self.v&self.mmask
    def em_mask(self):
        return (self.emask<<self.mprec)|self.mmask
    def is_inf(self):
        return self.has_inf_and_nan and self.exponent() == self.emask and self.mantissa() == 0
    def is_nan(self):
        return self.has_inf_and_nan and self.exponent() == self.emask and self.mantissa()
    def lower(self):
        r = self.construct(self.v+self.sign()*2-1,1)
        if self.sign() != r.sign():
            if self.sign():
                return self
            return self.construct(-0.)
        return r
    def upper(self):
        r = self.construct(self.v-self.sign()*2+1,1)
        if not self.sign() and r.sign():
            if r.sign():
                return self
            return self.construct(0.)
        return r

    def to_float(self,v):
        e = self.exponent()
        m = (self.mantissa() | ((e!=0)<<self.mprec))<<(e==0)
        return (1-2*self.sign())*m/(1<<self.mprec) * 2**(e-self.ebias)            
    def __float__(self):
        if self.is_inf() or self.is_nan():
            return struct.unpack('<f',struct.pack('<I',(self.sign()<<31)|(0xff<<23)|self.mantissa()))[0]
        return self.to_float(self.v)
    def from_float(self,v):
        fv = struct.unpack('<I',struct.pack('<f',v))[0]
        s = fv>>31
        e = (fv>>23)&0xff
        if self.has_inf_and_nan and e == 0xff and fv&0x7fffff:#nan
            return (s<<(self.mprec+self.eprec))|self.em_mask()
        else:
            m = (fv&0x7fffff | ((e != 0) << 23))
            #print(hex(m),hex(e),hex(fv))
            e += self.ebias-0x7f
            
            if e <= 0:
                m >>= 1-e
                e = 0
            elif e > self.emask:
                e = self.emask
                m = 0 if self.has_inf_and_nan else 0x7fffff

            rem = m
            m >>= 23-self.mprec
            rem -= m << (23-self.mprec)
            #round ties to even
            v = (s<<(self.mprec+self.eprec))|(e<<self.mprec)|(m&self.mmask)
            #print(hex(rem),m&1)
            if rem > (1<<(22-self.mprec)) - (m&1) and not (self.has_inf_and_nan and e == self.emask) and not self.em_mask()&v == self.em_mask():
                v += 1
            return v
    
    def m_base_conv(self,dl=0,dh=1,base=10):
        g = float(abs(self))
        l = (float(abs(self).lower())+g)/2
        if abs(self).upper().is_inf() or abs(self).upper().v == self.v:
            h = g+g-l
        else:
            h = (float(abs(self).upper())+g)/2

        #print(l,h,dl,dh)
        while not (l <= dl <= h and l <= dh <= h):
            s = dh-dl
            d = min(max(0,int(base*(g-dl)/s)),base-1)
            if d+1 < base and dl+s*(d+1)/base < h:
                nl = dl+d*s/base
                nm = nl+s/base
                if nm-l <= h-nm and nl <= l:
                    d += 1
            dl = dl+(d/base)*s
            dh = dl + s/base
            #print(d,dl,dh,s)
            yield d
    def num_str(self,base = '0123456789'):
        s = ['','-'][self.sign()]
        if self.is_inf():
            return s+'inf'
        if self.is_nan():
            return s+'nan'
        if self.v&self.em_mask() == 0:
            return s+base[0]
        e = self.exponent()-self.ebias
        hi = int(1+math.log(2)*e/math.log(len(base)))+1
        h = len(base)**hi
        g = self.m_base_conv(0,h,len(base))
        r = ''
        for v in g:
            hi -= 1
            if v != 0:
                r += base[v]
                break
        
        if e > self.mprec or e <= -self.mprec:
            #scientific notation
            s += r+'.'
            for v in g:
                s += base[v]
            return s.rstrip(base[0])+'e'+['+','-'][hi<0]+str(abs(hi))
        elif hi < 0:
            return (s+base[0]+'.'+base[0]*(-hi-1)+r+''.join((base[v] for v in g))).rstrip(base[0])
        else:
            s += r
            for v in g:
                if hi == 0:
                    s += '.'
                s += base[v]
                hi -= 1
            return s.rstrip(base[0])
            
    def __repr__(self):
        return 'h'+self.num_str()
    def __str__(self):
        return repr(self)

    #dunders and helpers
    def __abs__(self):
        return self.construct(self.v&self.em_mask(),1)
    def __neg__(self):
        return self.construct(self.v^(1<<(self.eprec+self.mprec)),1)
    def __pos__(self):
        return self
    def __inv__(self):
        return self.construct(self.v^self.mask,1)
    def __bool__(self):
        return bool(self.v&self.em_mask())
    def __int__(self):
        return int(float(self))
    def __complex__(self):
        return complex(float(self))
    def order(self):
        if self.sign():
            return (-self.mask-1)|(self.v^self.em_mask())
        else:
            return self.v
    #math binops
    def __add__(self,o):
        r = float(self)+float(o)
        if self.arg_is_sub(o):
            return self.construct(r)
        if self.arg_is_sup(o):
            return o.construct(r)
        return r
    def __radd__(self,o):
        r = float(o)+float(self)
        if self.arg_is_sub(o):
            return self.construct(r)
        if self.arg_is_sup(o):
            return o.construct(r)
        return r
    def __sub__(self,o):
        r = float(self)-float(o)
        if self.arg_is_sub(o):
            return self.construct(r)
        if self.arg_is_sup(o):
            return o.construct(r)
        return r
    def __rsub__(self,o):
        r = float(o)-float(self)
        if self.arg_is_sub(o):
            return self.construct(r)
        if self.arg_is_sup(o):
            return o.construct(r)
        return r
    def __mul__(self,o):
        r = float(self)*float(o)
        if self.arg_is_sub(o):
            return self.construct(r)
        if self.arg_is_sup(o):
            return o.construct(r)
        return r
    def __rmul__(self,o):
        r = float(o)*float(self)
        if self.arg_is_sub(o):
            return self.construct(r)
        if self.arg_is_sup(o):
            return o.construct(r)
        return r
    def __truediv__(self,o):
        r = float(self)/float(o)
        if self.arg_is_sub(o):
            return self.construct(r)
        if self.arg_is_sup(o):
            return o.construct(r)
        return r
    def __rtruediv__(self,o):
        r = float(o)/float(self)
        if self.arg_is_sub(o):
            return self.construct(r)
        if self.arg_is_sup(o):
            return o.construct(r)
        return r
    def __floordiv__(self,o):
        r = float(self)//float(o)
        if self.arg_is_sub(o):
            return self.construct(r)
        if self.arg_is_sup(o):
            return o.construct(r)
        return r
    def __rfloordiv__(self,o):
        r = float(o)//float(self)
        if self.arg_is_sub(o):
            return self.construct(r)
        if self.arg_is_sup(o):
            return o.construct(r)
        return r
    def __mod__(self,o):
        r = float(self)//float(o)
        if self.arg_is_sub(o):
            return self.construct(r)
        if self.arg_is_sup(o):
            return o.construct(r)
        return r
    def __rmod__(self,o):
        r = float(o)//float(self)
        if self.arg_is_sub(o):
            return self.construct(r)
        if self.arg_is_sup(o):
            return o.construct(r)
        return r
    def __pow__(self,o):
        r = float(self)**float(o)
        if self.arg_is_sub(o):
            return self.construct(r)
        if self.arg_is_sup(o):
            return o.construct(r)
        return r
    def __rpow__(self,o):
        r = float(o)**float(self)
        if self.arg_is_sub(o):
            return self.construct(r)
        if self.arg_is_sup(o):
            return o.construct(r)
        return r
    #bitwize stuff
    def __and__(self,o):
        assert self.same_type(o)
        return self.construct(self.v&o.v,1)
    def __rand__(self,o):
        assert self.same_type(o)
        return self.construct(o.v&self.v,1)
    def __or__(self,o):
        assert self.same_type(o)
        return self.construct(self.v|o.v,1)
    def __ror__(self,o):
        assert self.same_type(o)
        return self.construct(o.v|self.v,1)
    def __xor__(self,o):
        assert self.same_type(o)
        return self.construct(self.v^o.v,1)
    def __rxor__(self,o):
        assert self.same_type(o)
        return self.construct(o.v^self.v,1)
        
    
        
    #comparisons
    def __lt__(self,o):
        if self.same_type(o):
            return self.order() < o.order()
        return float(self) < float(o)
    def __gt__(self,o):
        if self.same_type(o):
            return self.order() > o.order()
        return float(self) > float(o)
    def __le__(self,o):
        if self.same_type(o):
            return self.order() <= o.order()
        return float(self) <= float(o)
    def __ge__(self,o):
        if self.same_type(o):
            return self.order() >= o.order()
        return float(self) >= float(o)
    def __eq__(self,o):
        if self.same_type(o):
            return self.v == o.v
        return float(self) == float(o)
    def __ne__(self,o):
        if self.same_type(o):
            return self.v != o.v
        return float(self) != float(o)



class minifloat(halffloat):
    has_inf_and_nan = False
    emask = 0x7
    eprec = 3
    ebias = 0x3
    mprec = 4
    mmask = 0xf
    mask = 0xff
    def __init__(self,v,literal=0):
        if literal:
            assert type(v) is int
            self.v = v&self.mask
        elif type(v) is minifloat:
            self.v = v.v
        else:
            self.v = self.from_float(float(v))

    @staticmethod
    def construct(*a):
        return minifloat(*a)
    @staticmethod
    def same_type(o):
        return type(o) is minifloat
    def __repr__(self):
        return 'm'+self.num_str()

def nibblefloat_(mp):
    class nibblefloat(minifloat):
        has_inf_and_nan = False
        mmask = (1<<mp) - 1
        mprec = mp
        emask = (1<<(3-mp))-1
        eprec = 3-mp
        ebias = (1<<(2-mp))-1
        mask = 0xf
        def __init__(self,v,literal=0):
            if literal:
                assert type(v) is int
                self.v = v&self.mask
            elif type(v) is nibblefloat:
                self.v = v.v
            else:
                self.v = self.from_float(float(v))

        @staticmethod
        def construct(*a):
            return nibblefloat(*a)
        @staticmethod
        def same_type(o):
            return type(o) is nibblefloat
        def __repr__(self):
            return 'n'+str(self.mprec)+"_"+self.num_str()
    return nibblefloat
nf0 = nibblefloat_(0)
nf1 = nibblefloat_(1)
nf2 = nibblefloat_(2)

def h(*a):
    return halffloat(*a)
def m(*a):
    return minifloat(*a)
def n0(*a):
    return nf0(*a)
def n1(*a):
    return nf1(*a)
def n2(*a):
    return nf2(*a)
