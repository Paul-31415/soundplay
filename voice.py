from filters import IIR, polymult

testIPA = "ðə beɪʒ hju ɑn ðə ˈwɔtərz ʌv ðə lɑk ɪmˈprɛst ɔl, ɪnˈkludɪŋ ðə frɛnʧ kwin, bɪˈfɔr ʃi hɜrd ðæt ˈsɪmfəni əˈgɛn, ʤʌst æz jʌŋ ˈɑrθər ˈwɑntəd."

vowelTest = "aeiou"

#tongue, lips, jaw
vowels = {
    "a":(0,0,0),
    "e":(1,0,0),
    "i":(1,0,1),
    "o":(0,1,0),
    "u":(0,1,1),
}

"""consonants = {
    "g":(
    "d":
    "b":
"""

def l(g,f):
    for i in g:
        yield f(i)

def tri(s):
    while 1:
        for i in range(s):
            yield i/s

def testVoiceFunc(p,f):
    return f.setPolys(f.rootPair(p[0]*440+220,.99),polymult(f.rootPair(440,1.01+p[1]*.1),f.rootPair(440+440*p[2],1.1-.09*p[2])))



def testVoice(s,time=.25,pitch=220,volume=.2,sr=48000,func = testVoiceFunc):
    gen = l(tri(sr//pitch),lambda x: x<volume)
    filt = IIR()
    out = filt(gen)
    for c in s:
        p = vowels[c]
        func(p,filt)
        for i in range(int(sr*time)):
            yield next(out)

class testVoiceSource:
    def __init__(self):
        pass

class testVoiceFilter:
    def __init__(self):
        pass
    
class testVoiceUtterance:
    def __init__(self,phonemes):
        self.phonemes = phonemes
    def __iter__(self):
        pass



            
class Voice:
    def __init__(self,gen,filt):
        self.gen = gen
        self.filt = filt
    def __call__(self,utterance):
        pass
    
