import pywt
import numpy as np
import scipy as sp



def stacked_waves_plot(waves,wavelet,t=1e301):
    from matplotlib import pyplot as plt
    #note:don't just scipy upsample the low freq waves, upsample them according to the wavelet
    fig, ax = plt.subplots()

    sig = pywt.waverec(waves,wavelet)
    
    y = .9
    for i in range(len(waves)-1,-1,-1):
        #plot the samples
        h = .8/(1<<(len(waves)-i))
        axx = plt.axes([0.05,y-h , 0.90, h])
        y -= h
        w = waves[i]
        step_size = (len(sig)/len(w))
        if step_size > t:
            axx.stem(np.arange(len(w))*step_size,w,"-",".")
        #plot the interpolated
        ws = [waves[j]*(j==i) for j in range(len(waves))]
        resig = pywt.waverec(ws,wavelet)
        axx.plot(np.arange(len(resig)),resig,linewidth=0.5)
    
    plt.show(block=0)
    

def waveogram_plot(data,l=4096,decimation_factor=2**(1/12),levels=128,overlap=1,gamma=0.5):
    from matplotlib import pyplot as plt
    window = (1-np.cos(np.arange(l)*np.pi*2/l))/2
    im_sect_len = sp.fft.next_fast_len((l//2 - (int(l//2 / (decimation_factor**overlap))&-2))*2)
    img = np.zeros((im_sect_len*((len(data))//l),levels),dtype=complex)
    ii = 0
    for i in range(0,len(data)-l,l//2):
        wd = window*data[i:i+l]
        ft = sp.fft.fft(wd)
        f = l//2
        pf = [f]*overlap
        for j in range(levels):
            f /= decimation_factor
            i_f = int(f)&-2
            p_f = pf[-overlap]
            
            wft = np.concatenate((ft[i_f:p_f],np.zeros(im_sect_len-(p_f-i_f)*2),ft[-p_f-1:-i_f-1]))

            wsig = sp.fft.ifft(wft)
            img[int(ii):int(ii)+im_sect_len,j] += wsig
            
            pf += [i_f]
        ii += im_sect_len/2

    
    fig, ax = plt.subplots()
    plt.imshow(abs(img.T)**gamma,aspect='auto')
    plt.show(block=0)

def waveogram_windowless_plot(data,decimation_factor=2**(1/12),levels=128,gamma=0.5):
    from matplotlib import pyplot as plt
    ft = sp.fft.fft(data,sp.fft.next_fast_len(len(data)*2))
    l = len(ft)
    im_width = sp.fft.next_fast_len((l//2 - (int(l//2 / (decimation_factor))))*2)
    img = np.zeros((im_width//2,levels),dtype=complex)
    f = l//2
    pf = f
    for j in range(levels):
        f /= decimation_factor
        i_f = int(f)&-2
        p_f = pf
        
        wft = np.concatenate((ft[i_f:p_f],np.zeros(im_width-(p_f-i_f)*2),ft[-p_f-1:-i_f-1]))
        
        wsig = sp.fft.ifft(wft)
        img[:,j] += wsig[:im_width//2]
        
        pf = i_f
        
    
    fig, ax = plt.subplots()
    plt.imshow(abs(img.T)**gamma,aspect='auto')
    plt.show(block=0)



    
def cwave(data,wavelet="morl",sf=2**(1/12),num=128,gamma=.75):
    m,f = pywt.cwt(data,sf**np.arange(num),wavelet,method='fft')
    
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    plt.imshow(abs(m)**gamma,aspect='auto')
    plt.show(block=0)
