def brailleGetByte(c):
    return ord(c)-ord("⠀")
    
def brailleSet(c,y,x):
    return chr(ord("⠀")+(ord(c)-ord("⠀"))|(1<<[0,1,2,6,3,4,5,7][4*y+x]))

braillePoses = [[0,3],[1,4],[2,5],[6,7]]

def brailleScreen(w,h):
    return [["⠀" for i in range(w//2)] for i in range(h//4)]

def brailleScreenGet(s,x,y):
    return brailleGetByte(s[y//4][x//2])&(1<<braillePoses[y%4][x%2]) != 0
def brailleScreenSet(s,x,y,v=1):
    s[y//4][x//2] = chr(ord("⠀")+(brailleGetByte(s[y//4][x//2])&~(1<<braillePoses[y%4][x%2]))|(v<<braillePoses[y%4][x%2]))
def brailleScreenPrint(s):
    print('\n'.join([''.join(row) for row in s]))

def oscope(f,low=0,hi=1,steps=256,s=.5+.5j,m=.5,w=40,h=20):
    scrn = brailleScreen(w*2,h*4)
    for i in range(steps):
        t = low+i*(hi-low)/steps
        v = f(t).conjugate()*m+s
        if 0<=int(v.real*w*2)<w*2 and 0<=int(v.imag*h*4) < h*4:
            brailleScreenSet(scrn,int(v.real*w*2),int(v.imag*h*4))
    brailleScreenPrint(scrn)

def hgraph(f,xl=0,xh=1,steps=1024,yl=-1,yh=1,w=40,h=20):
    scrn = brailleScreen(w*2,h*4)
    for i in range(steps):
        x = xl+i*(xh-xl)/steps
        xi = int(x*h)
        xih = int(((x*h)-xi)*2)
        vals = f(x)
        try:
            len(vals)
        except:
            vals = (vals,)
        for val in vals:
            y = (val-yl)/(yh-yl)
            if 0<=y<1:
                yi = int(y*w)
                yih = int(((y*w)-yi)*2)
                if 0<=int(yi*2+yih)<w*2 and 0<=int(xi*4+xih) < h*4:
                    brailleScreenSet(scrn,yi*2+yih,xi*4+xih)
    brailleScreenPrint(scrn)

    
def graph(f,xl=0,xh=4,yl=-2,yh=2,w=80,h=24):
    for xi in range(h):
        row = ["⠀"]*w
        for xih in range(4):
            x = (xi+xih/4)*(xh-xl)/h+xl
            vals = f(x)
            try:
                len(vals)
            except:
                vals = (vals,)
            for val in vals:
                y = (val-yl)/(yh-yl)
                if 0<=y<1:
                    yi = int(y*w)
                    yih = int(((y*w)-yi)*2)
                    row[yi] = brailleSet(row[yi],yih,xih)
        print("".join(row))

def graphLT(f,xl=0,xh=4,yl=-2,yh=2,w=80,h=24):
    for xi in range(h):
        row = ["⠀"]*w
        for xih in range(4):
            x = (xi+xih/4)*(xh-xl)/h+xl
            val = f(x)
            y = (val-yl)/(yh-yl)
            if 0<=y<1:
                for yi in range(int(y*w)+1):
                    for yih in range(2):
                        yv = (yi+yih/2)/w
                        if yv <= y:
                            row[yi] = brailleSet(row[yi],yih,xih)
        print("".join(row))
def lgraph(f,xl=0,xh=4,yl=-2,yh=2,w=80):
    res = ""
    for xi in range(w):
        c = "⠀"
        for xih in range(2):
            x = (xi+xih/2)*(xh-xl)/w+xl
            vals = f(x)
            try:
                len(vals)
            except:
                vals = (vals,)
            for val in vals:
                y = (val-yl)/(yh-yl)
                if 0<=y<1:
                    yih = int(y*4)
                    c = brailleSet(c,xih,3-yih)
        res += c
    return res



        
def graphL(f,xl=0,xh=4,yl=-2,yh=2,w=80,h=24):
    prev = f(xl)
    for xi in range(h):
        row = ["⠀"]*w
        for xih in range(4):
            x = (xi+xih/4)*(xh-xl)/h+xl
            val = f(x)
            low = min(prev,val)
            high = max(prev,val)
            for yi in range(w):
                for yih in range(2):
                    y = (yi+yih/2)*(yh-yl)/w+yl
                    if low<=y<high+(yh-yl)/w/2:
                        row[yi] = brailleSet(row[yi],yih,xih)
            prev = val
        print("".join(row))
                         
def graphI(f,xl=0,xh=4,yl=-2,yh=2,w=80,h=24):
    for xi in range(h):
        row = ["⠀"]*w
        for xih in range(4):
            x = (xi+xih/4)*(xh-xl)/h+xl
            for yi in range(w):
                for yih in range(2):
                    y = (yi+yih/2)*(yh-yl)/w+yl
                    if f(x,y):
                        row[yi] = brailleSet(row[yi],yih,xih)
        print("".join(row))
                        
                                                                                                                                
