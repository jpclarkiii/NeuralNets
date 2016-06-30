import numpy as np
from time import time
from common.common import print_table, print_percent, print_inplace, writeJson
class unit:
    def __init__(self):
        self.a=None
        self.nextUnits=[]
class layer:
    def __init__(self):
        self.W=None
        self.b=None
        self.units=[]
class perc:
    def __init__(self,w1,w2,w3,t=0):
        self.thresh=t
        self.w1=w1
        self.w2=w2
        self.w3=w3
        self.in1=None
        self.in2=None
        self.in3=None
        self.out=None
    def run(self):
        ret = None
        val = self.in1.val*self.w1 + self.in2.val*self.w2 + self.in3.val*self.w3
        if val > self.thresh:
            ret = 1
        else:
            ret = -1
        self.out.val=ret
    def yval(self, xin):
        xlst=xin
        if not isinstance(xin, list) and not isinstance(xin, np.ndarray):
            xlst=[]; xlst.append(xin)
        return [(float(self.thresh)-float(self.w3)*float(x)-float(self.w1))/float(self.w2) for x in xlst]
    def plot(self,plt):
        self.yval
        
    
class NeuralPerc:
    def __init__(self):
        self.layers=[[1,1],[1,1],[1,1],[1]]
        nl=len(self.layers)
        self.w = [ [ [np.random.random(size=None) for j in range(len(self.layers[l]))] for i in range(len(self.layers[l+1]))] for l in range(nl-1)]
        self.b=[ [np.random.random(size=None) for u in range(len(self.layers[l+1]))] for l in range(len(self.layers)-1)]
        self.fixed={}
        self.p1=perc(-1,-1,-1,0)
        self.p2=perc(-1,-1,-1,0)
        self.p3=perc(-1.5,-1,1)
        self.p4=perc(-1.5,1,-1)
        self.p5=perc(1.5,1,1)
        
        self.start=perc(1,1,1)
        self.p1.in2=self.start.in2
        self.p1.in3=self.start.in3
        self.p2.in2=self.start.in2
        self.p2.in3=self.start.in3
        
        self.p3.in2=self.p1.out
        self.p3.in3=self.p2.out
        self.p4.in2=self.p1.out
        self.p4.in3=self.p2.out
        
        self.p5.in2=self.p3.out
        self.p5.in3=self.p4.out
        
        self.result=self.p5.out
        
        self.start.compNext=self.p1
        self.p1.compNext=self.p2
        self.p2.compNext=self.p3
        self.p3.compNext=self.p4
        self.p4.compNext=self.p5
        
        self.conv=0.001
        self.alpha=1.0
        self.timeout=180.0
        
    def run(self, x):
        nl=len(self.layers)
        a=self.forwardProp(x, self.w, self.b)
        return a[nl-1]
        
    ## Abstract unit activation function as f and error function as cost
    def f(self, x):
        #return self.sigmoid(x)
        return self.tanh(x)
    def fP(self, a):
        #return self.sigmoidP(a)
        return self.tanhP(a)
    def cost(self, x, y, W, B):
        return self.squaredErrorCost(x,y,W,B)
    def forwardProp(self, x, W, B):
        a=[]
        nl=len(self.layers)
        a.append([c for c in x])
        for l in range(1,nl):
            #compute all a's
            a.append([])
            for i in range(len(self.layers[l])):
                z=sum([W[l-1][i][j]*a[l-1][j] for j in range(len(self.layers[l-1]))])
                z += B[l-1][i]
                
                a[l].append(self.f(z))
            #import pdb; pdb.set_trace()
        return a
    
    ## Specific options for activation funtion and cost function
    def sigmoid(self, x):
        steep=1
        return 1.0 / (1.0 + np.exp(-steep*x))
    def sigmoidP(self, a):
        return a*(1 - a)
    def tanh(self, x):
        steep=1
        return (np.exp(steep*x) - np.exp(-steep*x))/(np.exp(steep*x) + np.exp(-steep*x))
    def tanhP(self, a):
        return 1 - a**2
    def squaredErrorCost(self, x, y, W,B):
        #import pdb; pdb.set_trace()
        nl=len(self.layers)
        M=len(x)
        c=0
        for t in range(M):
            a=self.forwardProp(x[t],W,B)
            h=a[nl-1]
            c+=0.5*sum([(y[t][k]-h[k])**2 for k in range(len(h))])
        fc = c/float(M)
        return fc
    
    def setWFixed(self,l,i,j,v):
        k="{}_{}_{}".format(l,i,j)
        self.fixed[k]=v
        self.w[l][i][j]=v
    def setBFixed(self,l,i,v):
        k="{}_{}".format(l,i)
        self.fixed[k]=v
        self.b[l][i]=v
    def isWFixed(self,l,i,j):
        ret=False
        if self.fixed.get("{}_{}_{}".format(l,i,j)) is not None:
            ret = True
        return ret
    def isBFixed(self,l,i):
        ret=False
        if self.fixed.get("{}_{}".format(l,i)) is not None:
            ret = True
        return ret
    
    def train(self, data):
        #self.inputs is list of floats
        #self.outputs is list of floats
        #a's at l=0 are self.inputs
        #a's at l=nl-1 are self.outputs
        #self.layers is list of lists
        
        
        nl=len(self.layers)
        
        M=len(data['x'])
        assert len(data['y'])==M
        x=data['x']
        assert max([len(c) for c in x])==len(self.layers[0]) and min([len(c) for c in x])==len(self.layers[0])
        y=data['y']
        assert max([len(c) for c in y])==len(self.layers[nl-1]) and min([len(c) for c in y])==len(self.layers[nl-1])
        
        #self.w = [ [ [np.random.random(size=None) for j in range(len(self.layers[l]))] for i in range(len(self.layers[l+1]))] for l in range(nl-1)]
        #self.w = [ [ [0 for j in range(len(self.layers[l]))] for i in range(len(self.layers[l+1]))] for l in range(nl-1)]
        #self.b=[ [np.random.random(size=None) for u in range(len(self.layers[l+1]))] for l in range(len(self.layers)-1)]
        #self.b=[ [0 for u in range(len(self.layers[l+1]))] for l in range(len(self.layers)-1)]
        a=self.forwardProp(x[0], self.w, self.b)

        idx=0
        wgrad=[ [ [0 for j in range(len(self.layers[l]))] for i in range(len(self.layers[l+1]))] for l in range(len(self.layers)-1)]
        bgrad=[ [0 for u in range(len(self.layers[l+1]))] for l in range(len(self.layers)-1)]
        pwgrad=[ [ [0 for j in range(len(self.layers[l]))] for i in range(len(self.layers[l+1]))] for l in range(len(self.layers)-1)]
        pbgrad=[ [0 for u in range(len(self.layers[l+1]))] for l in range(len(self.layers)-1)]
        
        dgrad=1.0
        ngrad=1.0
        st=time()
        while ngrad > self.conv and time()-st < self.timeout:
        #while idx<50:
            idx+=1
            
            #Compute gradients across all training examples
            for t in range(M):
                ## Forward Feed
                #import pdb; pdb.set_trace()
                a=self.forwardProp(x[t], self.w, self.b)
                        
                #print "Training ex {}: a: {}".format(t,a)
            
                ## Back propogate
                g=[[0 for i in range(len(self.layers[l]))] for l in range(nl)]
                for k in range(len(self.layers[nl-1])):
                    g[nl-1][k] = -(y[t][k]-a[nl-1][k])*self.fP(a[nl-1][k])#a[nl-1][k]*(1.0-a[nl-1][k])
                for l in range(nl-2,0,-1):
                    for i in range(len(self.layers[l])):
                        g[l][i] = self.fP(a[l][i])*sum([self.w[l][p][i]*g[l+1][p] for p in range(len(self.layers[l+1]))])
                        
                #print "Training ex {}: g: {}".format(t,g)
                
                for l in range(nl-1):
                    for i in range(len(self.w[l])):
                        for j in range(len(self.w[l][i])):
                            wgrad[l][i][j] += a[l][j]*g[l+1][i]
                        bgrad[l][i] += g[l+1][i]

                #print "Training ex {}: wgrad: {}".format(t,wgrad)
                #print "Training ex {}: bgrad: {}".format(t,bgrad)
            
            for l in range(nl-1):
                for i in range(len(self.w[l])):
                    for j in range(len(self.w[l][i])):
                        wgrad[l][i][j] = wgrad[l][i][j]/float(M)
                    bgrad[l][i] = bgrad[l][i]/float(M)
            
            ## Check gradients:
            eps=0.0001
            total=0
            wagreed=0
            bagreed=0
            for l in range(nl-1):
                for i in range(len(self.w[l])):
                    for j in range(len(self.w[l][i])):
                        #set eps to W component
                        orig=self.w[l][i][j]
                        self.w[l][i][j]=orig+eps
                        try:
                            jp=self.cost(x,y,self.w,self.b)
                        except Exception as e:
                            import pdb; pdb.set_trace()
                            a=1
                        self.w[l][i][j]=orig-eps
                        try:
                            jm=self.cost(x,y,self.w,self.b)
                        except Exception as e:
                            import pdb; pdb.set_trace()
                            a=2
                        dj=(jp-jm)/(2*eps)
                        total+=1
                        if (wgrad[l][i][j]-dj)<0.01:
                            wagreed+=1
                        else:
                            print "wgrad disagree: l: {}, i: {}, j: {}, bgrad: {}, dj: {}".format(l,i,j,wgrad[l][i][j],dj)
                        self.w[l][i][j]=orig
                    #set eps for b component
                    orig=self.b[l][i]
                    self.b[l][i]=orig+eps
                    try:
                        jp=self.cost(x,y,self.w,self.b)
                    except Exception as e:
                        import pdb; pdb.set_trace()
                        a=3
                    self.b[l][i]=orig-eps
                    try:
                        jm=self.cost(x,y,self.w,self.b)
                    except Exception as e:
                        import pdb; pdb.set_trace()
                        a=4
                    dj=(jp-jm)/(2*eps)
                    total+=1
                    if np.abs(bgrad[l][i]-dj)<0.001:
                        bagreed+=1
                        #print "bgrad AGREE: l: {}, i: {}, bgrad: {}, dj: {}".format(l,i,bgrad[l][i],dj)
                    else:
                        print "bgrad disagree: l: {}, i: {}, bgrad: {}, dj: {}".format(l,i,bgrad[l][i],dj)
                    self.b[l][i]=orig
            
            #print "Step {}: Gradient Check: total: {}, wagreed: {}, bagreed: {}".format(idx,total,wagreed,bagreed)
            #print
            
            
            #Compute one update
            sos=0
            for l in range(nl-1):
                for i in range(len(self.w[l])):
                    for j in range(len(self.w[l][i])):
                        if not self.isWFixed(l,i,j):
                            self.w[l][i][j] = self.w[l][i][j] - self.alpha * wgrad[l][i][j]
                    if not self.isBFixed(l,i):
                        self.b[l][i] = self.b[l][i] - self.alpha * bgrad[l][i]
            
            #print "Step {}: w: {}".format(i,self.w)
            #print "Step {}: b: {}".format(i,self.b)
            
            dgrad=0.0
            ngrad=0.0
            for l in range(len(self.layers)-1):
                for i in range(len(self.layers[l+1])):
                    for j in range(len(self.layers[l])):
                        dgrad += (pwgrad[l][i][j] - wgrad[l][i][j])**2
                        ngrad += (wgrad[l][i][j])**2
                        pwgrad[l][i][j] = wgrad[l][i][j]
                    dgrad += (pbgrad[l][i] - bgrad[l][i])**2
                    ngrad += (bgrad[l][i])**2
                    pbgrad[l][i] = bgrad[l][i]
            for l in range(len(self.layers)-1):
                for i in range(len(self.layers[l+1])):
                    for j in range(len(self.layers[l])):
                        wgrad[l][i][j]=0
                    bgrad[l][i]=0
            
            dgrad = np.sqrt(dgrad)
            if idx%10==0:
                print("dgrad: {}".format(dgrad))
                print("ngrad: {}".format(ngrad))
                #print_inplace("dgrad: {}".format(dgrad))


def run():
    import matplotlib.pyplot as plt

    def compStep(s,e,n):
        return float(e-s)/float(n)

    pointDens=10
    xstart=1
    xend=10
    xstep=compStep(xstart, xend, pointDens)
    ystart=1
    yend=10
    ystep=compStep(ystart, yend, pointDens)
    points=[(x,y) for y in np.linspace(ystart, yend, 50) for x in np.linspace(xstart, xend, 50)]


    trainingData=[]
    p1=perc(0.0,1.0,-1.0,0.0)
    p2=perc(-10.0,1.0,1.0,0.0)
    for x1, x2 in points:
        if (x1>p1.yval(x2) and x1>p2.yval(x2)) or (x1<p1.yval(x2) and x1<p2.yval(x2)):
            item=(x1,x2,1.0)
            trainingData.append(item)
        else:
            item=(x1,x2,-1.0)
            trainingData.append(item)

    n=NeuralPerc()
    n.setBFixed(1,0,-1.5)#-1.5
    n.setBFixed(1,1,-1.5)#-1.5
    n.setWFixed(1,0,0,1.0)#1.0
    n.setWFixed(1,0,1,-1.0)#-1.0
    n.setWFixed(1,1,0,-1.0)#-1.0
    n.setWFixed(1,1,1,1.0)#1.0
    n.setBFixed(2,0,1.5)#1.5
    n.setWFixed(2,0,0,1.0)#1.0
    n.setWFixed(2,0,1,1.0)#1.0

    n.w[0][0][0]=-16.0#-1.0
    n.w[0][0][1]=3.0#1.0
    n.w[0][1][0]=5.0#1.0
    n.w[0][1][1]=1.0#1.0
    n.b[0][0]=0.0#0.0
    n.b[0][1]=-7.0#-10.0

        print
    print n.w
    print
    print n.b
    print

    tdata={}
    tdata['x']=[[x1,x2] for x1,x2,v in trainingData]
    tdata['y']=[[v] for x1,x2,v in trainingData]

    print "squared error loss: {}".format(n.squaredErrorCost(tdata['x'],tdata['y'],n.w,n.b))

    n.conv=0.1
    n.alpha=0.001
    n.timeout=900.0
    n.train(tdata)
    results=[]
    for x1,x2 in points:
        vlst=n.run([x1,x2])
        v=vlst[0]
        item=(x1,x2,v)
        results.append(item)

    print
    print n.w
    print
    print n.b
    print

    print "squared error loss: {}".format(n.squaredErrorCost(tdata['x'],tdata['y'],n.w,n.b))


    #results=trainingData
    posx=[x for x1,x2,v in trainingData if v>0]; posy=[y for x1,x2,v in trainingData if v>0]
    negx=[x for x1,x2,v in trainingData if v<=0]; negy=[y for x1,x2,v in trainingData if v<=0]

    lf=lambda x:x[2]
    rpos=[a for a in results if a[2]>0]
    rneg=[a for a in results if a[2]<=0]
    mx=max(rpos, key=lf)[2]
    mn=min(rpos, key=lf)[2]
    pt=min(sorted(rpos, key=lf)[-len(rpos)/4:], key=lf)[2]
    pm=max(sorted(rpos, key=lf)[:len(rpos)/2], key=lf)[2]
    pb=max(sorted(rpos, key=lf)[:len(rpos)/4], key=lf)[2]

    mx=max(rneg, key=lf)[2]
    mn=min(rneg, key=lf)[2]
    nt=min(sorted(rneg, key=lf)[-len(rneg)/4:], key=lf)[2]
    nm=max(sorted(rneg, key=lf)[:len(rneg)/2], key=lf)[2]
    nb=max(sorted(rneg, key=lf)[:len(rneg)/4], key=lf)[2]

    rposx1=[x2 for x1,x2,v in rpos if v>pt]; rposy1=[x1 for x1,x2,v in rpos if v>pt]
    rposx2=[x2 for x1,x2,v in rpos if v>pm and v<=pt]; rposy2=[x1 for x1,x2,v in rpos if v>pm and v<=pt]
    rposx3=[x2 for x1,x2,v in rpos if v<=pm and v>pb]; rposy3=[x1 for x1,x2,v in rpos if v<=pm and v>pb]
    rposx4=[x2 for x1,x2,v in rpos if v<=pb]; rposy4=[x1 for x1,x2,v in rpos if v<=pb]

    rnegx1=[x2 for x1,x2,v in results if v>nt]; rnegy1=[x1 for x1,x2,v in results if v>nt]
    rnegx2=[x2 for x1,x2,v in results if v>nm and v<=nt]; rnegy2=[x1 for x1,x2,v in results if v>nm and v<=nt]
    rnegx3=[x2 for x1,x2,v in results if v<=nm and v>nb]; rnegy3=[x1 for x1,x2,v in results if v<=nm and v>nb]
    rnegx4=[x2 for x1,x2,v in results if v<=nb]; rnegy4=[x1 for x1,x2,v in results if v<=nb]

    #plt.scatter(posx,posy, marker='+', color='b')
    #plt.scatter(negx,negy, marker='_', color='b')

    plt.scatter(rposx1,rposy1, marker='+', color='r', alpha=1)
    plt.scatter(rposx2,rposy2, marker='+', color='y', alpha=1)
    plt.scatter(rposx3,rposy3, marker='+', color='g', alpha=1)
    plt.scatter(rposx4,rposy4, marker='+', color='b', alpha=1)
    plt.scatter(rnegx1,rnegy1, marker='_', color='b', alpha=1)
    plt.scatter(rnegx2,rnegy2, marker='_', color='g', alpha=1)
    plt.scatter(rnegx3,rnegy3, marker='_', color='y', alpha=1)
    plt.scatter(rnegx4,rnegy4, marker='_', color='r', alpha=1)

    #x = np.linspace(xstart, xend)
    #line, = plt.plot(x, p1.yval(x), '-', linewidth=2, color='b')
    #line, = plt.plot(x, p2.yval(x), '-', linewidth=2, color='r')

    plt.xlabel("x2")
    plt.ylabel("x1")
    plt.show()
