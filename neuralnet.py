import math 
import random


def sigmoid(x):
    if abs(x) < 200:
        return 1 / (1 + math.exp(-x))
    else:
        return 0 + (x > 0)

def inverseSigmoid(y):
    if y == 0 or y == 1:
        return 200 * (2*y-1)
    else:
        return - math.log(1/y - 1)

def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

class Neuron:
    def __init__(self,outputNeurons = []):
        #outputneurons is an []
        self.value = 0
        self.reset = False
        self.axons = [[n,random.random()*2-1] for n in outputNeurons]
        for i in range(len(self.axons)):
            self.axons[i][0].addDendrite(self,i)
        self.dendrites = []

    def dendriteG(self):
        for d in self.dendrites:
            yield d

    def axonG(self):
        for a in self.axons:
            yield a
        
    def getValue(self):
        return sigmoid(self.value)
    
    def getdValue(self):
        return dsigmoid(self.value)
    
    def addDendrite(self,other,myId):
        self.dendrites.insert(len(self.dendrites),[other,myId])

    def addTo(self,v):
        if self.reset:
            self.value = 0
            self.reset = False
        self.value += v
    
    def fire(self):
        for a in self.axons:
            a[0].addTo(sigmoid(self.value)*a[1])
        self.reset = True

class NeuralNet:
    def __init__(self,inputN = 3,outputN = 1,hiddenLayers = 1,hiddenN = 2):
        self.learnR = 1
        self.propFrac = 0.5
        self.outputL = [Neuron() for i in range(outputN)]
        self.hiddenL = [[] for i in range(hiddenLayers)]
        nextLayer = self.outputL
        for i in range(hiddenLayers):
            self.hiddenL[i] = [Neuron(nextLayer) for n in range(hiddenN)]
            nextLayer = self.hiddenL[i]
        self.inputL = [Neuron(nextLayer) for n in range(inputN)]
        self.layers = [self.outputL]
        for l in self.hiddenL:
            self.layers.insert(len(self.layers),l)
        self.layers.insert(len(self.layers),self.inputL)
        self.hiddenL = [v for v in reversed(self.hiddenL)]
        self.output = [0 for i in range(outputN)]

    def computePropFrac(self,layer,mode=False):
        if mode:
            pass
        else:
            self.propFrac = (len(self.layers)-layer-2)/(len(self.layers)-layer-1)
    

    def think(self,inputs):
        for i in range(len(inputs)):
            self.inputL[i].value = inputs[i]
            self.inputL[i].fire()
        for l in self.hiddenL:
            for n in l:
                n.fire()
        for n in self.outputL: # for making reset work right
            n.fire()
        for i in range(len(self.outputL)):
            self.output[i] = self.outputL[i].getValue() 
        out = [self.outputL[i].getValue() for i in range(len(self.outputL))]
        return out
    
    def learn(self,expectedOutputs,layer = 0,propMode=False):
        assert( len(self.layers[layer]) >= len(expectedOutputs))
        self.computePropFrac(layer,propMode)
        Error = 0
        newExpected = [ i.getValue() for i in self.layers[layer+1]]
        for j in range(len(expectedOutputs)):
            Error += (self.layers[layer][j].getValue() - expectedOutputs[j]) ** 2
            Partialxj = (self.layers[layer][j].getValue() - expectedOutputs[j]) * self.layers[layer][j].getdValue()
            indx = -1
            for yi in self.layers[layer][j].dendriteG():
                indx += 1
                Partialwji = Partialxj * yi[0].getValue()
                Partialyi  = Partialxj * yi[0].axons[yi[1]][1]
                if layer < len(self.layers) - 2: #can't fix input
                    newExpected[indx] -= self.propFrac * Partialyi * self.learnR
                    
                    yi[0].axons[yi[1]][1] -= Partialwji * (1-self.propFrac) * self.learnR
                else:
                    yi[0].axons[yi[1]][1] -= Partialwji  * self.learnR

        totalError = Error
        if layer < len(self.layers) - 2: #can't fix input
            totalError += self.learn(newExpected,layer+1)[1]

        return [Error,totalError]
                
    def teach(self,dataset,n=1):
        for i in range(n):
            for d in dataset:
                self.think(d[0])
                err = self.learn(d[1])
            self.modifyRates(err)

    def modifyRates(self,n):
        self.learnR = n[1]
    
