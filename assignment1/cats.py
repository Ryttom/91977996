import numpy as np
import pickle as pkl
import random
import dill  #pip install dill

def catfunction(g):
    ris=[]
    for i in range(64):
        for j in range(64):
            for k in range(3):
                ris.append(g[i][j][k])
    return ris

def readmodel(file):
    with open(file, 'rb') as in_strm:
        return dill.load(in_strm)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def lrloss(yhat, y):
    return 0.0 if yhat==y else -1.0*(y*np.log(yhat)+(1-y)*np.log(1-yhat))

class Cat_Model:
    def lrpredict(self, x):
        return 1 if self(x)>0.5 else 0

    def __init__(self, dimension=None, weights=None, bias=None, activation=(lambda x: x), predict=None):
        self._dim = dimension
        self.w = weights or np.random.normal(size=self._dim)
        self.w = np.array(self.w)
        self.b = bias if bias is not None else np.random.normal()
        self._a = activation
        self.predict = predict.__get__(self)

    def __str__(self):
        return "Simple cell neuron\n\
        \tInput dimension: %d\n\
        \tBias: %f\n\
        \tWeights: %s\n\
        \tActivation: %s" % (self._dim, self.b, self.w, self._a.__name__)

    def __call__(self, x):
        yhat = self._a(np.dot(self.w, np.array(x)) + self.b)
        return yhat

    def load_model(self, file_path):
        pass

    def save_model(self):
        f=open('cat_model.pkl','bw')
        dill.dump(self,f)
        f.close()
        pass

class Cat_Trainer:

    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model
        #self.loss = ploss

    def accuracy(self, data):
        return 100*np.mean([1 if self.model.predict(x) == y else 0 for x, y in data.data])


    def train(self, lr, ne):
        print("training model on data...")
        accuracy = self.accuracy(self.dataset)
        print("initial accuracy: %.3f" % (accuracy))

        for epoch in range(ne):
            for d in self.dataset.data:
                x, y = d
                x = np.array(x)
                yhat = self.model.predict(x)
                error = y - yhat
                self.model.w += lr*(y-yhat)*x
                self.model.b += lr*(y-yhat)
            #print(model.w[:5])
            accuracy = self.accuracy(self.dataset)
            print('>epoch=%d, learning_rate=%.3f, accuracy=%.3f' % (epoch+1, lr, accuracy))

        print("training complete")
        print("final accuracy: %.3f" % (self.accuracy(self.dataset)))

class Cat_Data:

    def __init__(self, relative_path='/Users/thomas/Desktop/Coding/Visual/', data_file_name='cat_data.pkl'):
        self.index = -1
        with open(relative_path + data_file_name,"rb") as d:
            data = pkl.load(d)['train']
            cat = data['cat']
            no_cat = data['no_cat']

            self.data = [(list(catfunction(d)), 1) for d in cat]+[(list(catfunction(d)), 0) for d in no_cat]
            #random.shuffle(self.data)

    def __iter__(self):
        return self

    def __next__(self):
        def __next__(self):
            self.index += 1
            if self.index >= len(self.data):
                self.index = -1
                raise StopIteration

cat_data = Cat_Data()

model = Cat_Model(dimension=64*64*3,activation=sigmoid,predict=Cat_Model.lrpredict)  # specify the necessary arguments
trainer = Cat_Trainer(cat_data, model)
trainer.train(0.001, 600)

print(model)
print(trainer.accuracy(cat_data))