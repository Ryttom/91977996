import numpy as np
import pickle as pkl
import random

def perceptron(z):
    return -1 if z<=0 else 1

def ploss(yhat, y):
    return max(0, -yhat*y)


class Sonar_Model:
    def ppredict(self, x):
        return self(x)

    def __init__(self, dimension=None, weights=None, bias=None, activation=(lambda x: x), predict=ppredict):
        self._dim = dimension
        self.w = weights or np.random.normal(size=self._dim)
        self.w = np.array(self.w)
        self.b = bias if bias is not None else np.random.normal()
        self._a = activation
        self.predict = predict.__get__(self)


    def __str__(self):
        '''
        display the model's information
        '''
        return "Simple cell neuron\n\
        \tInput dimension: %d\n\
        \tBias: %f\n\
        \tWeights: %s\n\
        \tActivation: %s" % (self._dim, self.b, self.w, self._a.__name__)

    def __call__(self, x):
        '''
        return the output of the model for a given input
        '''
        yhat = self._a(np.dot(self.w, np.array(x)) + self.b)
        return yhat

    def load_model(self, file_path):
        '''
        open the pickle file and update the model's parameters
        '''
        pass

    def save_model(self):
        '''
        save your model as 'sonar_model.pkl' in the local path
        '''
        pass

class Sonar_Trainer:

    def __init__(self, dataset, model):

        self.dataset = dataset
        self.model = model
        self.loss = ploss

    def cost(self, data):

        return np.mean([self.loss(self.model.predict(x), y) for x, y in data])

    def accuracy(self, data):
        l = [1 if self.model.predict(x) == y else 0 for x, y in data]
        mean = np.mean(l)
        return 100*mean

    def train(self, lr, ne):

        print("training model on data...")
        accuracy = self.accuracy(self.dataset)
        print("initial accuracy: %.3f" % (accuracy))

        for epoch in range(ne):
            for d in self.dataset:
                x, y = d
                x = np.array(x)
                yhat = self.model(x)
                error = y - yhat
                self.model.w += lr*(y-yhat)*x
                self.model.b += lr*(y-yhat)
            accuracy = self.accuracy(self.dataset)
            print('>epoch=%d, learning_rate=%.3f, accuracy=%.3f' % (epoch+1, lr, accuracy))

        print("training complete")
        print("final accuracy: %.3f" % (self.accuracy(self.dataset)))

class Sonar_Data:

    def __init__(self, relative_path='/Users/thomas/Desktop/Coding/Visual/', data_file_name='sonar_data.pkl'):
        '''
        initialize self.index; load and preprocess data; shuffle the iterator
        '''
        self.index = -1
        with open(relative_path + data_file_name,"rb") as f:
            data = pkl.load(f)
            mines = data['m']
            rocks = data['r']

            self.data = [(list(d), -1) for d in mines]+[(list(d), 1) for d in rocks]
            random.shuffle(self.data)

    def __iter__(self):
        '''
        See example code (ngram) in lecture slides
        '''

        return self

    def __next__(self):
        self.index += 1
        if self.index >= len(self.data):
            self.index = -1
            raise StopIteration

        '''
        See example code (ngram) in slides
        '''
        return self.data[self.index]

def main():

    sonar_data = Sonar_Data()
    model = Sonar_Model(dimension=60, activation=perceptron, predict=Sonar_Model.ppredict)  # specify the necessary arguments
    print(model)
    trainer = Sonar_Trainer(sonar_data, model)
    trainer.train(0.009, 1000) # experiment with learning rate and number of epochs
    model.save_model()


if __name__ == '__main__':

    main()
