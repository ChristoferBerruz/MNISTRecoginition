import numpy as np
from sklearn.utils import shuffle
class NN(object):
    """A simple two layer neural network"""
    def __init__(self, num_neurons_l1, num_neurons_l2, input_size):
        #Initializing default weights and biases
        self.num_neurons_l1 = num_neurons_l1
        self.num_neurons_l2 = num_neurons_l2
        self.input_size = input_size
        self.w1 = np.random.uniform(low = -0.1, high = 0.1, size = (num_neurons_l1, input_size))
        self.w2 = np.random.uniform(low = -0.1, high = 0.1, size = (num_neurons_l2, num_neurons_l1))
        self.b1 = np.random.uniform(low = -0.1, high = 0.1, size = (num_neurons_l1, 1))
        self.b2 = np.random.uniform(low = -0.1, high = 0.1, size = (num_neurons_l2, 1))

    def train(self, X_train, y_train, learning_rate, epochs):
        for i in range(epochs):
            loss = 0
            X_train, y_train = shuffle(X_train, y_train) #Shuffling data to conserve i.i.d
            for k in range(X_train.shape[0]): #looping over m data points
                #Forward passing
                s1 = self.w1@X_train[i] + self.b1
                a1 = self.sigmoid(s1)
                s2 = self.w2@a1 + self.b2
                a2 = self.softmax(s2)
                #Back-propagation
                delta2 = a2 - y_train[k]
                gradb2 = delta2
                gradw2 = delta2@a1.T
                delta1 = (self.w2.T@delta2)*a1*(1-a1)
                gradb1 = delta1
                gradw1 = delta1@X_train[i].T
                #Updatting gradients
                self.b1 = self.b1 - learning_rate*gradb1
                self.b2 = self.b2 - learning_rate*gradb2
                self.w1 = self.w1 - learning_rate*gradw1
                self.w2 = self.w2 - learning_rate*gradw2
                print("epoch = %d | loss = %s" % (i, str(self.loss_function(y_train[k], a2))))
        print('Done training!')

    def test_model(self, testX, testY):
        print('Starting testing-----')
        accuracy_count = 0
        for i in range(testY.shape[0]):
            s1 = self.w1@testX[i] + self.b1
            a1 = self.sigmoid(s1)
            s2 = self.w2@a1 + self.b2
            a2 = self.softmax(s2)
            a2idx = a2.argmax(axis = 0)
            if testY[i, a2idx] == 1.0:
                accuracy_count += 1
        print('Accuracy: ', accuracy_count/testY.shape[0])

    def sigmoid(self, s1):
        return 1/(1+np.exp(-s1))

    def softmax(self, s2):
        return np.exp(s2) / np.sum(np.exp(s2))

    def loss_function(self, y, s):
        return np.sum(-y*np.log(s))