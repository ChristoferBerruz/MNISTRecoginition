import numpy as np
from sklearn.utils import shuffle
import math
class NN(object):
    """A simple two layer neural network with various activation functions in the hidden layer.
    The loss is defined as cross-entropy and ouput layer uses a softmax function.

    Please note that the activation function in both training and testing MUST be the same.
    """
    '''
    Inputs:
        - num_neurons_l1 : Number of neurons in hidden layer.
        - num_neurons_l2 : Number of neurons in output layer.
        - input_size : size of the input datum, remember that datum is a vector of dim = (input_size, 1)
    '''
    def __init__(self, num_neurons_l1, num_neurons_l2, input_size):
        #Initializing default weights and biases
        self.num_neurons_l1 = num_neurons_l1
        self.num_neurons_l2 = num_neurons_l2
        self.input_size = input_size
        self.w1 = np.random.uniform(low = -0.1, high = 0.1, size = (num_neurons_l1, input_size))
        self.w2 = np.random.uniform(low = -0.1, high = 0.1, size = (num_neurons_l2, num_neurons_l1))
        self.b1 = np.random.uniform(low = -0.1, high = 0.1, size = (num_neurons_l1, 1))
        self.b2 = np.random.uniform(low = -0.1, high = 0.1, size = (num_neurons_l2, 1))
    '''
    Stochastic Gradient Descent algorithm
    Inputs:
        - X_train: training input
        - y_train: training output, or matrix of expected outputs
        - learning_rate : learning rate of algorithm
        - epochs: number of epochs

    Output:
        - Void, but we save parameters to a text file
    '''
    def train_by_SGD(self, X_train, y_train, learning_rate, epochs):
        print('Training by SGD')
        for i in range(epochs):
            loss = 0
            X_train, y_train = shuffle(X_train, y_train) #Shuffling data to conserve i.i.d
            for k in range(X_train.shape[0]): #looping over m data points
                #Forward passing
                s1 = self.w1@X_train[k] + self.b1
                a1 = self.relu(s1)
                s2 = self.w2@a1 + self.b2
                a2 = self.softmax(s2)
                #Back-propagation
                delta2 = a2 - y_train[k]
                gradb2 = np.sum(delta2, axis = 0, keepdims = True)
                gradw2 = delta2@a1.T
                delta1 = (self.w2.T@delta2)*self.relu_derivative(s1)
                gradb1 = np.sum(delta1, axis = 0, keepdims = True)
                gradw1 = delta1@X_train[k].T
                #Updatting gradients
                self.b1 = self.b1 - learning_rate*gradb1
                self.b2 = self.b2 - learning_rate*gradb2
                self.w1 = self.w1 - learning_rate*gradw1
                self.w2 = self.w2 - learning_rate*gradw2
                #if k == (X_train.shape[0] - 1):
                #    print('Epoch = %d, Loss = %.4f'%(i, self.loss_function(y_train[k], a2)))
        print('Done training!')
        self.save_theta() #Saving learned parameters in a file
    
    '''
    Similar to SGD, but in batches. 
    Batch size MUST be a power of 2 IF possible, due to the size of training sample size, it's default is 10. 

    Output:
        - Void, but we save parameters to a text file.
    '''
    def train_by_MBGD(self, X_train, y_train, learning_rate, epochs, batch_size = 10):
        print('Training by Mini Batch SGD')
        for i in range(epochs):
            loss = 0
            X_train, y_train = shuffle(X_train, y_train) #Shuffling data to conserve i.i.d
            for batch in range(int(math.ceil(X_train.shape[0]/batch_size))):
                gradw1 = 0
                gradb1 = 0
                gradw2 = 0
                gradb2 = 0
                for k in range(batch_size):
                    #Forward passing
                    idx = batch*batch_size + k
                    s1 = self.w1@X_train[idx] + self.b1
                    a1 = self.sigmoid(s1)
                    s2 = self.w2@a1 + self.b2
                    a2 = self.softmax(s2)
                    #Back-propagation
                    delta2 = a2 - y_train[idx]
                    gradb2 += np.sum(delta2, axis = 0, keepdims = True)
                    gradw2 += delta2@a1.T
                    delta1 = (self.w2.T@delta2)*a1*(1-a1)
                    gradb1 += np.sum(delta1, axis = 0, keepdims = True)
                    gradw1 += delta1@X_train[idx].T
                    #if k == (batch_size - 1) and (batch % 5 == 0):
                    #    print('Epoch = %d, Batch = %d, Loss = %.4f'%(i, batch, self.loss_function(y_train[idx], a2)))
                #Updatting gradients
                self.b1 = self.b1 - learning_rate*(gradb1/batch_size)
                self.b2 = self.b2 - learning_rate*(gradb2/batch_size)
                self.w1 = self.w1 - learning_rate*(gradw1/batch_size)
                self.w2 = self.w2 - learning_rate*(gradw2/batch_size)
        print('Done Training!')
        self.save_theta()

    '''
    Runs a test for the model. Please, have common sense and do not use training as testing data.

    Input:
        - testX : testing input data
        - testY : testing expected output
        - file : bool, True only if training was done prior. Specifies to load parameters from file.

    Output:
        - accuracy: accuracy between [0, 1] as float
    '''
    def test_model(self, testX, testY, file = False):
        print('Starting testing-----')
        if file:
            self.load_theta()
        accuracy_count = 0
        for i in range(testY.shape[0]):
            s1 = self.w1@testX[i] + self.b1
            a1 = self.relu(s1)
            s2 = self.w2@a1 + self.b2
            a2 = self.softmax(s2)
            a2idx = a2.argmax(axis = 0)
            if testY[i, a2idx] == 1.0:
                accuracy_count += 1
        print('Accuracy: ', accuracy_count/testY.shape[0])
        return accuracy_count/testY.shape[0]
    
    '''
    Softmax activation function for output layer
    '''
    def softmax(self, s2):
        return np.exp(s2) / np.sum(np.exp(s2))


    '''
    Activation functions and its derivatives for HIDDEN Layer
    '''
    def sigmoid(self, s1):
        return 1/(1+np.exp(-s1))

    def sigmoid_derivative(self, s1):
        a = self.sigmoid(s1)
        return a*(1-a)

    def tanh(self, s1):
        a = np.exp(s1)
        b = np.exp(-s1)
        return (a - b)/(a + b)
    
    def tanh_derivative(self, s1):
        return 1 - self.tanh(s1)**2

    def relu(self, s1):
        return np.where(s1 <= 0, 0, s1)
    
    def relu_derivative(self, s1):
        return np.where(s1> 0, 1, 0)


    '''
    Loss function is cross-entropy
    '''
    def loss_function(self, y, s):
        return np.sum(-y*np.log(s))
    

    '''
    Methods for writing and reading to a text file
    '''
    def save_theta(self):
        np.savetxt('w1.txt', self.w1, delimiter = ',')
        np.savetxt('b1.txt', self.b1, delimiter = ',')
        np.savetxt('w2.txt', self.w2, delimiter = ',')
        np.savetxt('b2.txt', self.b2, delimiter = ',')

    def load_theta(self):
        self.w1 = np.loadtxt('w1.txt', delimiter = ',')
        self.b1 = np.loadtxt('b1.txt', delimiter = ',')
        self.b1 = np.reshape(self.b1, (self.num_neurons_l1, 1))
        self.w2 = np.loadtxt('w2.txt', delimiter = ',')
        self.b2 = np.loadtxt('b2.txt', delimiter = ',')
        self.b2 = np.reshape(self.b2, (self.num_neurons_l2, 1))