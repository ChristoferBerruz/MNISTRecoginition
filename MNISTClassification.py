import os
import sys
import cv2
import numpy as np
from sklearn.utils import shuffle
from NN import NN

def loadMatrices(): #Loading images 
    #Defining globals to be updated
    train = np.empty((1000, 28, 28), dtype='float64')
    trainY = np.zeros((1000, 10, 1))
    test = np.empty((10000, 28, 28), dtype='float64')
    testY = np.zeros((10000, 10, 1))
    #Loading train data
    i = 0
    for fname in os.listdir('C:/Users/chris/Downloads/Data/Data/Training1000'):
        y =  int(fname[0])
        trainY[i,y] = 1.0
        train[i] = cv2.imread('C:/Users/chris/Downloads/Data/Data/Training1000/{}'.format(fname), 0)/255.0 #Normalizing pixels values from 0 - 1
        i += 1
    #Loading test data
    i = 0
    for fname in os.listdir('C:/Users/chris/Downloads/Data/Data/Test10000'):
        y = int(fname[0])
        testY[i, y] = 1.0
        test[i] = cv2.imread('C:/Users/chris/Downloads/Data/Data/Test10000/{}'.format(fname), 0)/255.0
        i += 1
    trainX = train.reshape(train.shape[0], train.shape[1]*train.shape[2], 1)
    return trainX, trainY, np.reshape(test,(test.shape[0], test.shape[1]*test.shape[2], 1)), testY
def main():
    trainX, trainY, testX, testY = loadMatrices()
    nn = NN(100, 10, 784)
    nn.train(trainX, trainY, 0.01, 5)
    nn.test_model(testX, testY)
if __name__ == '__main__':main()