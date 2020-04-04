import os
import sys
import cv2
from Layer import Layer
from NN import NN
from ActivationTools import ActivationType
from GradientDecentType import GradientDecentType
import numpy as np

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
    testX = test.reshape(test.shape[0],test.shape[1]*test.shape[2],1)
    return trainX, trainY, testX, testY

def main():
    trainX, trainY, testX, testY = loadMatrices()
    #HiddenLayer = Layer(784, 50, ActivationType.RELU)
    #OuputLayer = Layer(50, 10, ActivationType.SOFTMAX, True)
    #layers = [HiddenLayer, OuputLayer]
    Digit_Classifier = NN([50, 10], ActivationType.SIGMOID, 784)
    Digit_Classifier.Train(trainX, trainY, epochs = 100, GradType = GradientDecentType.MINIBATCH)
    Digit_Classifier.test_accuracy(testX, testY)

if __name__ == '__main__':main()
