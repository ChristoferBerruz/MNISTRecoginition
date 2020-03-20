import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
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
    testX = test.reshape(test.shape[0],test.shape[1]*test.shape[2],1)
    return trainX, trainY, testX, testY

def main():
    trainX, trainY, testX, testY = loadMatrices()
    DigitNN = NN(100, 10, 784)
    DigitNN.train_by_SGD(trainX, trainY, 0.01, 100)
    DigitNN.test_model(testX, testY)

def plot_comparison(trainX, trainY, testX, testY):
    epochs = [25, 50, 100, 150]
    l1_vals = [25, 50, 100, 150]
    fig, a = plt.subplots(2, 2)
    for j in range(len(l1_vals)):
        sgd = []
        mini_batch = []
        for i in range(len(epochs)):
            l1 = l1_vals[j]
            epoch = epochs[i]
            nn = NN(l1, 10, 784)
            nn.train_by_SGD(trainX, trainY, 0.01, epoch)
            acc1 = nn.test_model(testX, testY)
            sgd.append(acc1)
            nn = NN(l1, 10, 784)
            nn.train_by_MBGD(trainX, trainY, 0.01, epoch)
            acc2 = nn.test_model(testX, testY)
            mini_batch.append(acc2)
        idx = 0
        if j >= 2:
            idx = 1
        a[idx][j%2].scatter(epochs, sgd, label = 'SGD')
        a[idx][j%2].scatter(epochs, mini_batch, label = 'Mini Batch')
        a[idx][j%2].set_title('Neurons in Hidden Layer = %d'%l1_vals[j])
    plt.legend()
    plt.show()

if __name__ == '__main__':main()