from GradientDecentType import GradientDecentType
from Layer import Layer
from ActivationTools import ActivationType
from sklearn.utils import shuffle
import numpy as np
class NN(object):
    """Neural Network"""

    '''
    def init_by_list(self, layer_list):
        """
        Initializing a neural network

        Parameters:
        - layer_list: list[Layer]
            Layers to build the network
        """
        self.layer_list = layer_list


    def init_by_layerfunc(self, numneuron_func, input_size):
        """
        Different initialization for NN

        Parameters:
        - numneuron_func : list[(int, ActivationType)]
            List of tuples such that each tupple specifies the number of neurons in each layer with the activation function of the layer.
        - input_size: int
            Size of the input of the Network
        """
        lastindex = len(numneuron_func) - 1
        layers = []
        for idx, pair in enumerate(numneuron_func):
            num_neurons, activation_function = pair
            layer = Layer(input_size, num_neurons, activation_function, idx == lastindex)
            input_size = num_neurons
            Layers.append(layer)
        self.init_by_list(layers)
    '''


    def __init__(self, numneurons_list, activationType_allLayers, input_size):
        """
        Main constructor for NN.

        Parameters:
        - numneurons_list: list[int]
            List of integers where each integer specifies the number of neurons at that layer
        - activationType_allLayer: ActivationType
            ActivationType to be applied to ALL hidden layers
        - input_size:
            Input size of the network itself
        """
        self.batchnorm = False
        self.drop_out = False
        self.optimizer = ''
        self.learning_rate = 0.0
        self.epochs = 0
        self.drop_percent = 0.2
        lastindex = len(numneurons_list) - 1
        layers = []
        for idx, num_neuron in enumerate(numneurons_list):
            layer = None
            if idx == lastindex:
                layer = Layer(input_size, num_neuron, ActivationType.SOFTMAX, True)
            else:
                layer = Layer(input_size, num_neuron, activationType_allLayers, False)
            input_size = num_neuron
            layers.append(layer)
        self.layer_list = layers


    def Train(self, X, Y, GradType = GradientDecentType.STOCHASTIC, batch_size = 10, drop_out = False, drop_percent = 0.2, epochs = 100, learning_rate = 0.01, optimizer = '', batchnorm = False):
        """
        Trains the network by Gradient Descent

        Paremeters:
        - X: ndarray
            Training data
        - Y: ndarray
            Expected output
        - GradType: GradientDecentType
            Specifies which Gradient Descent Variation to use
        - batch_size: float (optional)
            Batch size
        - drop_out: bool
            Whether to perform dropout or not
        - epochs: int
            Number of epochs
        - learning_rate:float
            Learning rate of algortihm
        - optimizer: str
            'Adam' or ''
        - batchnorm: bool
            Whether to perform batch normalization or not
        """
        self.batchnorm = batchnorm
        self.drop_out = drop_out
        self.optimizer = optimizer
        self.learning_rate =  learning_rate
        self.epochs = epochs
        self.drop_percent = drop_percent
        if(GradType == GradientDecentType.STOCHASTIC):
            batch_size = 1
        if(GradType == GradientDecentType.BATCH):
            batch_size = X.shape[0]
        self.mini_batch(X, Y, epochs, learning_rate, batch_size, drop_out = drop_out, optimizer = optimizer, batchnorm = batchnorm)


    def mini_batch(self, X, y, epochs, learning_rate, batch_size, drop_out, optimizer, batchnorm):
        """
        All versions of Gradient Descent can be thought as special cases of the Mini-Batch algorithm.

        Parameters:
        - X: ndarray
            Training data
        - y: ndarray
            Expected outputs
        - epochs: int
            Number of epochs
        - learning_rate: float
            Learning rate to be used.
        - batch_size = int
            Batch size to be used.
        - drop_out: bool
            Whether to perform drop_out
        - optimizer: str
            'Adam' or '' 
        - batchnorm: bool
            Whether to perform batch normalization or not
        """
        last_layer = self.layer_list[-1] #Pointer to last layer
        for epoch in range(epochs):
            X, y = shuffle(X, y) #Shuffling data to conserve i.i.d

            #------------------Creating dropout matrices if needed be ----------------------#
            if drop_out:
                for idx, layer in enumerate(self.layer_list):
                    if idx != last_layer:
                        layer.drop_out = True
                        DropM = np.random.binomial(1, (1-self.drop_percent), (layer.num_neurons))/(1-self.drop_percent)
                        layer.dropM = DropM

            #----------------------------------Batches-------------------------------------#
            for batch in np.arange(0, X.shape[0], batch_size):
                D = X[batch:batch+batch_size, :]
                prev_layer_data = D

                #Forward passing
                for layer in self.layer_list:
                    prev_layer_data = layer.forward_pass(prev_layer_data, batchnorm = batchnorm)
          
                #Back propagation
                next_layer_data = y[batch:batch+batch_size, :]
                expected = y[batch:batch+batch_size, :]
                for k in range(len(self.layer_list)-1, -1, -1):
                    layer = self.layer_list[k]
                    next_layer_data = layer.back_propagate(next_layer_data)
                    layer.update_gradients() #Updating gradients
                if batch %10 == 0:
                    print("Cummulative iteration = %s, Loss = %.7f" % (batch, last_layer.calculate_loss(expected)))

                #Updating parameters of each layer
                for layer in self.layer_list:
                    layer.update_thetas(learning_rate, optimizer)
        print("Done training!")


    def test_accuracy(self, X_test, Y_test):
        accu_count = 0
        for i in range(X_test.shape[0]):
            prev_layer_data = X_test[i:i+1, :]
            for layer in self.layer_list:
                prev_layer_data = layer.forward_pass(prev_layer_data, mode = 'Test', batchnorm=self.batchnorm)
            #After forward passing, prev_layer_data has the actual output
            idx_max = prev_layer_data.argmax(axis = 1)
            if Y_test[i, idx_max] == 1.0:
                accu_count += 1
        print("------------------- Network Result ---------------------")
        print("Layers: ", list(map(str, self.layer_list)))
        print("Dropout : %s\nOptimizer : %s\nLearning rate: %.6f\nEpochs: %d\nBatch Normalization: %s" 
              % (self.drop_out, self.optimizer, self.learning_rate, self.epochs, self.batchnorm ))
        print("Accuracy of network: %.5f" % (accu_count/X_test.shape[0]))
            