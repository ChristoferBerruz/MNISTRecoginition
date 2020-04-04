from GradientDecentType import GradientDecentType
from Layer import Layer
from ActivationTools import ActivationType
from sklearn.utils import shuffle
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


    def Train(self, X, Y, GradType = GradientDecentType.STOCHASTIC, batch_size = 10, epochs = 100, learning_rate = 0.01):
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
        - epochs: int
            Number of epochs
        - learning_rate:float
            Learning rate of algortihm
        """
        if(GradType == GradientDecentType.STOCHASTIC):
            batch_size = 1
        if(GradType == GradientDecentType.BATCH):
            batch_size = X.shape[0]
        self.mini_batch(X, Y, epochs, learning_rate, batch_size)


    def mini_batch(self, X, y, epochs, learning_rate, batch_size):
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
        """
        last_layer = self.layer_list[-1] #Pointer to last layer
        for epoch in range(epochs):
            X, y = shuffle(X, y) #Shuffling data to conserve i.i.d
            for batch in range(X.shape[0]//batch_size):

                for i in range(batch_size):
                    idx = batch*batch_size + i
                    prev_layer_data = X[idx]

                    #Forward passing to all layers
                    for layer in self.layer_list:
                        prev_layer_data = layer.forward_pass(prev_layer_data)
                    
                    #Back propagation
                    next_layer_data = y[idx]
                    expected = y[idx]
                    for k in range(len(self.layer_list)-1, -1, -1):
                        layer = self.layer_list[k]
                        next_layer_data = layer.back_propagate(next_layer_data)
                        layer.update_gradients()

                    if idx % 100 == 0:
                        print("Cummulative iteration = %s, Loss = %.7f" % (idx, last_layer.calculate_loss(expected)))
                for layer in self.layer_list:
                    layer.update_thetas(learning_rate)
        print("Done training!")


    def test_accuracy(self, X_test, Y_test):
        accu_count = 0
        for i in range(X_test.shape[0]):
            prev_layer_data = X_test[i]
            for layer in self.layer_list:
                prev_layer_data = layer.forward_pass(prev_layer_data)
            #After forward passing, prev_layer_data has the actual output
            idx_max = prev_layer_data.argmax(axis = 0)
            if Y_test[i, idx_max] == 1.0:
                accu_count += 1
        print("Current accuracy of network: %.5f" % (accu_count/X_test.shape[0]))
            