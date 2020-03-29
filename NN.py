from GradientDecentType import GradientDecentType
from sklearn.utils import shuffle
class NN(object):
    """Neural Network"""

    def __init__(self, layer_list):
        """
        A neural network is a list of layers. With last layer being last_layer = True

        Parameters:
        - layer_list: list[Layer]
            Layers to build the network
        """
        self.layer_list = layer_list


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
            