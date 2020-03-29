import numpy as np
from ActivationTools import ActivationType
from ActivationTools import ActivationFunctions as Functions
class Layer(object):
    """Layer class"""


    def __init__(self, input_size, num_neurons, Activation = ActivationType.SIGMOID, last_layer = False, drop_out = False, drop_percent = 0.2):
        """
        Constructs the layer
        Parameters:
        - input_size: int
            input size of the layer
        - num_neurons: int
            number of neurons in layer
        - Activation: ActivationType
            Activation type of all neurons in layer
        - last_layer: bool
            Specifies if layer is last layer. Only if last_layer = True, we can use calcula_loss method
        - drop_out: bool
            If layer support droput
        - drop_percent: float
            Percentage of neurons to be randomly shut down during training
        """
        self.W = np.random.uniform(low = -0.1, high = 0.1, size = (num_neurons, input_size))
        self.b = np.random.uniform(low = -0.1, high = 0.1, size = (num_neurons, 1))
        self.num_neurons = num_neurons
        self.input_size = input_size
        self.Activation = Activation
        self.a = np.zeros((num_neurons, 1)) #The output of the layer
        self.gradW  = np.zeros(shape = self.W.shape)
        self.gradb = np.zeros(shape = self.b.shape)
        self.last_layer = last_layer
        self.delta = np.zeros(shape = self.b.shape)
        self.updating_count = 1
        self.input_data = None


    def forward_pass(self, input_data):
        """
        Forward passing data.
        Parameters:
        - input_data: np.ndarray
            Represents data being passed to the layer. Once we forward pass, the layer stores the value as a property

        Returns:
        - self.a: np.ndarray
            Input for next layer. If self is last layer, this is the actual output

        Raises
         - Exception: Exception
            Raised when condition ActivationType == SIGMOID and laster_layer == True is not valid
        """
        self.input_data = input_data
        s1 = self.W@input_data + self.b
        if(self.Activation == ActivationType.SIGMOID):
            self.a = Functions.sigmoid(s1)
        elif(self.Activation == ActivationType.RELU):
            self.a = Functions.relu(s1)
        elif(self.Activation == ActivationType.TANH):
            self.a = Functions.tanh(s1)
        elif(self.Activation == ActivationType.SOFTMAX and self.last_layer):
            self.a = Functions.softmax(s1)
        else:
            raise Exception("Please change boolean parameter or activation type")
        return self.a


    def back_propagate(self, weighted_deltas):
        """
        Back propagates weighted deltas back to network
        Parameters:
        - weighted_deltas: ndarray
            Deltas of the next layer weighted by next layer's W matrix.

        Returns
            weighted deltas of self as np.ndarray
        """
        if(self.Activation == ActivationType.SIGMOID):
            self.delta = weighted_deltas*self.a*(1-self.a)
        elif(self.Activation == ActivationType.RELU):
            self.delta = weighted_deltas*np.where(self.a >= 0, 1, 0)
        elif(self.Activation == ActivationType.TANH):
            self.delta = weighted_deltas*(1-self.a**2)
        elif(self.Activation == ActivationType.SOFTMAX and self.last_layer):
            #the weighted deltas for this case are the expected result Y
            self.delta = (self.a - weighted_deltas)
        else:
            raise Exception("Please change last_layer or activation type")
        return self.W.T@self.delta


    def update_gradients(self, regularized = True, reg_val = 0.01):
        """
        Accumulate gradients with regularization by default. Layers keeps track how many times the gradients were accumulated.
        This feature is useful when implementation iterative algorithms

        Parameters:
        - regularized: bool
            If regularization is implemented in layer
        - reg_val: float
            Regularization value
        """
        self.updating_count += 1
        self.gradW += self.delta@self.input_data.T
        self.gradb += self.delta
        if regularized:
            self.gradW += reg_val*self.W


    def update_thetas(self, learning_rate = 0.01):
        """
        Update the W matrix and biases for the layer. Once we update thetas, the gradW and gradb are set to zero.
        We also reset the count of how many times we acumulated the gradients.

        Parameters:
        - learning_rate: float
            Learning rate to use to update thetas
        """
        self.W += -learning_rate*self.gradW/self.updating_count
        self.b += -learning_rate*self.gradb/self.updating_count
        self.gradW = np.zeros(shape = self.W.shape)
        self.gradb = np.zeros(shape = self.b.shape)
        self.updating_count = 0


    def calculate_loss(self, y_expected):
        """
        Calculates loss only at last layer

        Parameters:
        - y_expected: ndarray
            Exepected value of the last layer

        Returns:
        - float
            Loss value

        Raises:
        - Exception: Exception
            When layer is not last layer and client wants to calculate loss
        """
        if not self.last_layer:
            raise Exception("Loss can only be calculated at last layer")
        return self.loss(y_expected)


    def loss(self, y_expected):
        return np.sum(-y_expected*np.log(self.a))