import numpy as np
from ActivationTools import ActivationType
from ActivationTools import ActivationFunctions as Functions
from Optimizers import Adam, DefaultOptimizer
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
        self.W = np.random.uniform(low = -0.1, high = 0.1, size = (input_size, num_neurons))
        self.b = np.random.uniform(low = -0.1, high = 0.1, size = (1, num_neurons))
        self.num_neurons = num_neurons
        self.input_size = input_size
        self.Activation = Activation
        self.a = None #The output of the layer
        self.gradW  = np.zeros(shape = self.W.shape)
        self.gradb = np.zeros(shape = self.b.shape)
        self.last_layer = last_layer
        self.delta = None
        self.input_data = None
        self.Optimizer = DefaultOptimizer() #We use the otimizer to calculate new parameters
        self.batchnorm = False
        self.dropM = None
        self.drop_percent = drop_percent
        self.drop_out = drop_out
        self.mode = 'Train'
        #------------------------------------ for Batch Normalization --------------------------------#
        if not last_layer:
            self.gamma = np.random.uniform(low = -0.1, high = 0.1, size = self.b.shape)
            self.beta = np.random.uniform(low = -0.1, high = 0.1, size = self.b.shape)
            self.sihat = None
            self.sb = None
            self.variance = None
            self.mean = None
            self.deltaBN = None
            self.gradGamma = np.zeros(shape = self.gamma.shape)
            self.gradBeta = np.zeros(shape = self.beta.shape)
            self.cumulative_mean = 0
            self.cumulative_variance = 0
            self.EPSILON = 10**(-6)


    def forward_pass(self, input_data, mode = 'Train', batchnorm = False):
        """
        Forward passing data.
        Parameters:
        - input_data: np.ndarray
            Represents data being passed to the layer. Once we forward pass, the layer stores the value as a property
        - mode: str
            'Train' if in training stage. Test otherwise
        - batchnotm: bool
            Whether to perform bacth normalization

        Returns:
        - self.a: np.ndarray
            Input for next layer. If self is last layer, this is the actual output

        Raises
         - Exception: Exception
            Raised when condition ActivationType == SIGMOID and laster_layer == True is not valid
        """
        self.input_data = input_data
        s1 = input_data@self.W + self.b
        self.batchnorm = batchnorm


        #-------------------------- for batch normalization --------------------------------#
        if (not self.last_layer) and self.batchnorm:
            if mode == 'Train':
                self.variance = np.var(s1, axis = 0, keepdims = True, dtype = np.float64)
                self.mean = np.mean(s1, axis = 0, keepdims = True)
                self.cumulative_mean = 0.9*self.cumulative_mean + (1-0.9)*self.mean
                self.cumulative_variance = 0.9*self.cumulative_variance + (1-0.9)*self.variance
            else:
                self.mean = self.cumulative_mean
                self.variance = self.cumulative_variance
            self.sihat = (s1 - self.mean)/np.sqrt(self.variance + self.EPSILON)
            self.sb = self.gamma*self.sihat + self.beta
            s1 = self.sb


        #------------------------- Calculating activations ------------------------------#
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


        #Zeroing out some weights
        if (not self.last_layer) and self.drop_out and (mode == 'Train'):
            self.a = self.a * self.dropM


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

        #------------------------ Back propagating --------------------------#
        if(self.Activation == ActivationType.SIGMOID):
            self.delta = weighted_deltas*self.a*(1-self.a)
        elif(self.Activation == ActivationType.RELU):
            derivative = np.where(self.a >= 0, 1, 0) if not self.batchnorm else np.where(self.a > self.EPSILON, 1, self.EPSILON) #Just for batch normalization
            self.delta = weighted_deltas*derivative
        elif(self.Activation == ActivationType.TANH):
            self.delta = weighted_deltas*(1-self.a**2)
        elif(self.Activation == ActivationType.SOFTMAX and self.last_layer):
            #the weighted deltas for this case are the expected result Y
            self.delta = (self.a - weighted_deltas)
        else:
            raise Exception("Please change last_layer or activation type")
        if(not self.last_layer) and self.drop_out:
            self.delta = self.delta * self.dropM
        if self.last_layer or (not self.batchnorm):
            return (self.W@self.delta.T).T
        

        #------------------------- Calculating deltabn and back propagate ---------------------#
        denominator = self.input_data.shape[0]*np.sqrt(self.variance + 10**(-8))
        self.deltaBN = self.delta*self.gamma*(self.input_data.shape[0] - 1 - self.sihat**2)/denominator
        return (self.W@self.deltaBN.T).T


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

        if self.last_layer or (not self.batchnorm):
            self.gradW = (self.input_data.T@self.delta)/self.input_data.shape[0] #Average gradient
            self.gradb = np.mean(self.delta, axis = 0, keepdims = True) #average gradient
        else:
            #-------------------------- for Batch Norm --------------------------#
            self.gradW = (self.input_data.T@self.deltaBN)/self.input_data.shape[0]
            self.gradb = np.mean(self.deltaBN, axis = 0, keepdims = True)
            self.gradBeta = np.mean(self.delta, axis = 0, keepdims = True)
            self.gradGamma = np.mean(self.sihat*self.delta, axis = 0, keepdims = True)
        if regularized:
            self.gradW += reg_val*self.W


    def update_thetas(self, learning_rate = 0.01, optimizer = ''):
        """
        Update the W matrix and biases for the layer. Once we update thetas, the gradW and gradb are set to zero.
        We also reset the count of how many times we acumulated the gradients.

        Parameters:
        - learning_rate: float
            Learning rate to use to update thetas
        - optimizer: str
            Which optimizer to use. Empty string is default
        """
        createInstance = True #We only create one instance of the Optimizer
        if createInstance and optimizer == 'Adam': #Taking advantage of lazy and
            self.Optimizer = Adam()
            createInstance = False
        self.W = self.Optimizer.newWeight(self.W, self.gradW, learning_rate)
        self.b = self.Optimizer.newBias(self.b, self.gradb, learning_rate)
        if (not self.last_layer) and self.batchnorm:
            self.gamma += -learning_rate*self.gradGamma
            self.beta += -learning_rate*self.gradBeta

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

    def __str__(self):
        """
        String representation of layer object
        """
        layer_type = "Output" if self.last_layer else "Hidden"
        funcType = ''
        if self.Activation == ActivationType.SIGMOID:
            funcType = "SIGMOID"
        elif self.Activation ==  ActivationType.RELU:
            funcType = "RELU"
        elif self.Activation == ActivationType.SOFTMAX:
            funcType = "SOFTMAX"
        else:
            funcType = "TANH"
        return "(Type: %s, Activation: %s, Size: %d)"%(layer_type, funcType, self.num_neurons)