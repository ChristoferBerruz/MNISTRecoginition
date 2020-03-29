import numpy as np
class ActivationType(object):
    """Simple class that includes some tags for different activation functions"""

    SIGMOID = 1
    TANH = 2
    RELU = 3
    SOFTMAX = 4 #only used in last layer

class ActivationFunctions(object):
    """
    Simple class to wrap well known activation functions
    """
    def sigmoid(s1):
        return 1/(1+np.exp(-s1))

    def relu(s1):
        return np.where(s1 >= 0, s1, 0)

    def tanh(s1):
        return np.tanh(s1)

    def softmax(s1):
        a = np.exp(s1)
        return a/np.sum(a)