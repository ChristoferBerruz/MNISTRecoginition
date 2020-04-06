"""
All classes in Optimizers define a method newWeight, and newBias which is used in training.
You can add different kind of optimizer as long as those conditions are true.
Please take a look at the function signature
"""

class Adam(object):
    """Adam Optimizer"""
    BETA1 = 0.9
    BETA2 = 0.999
    EPSILON = 10**(-8)

    def __init__(self):
        self.momentumW = 0.0
        self.momentumB = 0.0
        self.velocityW = 0.0
        self.velocityB = 0.0

    def newWeight(self, W, gradW, learning_rate):
        curmomentW = self.BETA1*self.momentumW + (1-self.BETA1)* gradW
        curvelocityW = Adam.BETA2*self.velocityW + (1-Adam.BETA2)*(gradW**2)
        moment_hatW = curmomentW/(1-Adam.BETA1)
        velocity_hatW = curvelocityW/(1-Adam.BETA2)
        newW = W - learning_rate*moment_hatW/(velocity_hatW**0.5 + Adam.EPSILON)
        self.momentumW = curmomentW
        self.velocityW = curvelocityW
        return newW

    def newBias(self, b, gradb, learning_rate):
        curmomentB = Adam.BETA1*self.momentumB  + (1-Adam.BETA1)*gradb
        curvelocityB = Adam.BETA2*self.velocityB + (1-Adam.BETA2)*(gradb**2)
        moment_hatB = curmomentB/(1-Adam.BETA1)
        velocity_hatB = curvelocityB/(1-Adam.BETA2)
        newB = b - learning_rate*moment_hatB/(velocity_hatB**0.5 + Adam.EPSILON)
        self.momentumB = curmomentB
        self.velocityB = curvelocityB
        return newB


class DefaultOptimizer(object):
    """
    The default optimizer is the standar formula to get new parameters theta
    based on existing parameters, and the gradients
    """

    def __init__(self):
        pass

    def newWeight(self, W, gradW, learning_rate):
        return W - learning_rate*gradW

    def newBias(self, b, gradb, learning_rate):
        return b - learning_rate*gradb

    def newTheta(self, theta, gradtheta, learning_rate):
        return theta - learning_rate*gradtheta