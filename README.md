# MNISTRecoginition
NN for digit recognition. This version is allows user to build any network with different number
of hidden layer and neurons in each.

Current accuracy is 0.89 using ReLu with 30 neurons in Hidden Layer. Network does not support weight
decay yet (working on that). 

For better accuracy I recommend STOCHASTIC, use MINIBATCH if epochs > 100.

New update included ADAM optimizer capability. Computations are batch-based, finding convergence faster.
