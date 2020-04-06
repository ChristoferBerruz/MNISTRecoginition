# MNISTClassification
Neural Network for digit classification. Trained using 1000 images from the MNIST Dataset. Validated with 10000 images.

## Features:
*ADAM Optimizer: 
Uses ADAM optimizer for Gradient Descent. NN.Train(optimizer = 'Adam') to use. Default is ''. If Adam is being used,
set learning_rate = 0.001 for better accuracy (~88%). It achieves convergence in around 10 epochs. 
*Stochastic, Mini-batch, and Batch GD: 
Performs different types of Gradient Descent Algorithms.
*Batch Normalization:
Allows batch normalization by setting batch_norm = True. Yields accuracy (~88%).
*Drop-out and weight decay for regularization: 
Use parameters drop_out : bool, drop_percent : float, reg_val : float.
Weight decay is default as well as no drop-out. For drop-out, we drop in every epoch.
*Activation Functions:
RELU, SOFTMAX, SIGMOID, TANH

