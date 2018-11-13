#!/usr/bin/python
import numpy as np

## define a set of measurements m1(array of length) and m2 (array of height) as inputs to neural net
## define a the weights and biases used for backpropagation
## square_error aka cost function = (network_prediction - desired_output)^2


def sigmoid(x):
  """logistic curve to squash the values in interval [0,1]
  aka activation function"""
    return 1/(1 + np.exp(-x))
  
def cost(b):
    return (b-4) ** 2

def slope(b):
    return 2*(b-4)
  
 def training_loop(iterations):
     for i in range(int(iterations)):
         b = b - .1 * slope(b)

def neural_net(m1, m2, w1, w2, b):
  """ net as a function without hidden layers """
      z = w1 * m1 + w2 * m2 + b
      return sigmoid(z)


