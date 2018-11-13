#!/usr/bin/python
import numpy as np

## define a set of measurements m1(array of length) and m2 (array of height) as inputs to neural net
## define a the weights and biases used for backpropagation

## mean_square_error MSE = estimated_value(prediction) - estimator(dataset)
## for back propagation we need to chage the estimated_value using the bias b
## mean_square_error aka cost function = (network_prediction - desired_output)^2


def cost(b):
    return (b-4) ** 2

def slope(b):
  """ cost function derivative with respect to b """
    return 2*(b-4)
  
  
def sigmoid(x):
  """logistic curve to squash the values in interval [0,1]
  aka activation function"""
    return 1/(1 + np.exp(-x))
  

  
 def training_loop(iterations):
     for i in range(int(iterations)):
         b = b - .1 * slope(b)

def neural_net(m1, m2, w1, w2, b):
  """ net as a function without hidden layers """
      z = w1 * m1 + w2 * m2 + b
      return sigmoid(z)
    
  if __name__ == "__main__":
      #give random values to weights and bias
      w1 = np.random.rand()
      w2 = np.random.rand()
      b = np.rand.rand()
      print(neural_net(3,1.5,w1,w2,b)) 


