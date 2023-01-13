# Neural Network Introduction

"""
Neuron: Takes inputs, does (math) with them and produces one output
	- E/a input multi. by a weight
	- All weigthed inputs are added together with a bias
	- This sum is passed though an activation function which spits out 1 output

Feedforward: Process of passing inputs forward to get an output
"""

import numpy as np


def sigmoid(x):
	# Activation function f(x) = 1 / (1 + e^(-x))
	return 1 / (1 + np.exp(-x))

class Neuron:
	"""docstring for ClassName"""
	def __init__(self, weight, bias):
		self.weight = weight
		self.bias = bias
	
	def feedforward(self, input):
		total = np.dot(self.weight, input) + self.bias
		return sigmoid(total)



weight = np.array([0,1])
bias = 4
n = Neuron(weight, bias)

x = np.array([2,3])

print(n.feedforward(x))