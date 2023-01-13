# Neural Network Introduction

"""
Neuron: Takes inputs, does (math) with them and produces one output
	- E/a input multi. by a weight
	- All weigthed inputs are added together with a bias
	- This sum is passed though an activation function which spits out 1 output

Feedforward: Process of passing inputs forward to get an output
"""


"""
Neural Network put simply:
	- Is a Network of layers, w/ any number of neurons (input layer, neuron(hidden) layer, output layer)
	- Feed input(s) forward through neurons to get output(s)
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

class NeuralNetwork:

	"""
	- 2 input
	- hidden layer w/ 2 neurons(h1, h2)
	- output layer w/ 1 neuron (o1)

	weight = [0,1]
	bias = 0
	"""


	def __init__(self):
		weights = np.array([0,1])
		bias = 0


		self.h1 = Neuron(weights, bias)
		self.h2 = Neuron(weights, bias)

		self.o1 = Neuron(weights, bias)

	def feedforward(self, x):
		h1_out = self.h1.feedforward(x)
		h2_out = self.h2.feedforward(x)

		o1_out = self.o1.feedforward(np.array([h1_out, h2_out]))

		return o1_out


network = NeuralNetwork()

x = np.array([2,3])
print(network.feedforward(x))








