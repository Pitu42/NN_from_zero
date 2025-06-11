import numpy as np

# z = xw+b
# x = number of inputs
# w = number of weight, 1 per input and a set per neuron
# b = 1 per neuron

# z = b = len(neurons)

np.random.seed(42)

class Architecture():
    # z = Xw + b
    def __init__(self, num_inputs, num_neurons):
        # Generate random values for the weights
        weights = np.random.rand(num_inputs, num_neurons)
        # weights to be used in forward
        self.weights = weights
        # Generate initialization of bias = 1
        bias = np.full((1,num_neurons),1)
        # bias to be used in forward
        self.bias = bias
    def forward(self, inputs):
        # Calculate output of layer
        z = np.dot(inputs,self.weights) + self.bias
        self.outputs = z

class ActivationFunction():
    def ReLU(self, outputs):
        # z = z when z > 0
        self.outputs = np.maximum(outputs, 0)
        pass



X = np.array([1, 2, 3, 4])
# 4 inputs, 5 neurons, 1 output



layer1 = Architecture(4,5)
activation1 = ActivationFunction()

layer1.forward(X)
activation1.ReLU(layer1.outputs)

layer2 = Architecture(5,1)
activation2 = ActivationFunction()

layer2.forward(activation1.outputs)
activation2.ReLU(layer2.outputs)

print(activation2.outputs)
