import numpy as np

class Architecture():
    # z = Xw + b
    def __init__(self, num_inputs, num_neurons):
        # Generate random small values for the weights
        weights = 0.1 * np.random.randn(num_inputs, num_neurons)
        # weights to be used in forward
        self.weights = weights
        # Generate initialization of bias = 1
        bias = np.full((1, num_neurons), 0.01)
        # bias to be used in forward
        self.bias = bias
    def forward(self, inputs):
        # Remember the inputs
        self.inputs = inputs
        # Calculate output of layer
        self.outputs = np.dot(inputs,self.weights) + self.bias
    def backward(self, dvalues):
        # Gradient
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dinputs = np.dot(dvalues, self.weights.T)
        self.dbias = np.sum(dvalues, axis=0, keepdims=True)

class ReLU():
    def forward(self, inputs):
        # Rember inputs
        self.inputs = inputs
        # z = z when z > 0
        self.outputs = np.maximum(inputs, 0)
    def backward(self, dvalues):
        # Copy dL/dz
        self.dinputs = dvalues.copy()
        # 0 when ReLU(z) < 0
        self.dinputs[self.inputs < 0] = 0

class SoftMax():
    def forward(self, outputs):
        # exp(z)/sum(exp(z))
        self.inputs = outputs
        exp_values = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))  # Stability fix
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.outputs = probabilities
    def backward(self, target):
        # dL/dz = y_pred - target
        # To one hot encode
        if len(target.shape) == 1:
            target = np.eye(self.outputs.shape[1])[target]
        self.dinputs = (self.outputs - target)  / target.shape[0]

class CrossEntropy():
    def forward(self, predictions, target):
        self.predictions = predictions
        self.target = target
        self.samples = target.shape[0]
        # one hot encoder
        if len(target.shape) == 1:
            target = np.eye(predictions.shape[1])[target]
        self.target = target
        # Always loss > 0
        loss = -np.sum(target * np.log(predictions + 1e-9)) / self.samples
        self.loss = loss

class Optimizer():
    def __init__(self, learning_rate=1):
        self.learning_rate = learning_rate
    def update(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.bias -= self.learning_rate * layer.dbias
        self.dweights = layer.weights
        self.dbias = layer.bias
