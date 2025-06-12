import main as nn
import numpy as np
import pandas as pd

# Imported data from: https://www.kaggle.com/competitions/digit-recognizer
data = pd.read_csv("train.csv")

# Shuffle data
data = data.sample(frac = 1, random_state=42).reset_index(drop=True)

# Calculate the number of records for the test set (20%)
test_size = int(0.2 * len(data))

# Take the slice for the test set
test_data = data.iloc[:test_size]
# Take the slice for the training set (80%)
train_data = data.iloc[test_size:]

# Split data between y and X features
y = np.array(train_data["label"])
X = np.array(train_data.drop("label", axis=1))

# from gradient to binary in order to simplify the model
X[X>0]=1

# Split features for test data
y_test = np.array(test_data["label"])
X_test = np.array(test_data.drop("label", axis=1))
X_test[X_test>0]=1

# Create a first layer with 784 neurons in the input layer and 64 in the second layer
layer1 = nn.Architecture(784,64)
# ReLU as activation function for the first layer 
activation1 = nn.ReLU()

# Create second layer with 64 input neurons and 10 in the output (10 classes)
layer2 = nn.Architecture(64,10)
# Softmax as activation function
activation2 = nn.SoftMax()
# Cost function categorical cross-entropy
cost = nn.CrossEntropy()

# Learning rate 0.3
optimize = nn.Optimizer(learning_rate=0.3)

# Training
for i in range(1001):
    layer1.forward(X)
    activation1.forward(layer1.outputs)

    layer2.forward(activation1.outputs)
    activation2.forward(layer2.outputs)

    cost.forward(activation2.outputs, y)
    predictions = np.argmax(cost.predictions, axis=1)
    accuracy = np.mean(predictions==y)

    activation2.backward(y)

    layer2.backward(activation2.dinputs)
    activation1.backward(layer2.dinputs)
    layer1.backward(activation1.dinputs)

    optimize.update(layer1)
    optimize.update(layer2)

    if i % 50 == 0:
        print(f'step {i}: loss {cost.loss:.3f}' + f' accuracy {accuracy:.3f}')

# Testing
layer1.forward(X_test)
activation1.forward(layer1.outputs)

layer2.forward(activation1.outputs)
activation2.forward(layer2.outputs)

cost.forward(activation2.outputs, y_test)
predictions = np.argmax(cost.predictions, axis=1)
accuracy = np.mean(predictions==y_test)

print(f'validation: loss {cost.loss:.3f}' + f' accuracy {accuracy:.3f}')
