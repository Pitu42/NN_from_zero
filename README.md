# NN_from_zero
This project is a neural network implemented from scratch in Python using only NumPy.
Although modern libraries are vastly more efficient and feature-rich, I wanted to challenge myself by building a complete feedforward neural network using just mathematical operations.

The NN can:
- Solve classification tasks
- Support arbitrary layer sizes
- Use ReLU activation function for hidden layers and Softmax in the output layer
- Train using categorical cross-entropy loss
- Train using backpropagation and gradient descent

## Architecture Overview
The network supports multiple layers and operates in two main phases:
### 1. Forward Propagation
Each neuron computes a weighted sum of its inputs plus a bias, then applies an activation function to produce an output:
``` math
\begin{align}
z=\vec{w}.\vec{x}+b \\
o=(\vec{x})=s(z) 
\end{align}
```
Where:
- $\vec{w}$ is the weight vector of the neuron
- $\vec{x}$ is the input vector
- $b$ is the bias
- $s$ is the activation function (ReLU or Softmax)

For a full layer, we vectorize the computation:
``` math
O(X)=s(W.X+\vec{b})
```
- $X\in\mathbb{R}^{nxd}$: batch of $n$ input vectors of dimension $d$
- $W\in\mathbb{R}^{dxm}$: weights matrix mapping to $m$ neurons
- $\vec{b}\in\mathbb{R}^{1xm}$: bias vector shared across inputs
- $s$: activation function

### ReLU
$ReLU(z) = max(0,z)$
### Softmax
$Softmax(z_i)=\frac{e^{z_i}}{\sum_je^{z_j}}$
### 2. backpropagation
Backpropagation computes the gradients of the loss function with respect to each parameter in the network using the chain rule. The loss function used is categorical cross-entropy:\
$L = -\sum_{i=1}^C y_i log(\hat{y}_i)$\
Where:
- $C$ is the number of classes
- $y$ is the hot encoded true label
- $\hat{y}$ is the predicted sofmax probability
Gradients are calculated layer by layer starting from the output:\
``` math
\begin{align}
\frac{\partial L}{\partial w} &= \frac{\partial L}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial z} \frac{\partial z}{\partial w} \\
\frac{\partial L}{\partial z} &= \hat{y}-y \\
\frac{\partial z}{\partial w} &= x \\
\frac{\partial z}{\partial x} &= w \\
\frac{\partial z}{\partial b} &= 1
\end{align}
```
The parameters are updated using simple stochastic gradient descent.
### Weight and bias initialization
- Weights are initialized with numpy.random.randn() scaled by 0.1
- Biases are initialized to 0.01 for better symmetry breaking
### Results
Achieves ~95% accuracy on the MNIST digit classification task (0–9)

### Disclaimer
This project is educational and not optimized for production use. It is slow compared to frameworks like PyTorch or TensorFlow, but it gives full control and visibility into the internal operations of a neural network. 
Before building the neuronal network I read a lot about it, including the book "Neural Networks from Scratch" https://nnfs.io/ . Therefore the structure and funcions are similars to the one in the book
