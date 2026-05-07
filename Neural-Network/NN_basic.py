import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        # layers: e.g., [2, 3, 1] means 2 inputs, 3 hidden, 1 output
        self.weights = [np.random.randn(layers[i], layers[i+1]) for i in range(len(layers)-1)]
        self.biases = [np.zeros((1, layers[i+1])) for i in range(len(layers)-1)]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.activations = [X]
        for i in range(len(self.weights)):
            net_input = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.activations.append(self.sigmoid(net_input))
        return self.activations[-1]

    def backward(self, X, y, learning_rate):
        error = y - self.activations[-1]
        delta = error * self.sigmoid_derivative(self.activations[-1])

        for i in reversed(range(len(self.weights))):
            # Update weights and biases
            self.weights[i] += self.activations[i].T.dot(delta) * learning_rate
            self.biases[i] += np.sum(delta, axis=0, keepdims=True) * learning_rate
            
            # Calculate delta for the next (previous) layer
            if i > 0:
                delta = delta.dot(self.weights[i].T) * self.sigmoid_derivative(self.activations[i])

# Training Loop
# Simple XOR Problem
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork([2, 4, 1])

for epoch in range(10000):
    nn.forward(X)
    nn.backward(X, y, learning_rate=0.1)

print("Final Predictions:")
print(nn.forward(X))
