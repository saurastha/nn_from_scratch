import numpy as np


class LinearLayer:
    def __init__(self, in_features, out_features):
        self.weights = 0.01 * np.random.randn(in_features, out_features)
        self.biases = np.zeros((1, out_features))
        self.output = None

    def forward(self, x):
        self.output = np.dot(x, self.weights) + self.biases




