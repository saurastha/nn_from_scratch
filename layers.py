import numpy as np


class LinearLayer:
    """
        Linear layer for neural network computations.

        This class represents a linear layer in a neural network, which performs
        a linear transformation on the input data using weights and biases.

        Attributes:
            weights (numpy.ndarray): The weights matrix for the linear transformation.
            biases (numpy.ndarray): The biases matrix for the linear transformation.
            output (numpy.ndarray): The result of the forward pass.

        Methods:
            forward(x): Performs the forward pass of the linear layer.

        Example:
            linear_layer = LinearLayer(in_features=5, out_features=3)
            input_data = np.random.randn(10, 5)
            linear_layer.forward(input_data)
            print(linear_layer.output)
    """
    def __init__(self, in_features: int, out_features: int):
        self.weights = 0.01 * np.random.randn(in_features, out_features)
        self.biases = np.zeros((1, out_features))
        self.output = None

    def forward(self, x: np.array) -> np.array:
        self.output = np.dot(x, self.weights) + self.biases




