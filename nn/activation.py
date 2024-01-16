import numpy as np
from loss import CategoricalCrossEntropy


class ReLU:
    """
        Rectified Linear Unit (ReLU) activation function.

        The ReLU activation function sets all negative values in the input to zero
        and leaves positive values unchanged.

        Attributes:
            output (numpy.ndarray): The result of the forward pass.

        Methods:
            forward(x): Performs the forward pass of the ReLU activation function.

        Example:
            relu = ReLU()
            input_data = np.array([-2, -1, 0, 1, 2])
            relu.forward(input_data)
            print(relu.output)  # Output: [0, 0, 0, 1, 2]
    """

    def forward(self, x: np.array) -> np.array:
        self.inputs = x
        self.output = np.maximum(0, x)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Softmax:
    """
        Softmax activation function for converting input values into probabilities.

        The Softmax function takes an input array, calculate exponential of each element,
        and normalizes the values to obtain probabilities that sum to 1.

        Attributes:
            output (numpy.ndarray): The result of the forward pass representing probabilities.

        Methods:
            forward(x): Performs the forward pass of the Softmax activation function.

        Example:
            softmax = Softmax()
            input_data = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
            softmax.forward(input_data)
            print(softmax.output)
            # Output: array([[0.09003057, 0.24472847, 0.66524096],
            #                 [0.01587624, 0.11731043, 0.86681333]])
    """

    def __init__(self):
        self.output = None

    def forward(self, x: np.array) -> np.array:
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))

        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)

            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class SoftmaxCategoricalCrossEntropy():
    def __init__(self):
        self.activation = Softmax()
        self.loss = CategoricalCrossEntropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()

        self.dinputs[range(samples), y_true] -= 1

        self.dinputs = self.dinputs / samples
