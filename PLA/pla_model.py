"""
Created on  2019-10-13
@author: Jingchao Yang

https://medium.com/@thomascountz/19-line-line-by-line-python-perceptron-b6f113b161f3
"""
import numpy as np


class Perceptron(object):

    def __init__(self, no_of_inputs, epochs=100, learning_rate=0.01):
        """

        :param no_of_inputs: how many weights we need to learn
        :param epochs: # of iterations before ending
        :param learning_rate: determine the magnitude of change for weights during each step through training data
        """
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation

    def train(self, training_inputs, labels):
        """
        adjust weights by trying to get label - prediction == 0
        w1 = w0 + a(d - y)x

        w1 = weight updated
        w0 = current weight
        a = learning rate
        d = desired output
        y = actual output
        x = input

        :param training_inputs:
        :param labels:
        :return:
        """
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)

                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)


training_inputs = []
labels = []

# generating training sets (1000 records)
for i in range(1000):
    x = np.random.randint(2, size=2)
    training_inputs.append(x)
    if x[0] == 1 and x[1] == 1:
        labels.append(1)
    else:
        labels.append(0)
labels = np.asarray(labels)

perceptron = Perceptron(2)  # 2 inputs plus a bis
perceptron.train(training_inputs, labels)  # train the perceptron

inputs = np.array([1, 1])  # test
print(perceptron.predict(inputs))
# => 1

inputs = np.array([1, 0])  # test
print(perceptron.predict(inputs))
# => 0
