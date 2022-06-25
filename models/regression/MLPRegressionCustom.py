import numpy as np
from models import model

import numpy as np


class MLP:
    """ A Multi-Layer Perceptron"""

    def __init__(self, X, y, number_hidden=3, eta=0.25, niterations=101, beta=1, momentum=0.9, outtype='logistic',
                 metric=None, encoder=None):
        """ Constructor """
        # Set up network size
        self.nin = np.shape(X)[1]
        self.nout = np.shape(y)[1]
        self.ndata = np.shape(X)[0]
        self.nhidden = number_hidden

        self.beta = beta
        self.momentum = momentum
        self.outtype = outtype
        self.eta = eta
        self.niterations = niterations

        # Initialise network
        self.weights1 = (np.random.rand(self.nin + 1, self.nhidden) - 0.5) * 2 / np.sqrt(self.nin)
        self.weights2 = (np.random.rand(self.nhidden + 1, self.nout) - 0.5) * 2 / np.sqrt(self.nhidden)
        self.errors_train = []
        self.errors_valid = []
        self.metric = metric
        self.encoder = encoder

    def earlystopping(self, X_train, y_train, X_test, y_test):

        X_test = np.concatenate((X_test, -np.ones((np.shape(X_test)[0], 1))), axis=1)

        old_val_error1 = 100002
        old_val_error2 = 100001
        new_val_error = 100000

        self.fit(X_train, y_train)

        count = 0
        while (((old_val_error1 - new_val_error) > 0.001) or ((old_val_error2 - old_val_error1) > 0.001)):
            count += 1
            print(count)
            # self.fit(X_train, y_train)
            old_val_error2 = old_val_error1
            old_val_error1 = new_val_error
            validout = self.predict(X_test)

            if self.metric is None:
                new_val_error = 0.5 * np.sum((y_test - validout) ** 2)  # TODO Função custo
            else:
                new_val_error = self.metric.measure(y_test, validout)

            self.errors_valid.append(new_val_error)

        print("Stopped", new_val_error, old_val_error1, old_val_error2)
        return new_val_error

    def fit(self, X, _y, X_test=None, y_test=None):
        """ Train the thing """
        # Add the inputs that match the bias node
        X = np.concatenate((X, -np.ones((self.ndata, 1))), axis=1)
        X_test = np.concatenate((X_test, -np.ones((np.shape(X_test)[0], 1))), axis=1)

        updatew1 = np.zeros((np.shape(self.weights1)))
        updatew2 = np.zeros((np.shape(self.weights2)))

        for n in range(self.niterations):

            if X_test is not None:
                _y_test = self.predict(X_test)
                if self.metric is None:
                    error_valid = 0.5 * np.sum((y_test - _y_test) ** 2)  # TODO Função custo
                elif self.outtype in ["logistic", "softmax"]:
                    y1 = self.encoder.inverse_transform(y_test)
                    y2 = self.encoder.inverse_transform(_y_test)
                    error_valid = self.metric.measure(y1, y2)
                else:
                    error_valid = self.metric.measure(y_test, _y_test)
                self.errors_valid.append(error_valid)

            self.y = self.predict(X)

            if self.metric is None:
                error = 0.5 * np.sum((self.y - _y) ** 2)  # TODO Função custo
            elif self.outtype in ["logistic", "softmax"]:
                y1 = self.encoder.inverse_transform(self.y)
                y2 = self.encoder.inverse_transform(_y)
                error = self.metric.measure(y1, y2)
            else:
                error = self.metric.measure(self.y, _y)

            self.errors_train.append(error)

            if (np.mod(n, 100) == 0):
                print("Iteration: ", n, " Error: ", error)

            if self.outtype == 'linear':
                delta_output = (self.y - _y) / self.ndata
            elif self.outtype == 'logistic':
                delta_output = self.beta * (self.y - _y) * self.y * (1.0 - self.y)
            elif self.outtype == 'softmax':
                delta_output = (self.y - _y) * (self.y * (-self.y) + self.y) / self.ndata
            else:
                print("error")

            delta_hidden = self.hidden * self.beta * (1.0 - self.hidden) * (np.dot(delta_output, self.weights2.T))

            updatew1 = self.eta * (np.dot(X.T, delta_hidden[:, :-1])) + self.momentum * updatew1
            updatew2 = self.eta * (np.dot(self.hidden.T, delta_output)) + self.momentum * updatew2
            self.weights1 -= updatew1
            self.weights2 -= updatew2

    def predict(self, inputs):
        """ Run the network forward """

        self.hidden = np.dot(inputs, self.weights1);
        self.hidden = 1.0 / (1.0 + np.exp(-self.beta * self.hidden))  # TODO funcao ativação sigmode
        self.hidden = np.concatenate((self.hidden, -np.ones((np.shape(inputs)[0], 1))), axis=1)

        outputs = np.dot(self.hidden, self.weights2);

        if self.outtype == 'linear':
            return outputs
        elif self.outtype == 'logistic':
            return 1.0 / (1.0 + np.exp(-self.beta * outputs))
        elif self.outtype == 'softmax':
            normalisers = np.sum(np.exp(outputs), axis=1) * np.ones((1, np.shape(outputs)[0]))
            return np.transpose(np.transpose(np.exp(outputs)) / normalisers)
        else:
            print("error")

# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     from sklearn.model_selection import train_test_split
#     from models.metrics.regression import RMSE
#     from sklearn.preprocessing import MinMaxScaler
#
#     df = np.genfromtxt('../../Lista4/concrete.csv', delimiter=',')
#     X = df[:, :-1]
#     y = df[:, -1].reshape(-1, 1)
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
#
#     X_train = MinMaxScaler().fit_transform(X_train)
#     y_train = MinMaxScaler().fit_transform(y_train)
#     X_test = MinMaxScaler().fit_transform(X_test)
#     y_test = MinMaxScaler().fit_transform(y_test)
#
#     net = MLP(X_train, y_train, 10, outtype='linear', metric=RMSE())
#     net.fit(X_train, y_train, X_test, y_test)
#     plt.plot(net.errors_valid)
#     plt.plot(net.errors_train)
