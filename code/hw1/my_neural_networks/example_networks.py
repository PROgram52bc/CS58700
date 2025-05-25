import logging
import torch
from collections import OrderedDict


class TorchNeuralNetwork:
    """Implementation that uses torch.nn

        Neural network classifier with cross-entropy loss
        and ReLU activations
    """
    def __init__(self, shape, gpu_id=-1):
        """Initialize the network

        Args:
            shape: a list of integers that specifieds
                    the number of neurons at each layer.
            gpu_id: -1 means using cpu. 
        """
        self.shape = shape
        logging.info("shape = {}".format(shape))
        # define the network structure.
        # always use ReLU on the hidden layers
        # and Softmax on the output layer
        assert len(shape) >= 2, \
               "at least an input and output layer have to be specified"

        # add the first linear layer
        net_dict = OrderedDict([('W_0', torch.nn.Linear(shape[0], shape[1]))])
        for idx, (i, j) in enumerate(zip(shape[1:-1], shape[2:])):
            # add activation, assume using ReLU
            key = 'a_{}'.format(idx)
            net_dict[key] = torch.nn.ReLU()

            # add the next linear layer
            key = 'W_{}'.format(idx+1)
            net_dict[key] = torch.nn.Linear(i, j)

        # declair the network
        if gpu_id == -1:
            self.model = torch.nn.Sequential(net_dict)
        else:
            self.model = torch.nn.Sequential(net_dict).cuda(gpu_id)
        logging.info("self.model = {}".format(self.model))

        # define loss function
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def train_one_epoch(self, X, y, y_1hot, learning_rate):
        """Train for one epoch

        Args:
            X: (n_examples, n_features)
            y: (n_examples)
            y_1hot: (n_examples, n_classes)

        Returns:
            Loss
        """
        X_train = X
        y_train = y

        # forward
        y_pred = self.model(X_train)
        loss = self.loss_fn(y_pred, y_train.squeeze())

        # backward
        self.model.zero_grad()
        loss.backward()

        # update weights and biases
        for param in self.model.parameters():
            param.data -= learning_rate * param.grad.data
        return loss.item()
    
    def loss(self, X, y, y_1hot):
        """Compute feed forward loss

        Args:
            X: (n_examples, n_features)
            y: (n_examples). don't care
            y_1hot: (n_examples, n_classes)

        Returns:
            Loss for the given input
        """
        y_pred = self.model(X)
        loss = self.loss_fn(y_pred, y.squeeze())
        return loss.item()

    def predict(self, X):
        """Predict

            Make predictions for X using the current model

        Args:
            X: (n_examples, n_features)

        Returns:
            (n_examples, )
        """
        y_pred = self.model(X)
        return torch.max(y_pred, 1)[1]
