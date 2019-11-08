import torch
from torch import Tensor
from torch.utils.data import DataLoader
from collections import namedtuple

from .losses import ClassifierLoss


class LinearClassifier(object):

    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO:
        #  Create weights tensor of appropriate dimensions
        #  Initialize it from a normal dist with zero mean and the given std.

        self.weights = None
        # ====== YOUR CODE: ======
        w = torch.randn(n_features, n_classes)
        w_mean = torch.mean(w)
        w_std = torch.std(w)
        w = (w - w_mean) / w_std
        w *= weight_std
        # w.requires_grad = True
        self.weights = w
        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO:
        #  Implement linear prediction.
        #  Calculate the score for each class using the weights and
        #  return the class y_pred with the highest score.

        y_pred, class_scores = None, None
        # ====== YOUR CODE: ======
        class_scores = torch.mm(x, self.weights)
        _, y_pred = class_scores.max(1)
        # ========================

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO:
        #  calculate accuracy of prediction.
        #  Use the predict function above and compare the predicted class
        #  labels to the ground truth labels to obtain the accuracy (in %).
        #  Do not use an explicit loop.

        acc = None
        # ====== YOUR CODE: ======
        num_correct = (y == y_pred).sum()
        total = y.shape[0]
        acc = float(num_correct) / total
        # ========================

        return acc * 100

    def train(self,
              dl_train: DataLoader,
              dl_valid: DataLoader,
              loss_fn: ClassifierLoss,
              learn_rate=0.1, weight_decay=0.001, max_epochs=100):

        Result = namedtuple('Result', 'accuracy loss')
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print('Training', end='')
        for epoch_idx in range(max_epochs):
            total_correct = 0
            average_loss = 0

            # TODO:
            #  Implement model training loop.
            #  1. At each epoch, evaluate the model on the entire training set
            #     (batch by batch) and update the weights.
            #  2. Each epoch, also evaluate on the validation set.
            #  3. Accumulate average loss and total accuracy for both sets.
            #     The train/valid_res variables should hold the average loss
            #     and accuracy per epoch.
            #  4. Don't forget to add a regularization term to the loss,
            #     using the weight_decay parameter.

            # ====== YOUR CODE: ======
            epoch_train_acc = 0
            epoch_valid_acc = 0
            epoch_train_loss = 0
            epoch_valid_loss = 0

            for x_train, y_train in dl_train:
                y_train_pred, class_scores = self.predict(x_train)
                train_loss = loss_fn(x_train, y_train, class_scores, y_train_pred)
                train_loss += weight_decay * torch.sum(self.weights ** 2) / 2
                grad = loss_fn.grad()
                grad += weight_decay * self.weights
                self.weights -= learn_rate * grad
                train_acc = self.evaluate_accuracy(y_train, y_train_pred)
                epoch_train_loss += train_loss
                epoch_train_acc += train_acc

            for x_valid, y_valid in dl_valid:
                y_valid_pred, class_scores = self.predict(x_valid)
                valid_loss = loss_fn(x_valid, y_valid, class_scores, y_valid_pred)
                valid_loss += weight_decay * torch.sum(self.weights ** 2) / 2
                valid_acc = self.evaluate_accuracy(y_valid, y_valid_pred)
                epoch_valid_loss += valid_loss
                epoch_valid_acc += valid_acc

            train_num_batches = len(dl_train)
            valid_num_batches = len(dl_valid)
            epoch_train_acc /= train_num_batches
            epoch_valid_acc /= valid_num_batches
            epoch_train_loss /= train_num_batches
            epoch_valid_loss /= valid_num_batches
            train_res.accuracy.append(epoch_train_acc)
            train_res.loss.append(epoch_train_loss)
            valid_res.accuracy.append(epoch_valid_acc)
            valid_res.loss.append(epoch_valid_loss)
            # ========================
            print('.', end='')

        print('')
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO:
        #  Convert the weights matrix into a tensor of images.
        #  The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        w = self.weights
        if has_bias:
            w = w[1:]
        w_images = w.t().view(w.shape[1], *img_shape)
        # ========================

        return w_images


def hyperparams():
    hp = dict(weight_std=0., learn_rate=0., weight_decay=0.)

    # TODO:
    #  Manually tune the hyperparameters to get the training accuracy test
    #  to pass.
    # ====== YOUR CODE: ======
    hp['weight_std'] = 0.01
    hp['learn_rate'] = 0.01
    hp['weight_decay'] = 0.01
    # ========================

    return hp
