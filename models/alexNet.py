import torch.nn as nn
import torch.nn.functional as F

# Implementation of Alexnet for Cifar 10 dataset #
class CNN_AlexNet(nn.Module):
    """
    Parameters
    ----------
    number_of_classes = 10, optional

    Returns
    ----------
    CNN_AlexNet class Object.

    Notes
    ----------
    Returns an Neural nerwork object, based on the AlexNet paper.
    """

    def __init__(self, number_of_classes: int = 10):
        # Implementation of layers for the Neural network #
        super().__init__()

        # Convolutional Layers Features #
        self.convolutional_features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Average Pooling Features #
        self.average_pooling = nn.AdaptiveAvgPool2d((6, 6))

        # Linear Layers Classification Features #
        self.classifiying_features = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, number_of_classes),
        )

    def forward(self, x):
        convolution_output = self.convolutional_features(x)
        average_pooling_output = self.average_pooling(convolution_output)
        linear_output = average_pooling_output.view(
            average_pooling_output.size(0), 256 * 6 * 6
        )
        classifier_output = self.classifiying_features(linear_output)
        probabilities = F.softmax(classifier_output, dim=1)
        return classifier_output