# External Libraries Imports #
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import numpy as np

# System Imports #
import random

# Matplotlib plot image #
def plot_image(image : np.array, label:str = "", title:str = "") -> None:
    """
    Parameters
    ----------
    image : np.array

    Returns
    ----------
    None

    Notes
    ----------
    Plot a given image
    """

    print(f"The image Class is: {label}")

    # Normalize image colors for matplotlib #
    image = (image - image.min()) / (image.max() - image.min())
    image = np.transpose(image, [1, 2, 0])

    plt.title(title)
    plt.imshow(image)


# Matplotlib plot Tensor as Image #
def plot_figure(dataLoader: DataLoader) -> None:
    """
    Parameters
    ----------
    dataLoader : DataLoader

    Returns
    ----------
    None

    Notes
    ----------
    Plot a random image from the dataloader given.
    """

    # Get loader reference for classes #
    data_Classes = dataLoader.dataset.classes

    # Get Random image and Label #
    random_Index = np.random.randint(len(dataLoader))

    label = data_Classes[dataLoader.dataset[random_Index][1]]
    image = dataLoader.dataset[random_Index][0]

    print(f"The image Class is: {label}")

    # Normalize image colors for matplotlib #
    image = (image - image.min()) / (image.max() - image.min())

    # Plot image #
    plt.imshow(np.transpose(image, (1, 2, 0)))
    plt.axis("off")
    plt.show()


# MatplotLib plot Tensor as Grid #
def plot_figures_grid(
    dataLoader: DataLoader, samples: int = 10, gridsize: [int, int] = [10, 10]
) -> None:
    """
    Parameters
    ----------
    dataLoader : DataLoader
    samples : int = 10, optional
    gridsize : [int, int] = [10, 10], optional

    Returns
    ----------
    None

    Notes
    ----------
    Plot a grid of random images from the dataloader given.
    """

    # Get loader reference for its classes #
    data_Classes = dataLoader.dataset.classes

    plt.figure(figsize=gridsize)

    # Iterate the classes looking for randomized image to show #
    for label, sample in enumerate(data_Classes):
        # Randomize indices for images
        class_indices = np.flatnonzero(label == np.array(dataLoader.dataset.targets))
        sample_indices = np.random.choice(class_indices, samples, replace=False)

        for i, index in enumerate(sample_indices):
            plot_index = i * len(data_Classes) + label + 1
            plt.subplot(samples, len(data_Classes), plot_index)
            plt.imshow(dataLoader.dataset.data[index])
            plt.axis("off")

            if i == 0:
                plt.title(sample)

    # Plot image #
    plt.show()


def plot_minibatch_loss(logging_Dict: dict, loss_per_batch_label: str) -> None:
    """
    Parameters
    ----------
    logging_Dict : dict
    loss_per_batch_label : str

    Returns
    ----------
    None

    Notes
    ----------
    Plot the Loss and average loss of the Trained Network, given a logging dictionary and its key.
    """

    loss_minibatch_list = logging_Dict[loss_per_batch_label]

    plt.plot(loss_minibatch_list, label="Minibatch loss Function")

    # Calculate Average #
    plt.plot(
        np.convolve(
            loss_minibatch_list,
            np.ones(
                200,
            )
            / 200,
            mode="valid",
        ),
        label="Running average",
    )

    # Plot Graph #
    plt.ylabel("Cross Entropy")
    plt.xlabel("Iteration")
    plt.legend()
    plt.show()


def plot_accuracy_epochs(
    logging_Dict: dict, total_epochs: int, accuracy_per_epoch_labels: [str, str]
) -> None:
    """
    Parameters
    ----------
    logging_Dict : dict
    total_epochs : int
    accuracy_per_epoch_labels : [str, str]

    Returns
    ----------
    None

    Notes
    ----------
    Plot the accuracy of the Trained Network for the train and validation datasets, given a logging dictionary and their keys.
    """

    # Arrange data #
    plt.plot(
        np.arange(1, total_epochs + 1),
        logging_Dict[accuracy_per_epoch_labels[0]],
        label="Training",
    )
    plt.plot(
        np.arange(1, total_epochs + 1),
        logging_Dict[accuracy_per_epoch_labels[1]],
        label="Validation",
    )

    # Plot Graph #
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


# Plots the confussion matrix of the trained model #
def plot_confusion_matrix(matrix: any, labels: [], text_show: str) -> None:
    """
    Parameters
    ----------
    matrix : any
    labels : []
    text_show : str

    Returns
    ----------
    None

    Notes
    ----------
    Plots the confusion matrix of the trained network.
    """

    fig, ax = plt.subplots()
    im = ax.imshow(matrix)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(matrix[0])):
        for j in range(len(matrix[0])):
            text = ax.text(j, i, matrix[i, j], ha="center", va="center", color="w")

    ax.set_title(text_show)
    fig.tight_layout()
    plt.show()


# used for plotting the weights of a single layer convolutional layer #
def plot_filters_single_channel(numpy_Array: np.array, columns: int = 10) -> None:
    """
    Parameters
    ----------
    numpy_Array : np.array
    columns : int = 10, optional

    Returns
    ----------
    None

    Notes
    ----------
    Plots the weight of a single channel input convolutional layer.
    """

    # Set number of plots to do per line #
    number_Images = numpy_Array.shape[0] * numpy_Array.shape[1]

    rows = 1 + number_Images // columns

    # Set figure size #
    fig = plt.figure(figsize=(columns, rows))

    # Counter for subplots #
    counter = 0

    # Loop for kernels for plotting
    for i in range(numpy_Array.shape[0]):
        for j in range(numpy_Array.shape[1]):
            counter += 1

            # Add subplot to figure #
            ax1 = fig.add_subplot(rows, columns, counter)

            # Take image from kernel #
            npimg = np.array(numpy_Array[i, j], np.float32)
            npimg = (npimg - np.mean(npimg)) / np.std(npimg)
            npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))

            # Show image with labels to identify kernel #
            ax1.imshow(npimg)
            ax1.set_title(str(i) + "," + str(j))
            ax1.axis("off")
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])

    plt.tight_layout()
    plt.show()


# Used for plotting the weights of a multiple layer convolutional layer #
def plot_filters_multiple_channels(
    tensor: torch.Tensor, kernel: int = 0, columns: int = 10
) -> None:
    """
    Parameters
    ----------
    tensor : torch.Tensor
    kernel : int = 0, optional
    columns : int = 10, optional

    Returns
    ----------
    None

    Notes
    ----------
    Plots the weights of a multiple chanel input convolutional layer.
    """

    # get the number of kernals
    number_kernels = tensor.shape[0]

    # Set row size #
    rows = number_kernels

    # Set the figure size #
    fig = plt.figure(figsize=(columns, rows))

    fig.suptitle("Kernels :" + str(kernel))

    # Loop through all kernels #
    for i in range(tensor.shape[0]):
        # Create subplot #
        ax1 = fig.add_subplot(rows, columns, i + 1)

        # Convert tensor to numpy array #
        image = np.array(tensor[i].numpy(), np.float32)

        # Normalize the numpy array #
        image = (image - np.mean(image)) / np.std(image)
        image = np.minimum(1, np.maximum(0, (image + 0.5)))
        # image = image.transpose((1, 2, 0))

        ax1.imshow(image[kernel])
        ax1.axis("off")
        ax1.set_title(str(i))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.tight_layout()
    plt.show()


# Used for plotting the weights of a multiple layer convolutional layer #
def plot_tensor_array(tensor: torch.Tensor, image: np.array, label: str = "") -> None:
    """
    Parameters
    ----------
    tensor : torch.Tensor
    image : np.array
    label : str = "" , optional

    Returns
    ----------
    None

    Notes
    ----------
    Plots tsidde to side comparison of a tensor and an image.
    """

    # Modify bounds for tensor image #
    tensor_image = tensor[0].cpu()
    tensor_image = tensor_image - tensor_image.min()
    tensor_image /= tensor_image.max()

    # Set axis #
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(3, 2, 1)

    # Plot Image #
    image = np.transpose(image, [1, 2, 0])

    ax1.imshow(image)
    ax1.set_title("Orginal Image")
    ax1.axis("off")
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])

    # Plot Tensor #
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.set_title(label)
    ax1.imshow(np.transpose(tensor_image, [1, 2, 0]))
    ax1.axis("off")
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])

    plt.show()

def plot_model_outputs(layer_outputs: dict, columns:int = 16) -> None:
    """
    Parameters
    ----------
    layer_outputs : dict
    columns : int = 16, optional

    Returns
    ----------
    None

    Notes
    ----------
    Plots convolutional and maxpooling outputs to visualize network performance.
    """

    for key in layer_outputs.keys():
        number_of_outputs = layer_outputs[key][0].shape[0]
        rows = int(number_of_outputs / columns)
        # Set the figure size #
        fig = plt.figure(figsize=(columns, rows))
        fig.suptitle("Layer :" + key)

        for i in range(number_of_outputs):
             # Create subplot #
            ax1 = fig.add_subplot(rows, columns, i + 1)

            # Convert tensor to numpy array #
            image =  layer_outputs[key][0][i].cpu().numpy()

            # Normalize the numpy array #
            #image = (image - np.mean(image)) / np.std(image)
            #image = np.minimum(1, np.maximum(0, (image + 0.5)))

            ax1.imshow(image)
            ax1.axis("off")

        plt.show()
