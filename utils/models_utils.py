# External Libraries Imports #
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as datasets
import torchvision.transforms as T
import numpy as np

import captum
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution
from captum.attr import visualization as viz

# System Libraries #
import time
from itertools import product
from utils.matplotlib_utils import plot_filters_single_channel, plot_filters_multiple_channels, plot_tensor_array, plot_image
from utils.dataset_utils import get_item


# Calculate Accuracy of Model #
def model_accuracy(model: any, dataLoader: DataLoader, device: torch.device) -> float:
    """
    Parameters
    ----------
    model : any
    dataLoader : DataLoader
    device : torch.device

    Returns
    ----------
    float

    Notes
    ----------
    Function that calculates Model accuracy for a given dataloader.
    """

    # Evaluation of Model #
    model.eval()
    with torch.no_grad():
        # Prediction variables #
        correct_pred: int = 0
        num_examples: int = 0

        # Extract features from Dataloader #
        for i, (features, targets) in enumerate(dataLoader):
            # Change images to analysis device #
            images = features.to(device)
            labels = targets.to(device)

            # Calculate Prediction for input #
            prediction = model(images)

            # Get the max value from the prediction array #
            _, predicted_labels = torch.max(prediction, 1)

            # Set per batch #
            num_examples += targets.size(0)

            # Analyze Prediction #
            correct_pred += (predicted_labels == labels).sum()

    return correct_pred.float() / num_examples * 100


# Evaluate Loss per Epoch #
def model_loss_epoch(model: any, dataLoader: DataLoader, device: torch.device) -> float:
    """
    Parameters
    ----------
    model : any
    dataLoader : DataLoader
    device : torch.device

    Returns
    ----------
    float

    Notes
    ----------
    Function the epoch loss of a model for a given Dataloader.
    """

    # Evaluation of Model #
    model.eval()
    with torch.no_grad():
        # Prediction variables #
        curr_loss: float = 0.0
        num_examples: int = 0

        # Extract features from Dataloader #
        for features, targets in dataLoader:
            # Send images to analysis device #
            images = features.to(device)
            labels = targets.to(device)

            # Calculate Prediction for input #
            prediction = model(images)

            # Calculate loos per epoch using reguction #
            loss = F.cross_entropy(prediction, labels, reduction="sum")

            # Count loss #
            num_examples += targets.size(0)
            curr_loss += loss

        return curr_loss / num_examples


# Show confusion matrix after training #
def compute_confusion_matrix(
    model: any, data_loader: DataLoader, device: torch.device
) -> np.array:
    """
    Parameters
    ----------
    model : any
    dataLoader : DataLoader
    device : torch.device

    Returns
    ----------
    np.array

    Notes
    ----------
    Function that calculates the confussion matrix of the trained model for a given dataloader.
    """

    prediction_targets, model_prediction = [], []

    # Dont optimize model #
    with torch.no_grad():
        for i, (features, targets) in enumerate(data_loader):
            # Send data to devices #
            images = features.to(device)
            labels = targets.to(device)

            # Calculate Prediction #
            prediction = model(images)

            # Get most probable prediction #
            _, predicted_labels = torch.max(prediction, 1)

            # Send back data to cpu #
            model_prediction.extend(predicted_labels.to("cpu"))
            prediction_targets.extend(labels.to("cpu"))

    # Convert to Numpy arrays for Analysis #
    model_prediction = np.array(model_prediction)
    prediction_targets = np.array(prediction_targets)

    # Concatenate and check for unique labels #
    class_labels = np.unique(np.concatenate((prediction_targets, model_prediction)))

    # Check for shape of labels #
    if class_labels.shape[0] == 1:
        if class_labels[0] != 0:
            class_labels = np.array([0, class_labels[0]])
        else:
            class_labels = np.array([class_labels[0], 1])

    n_labels = class_labels.shape[0]

    lst = []

    z = list(zip(prediction_targets, model_prediction))

    # Append repeated labels to check for prediction errors #
    for combi in product(class_labels, repeat=2):
        lst.append(z.count(combi))

    # Append labels strings #
    labels = []
    data_Classes = data_loader.dataset.classes

    for i in range(len(class_labels)):
        labels.append(data_Classes[data_loader.dataset[i][1]])

    # Return Numpy Array for plotting #
    return np.asarray(lst)[:, None].reshape(n_labels, n_labels), data_loader.dataset.classes


def compute_total_accuracy(
    model: any,
    train_dataLoader: DataLoader,
    validation_dataLoader: DataLoader,
    test_dataLoader: DataLoader,
    device: torch.device,
) -> None:
    """
    Parameters
    ----------
    train_dataLoader : DataLoader
    validation_dataLoader : DataLoader
    test_dataLoader : DataLoader
    device : torch.device

    Returns
    ----------
    None

    Notes
    ----------
    Function that calculates the the total acuracy of the model for al three dataloaders and prints it to console.
    """

    # Calculate Accuracy #
    with torch.set_grad_enabled(False):
        train_accuracy = model_accuracy(
            model=model, dataLoader=train_dataLoader, device=device
        )

        validation_accuracy = model_accuracy(
            model=model, dataLoader=validation_dataLoader, device=device
        )

        test_accuracy = model_accuracy(
            model=model, dataLoader=test_dataLoader, device=device
        )

    print(f"Train ACC: {train_accuracy:.2f}%")
    print(f"Validation ACC: {validation_accuracy:.2f}%")
    print(f"Test ACC: {test_accuracy:.2f}%")


# Training Loop #
def train_cnn(
    epochs: int,
    model: any,
    optimizer: any,
    scheduler: any,
    device: torch.device,
    training_loader: DataLoader,
    validation_loader: DataLoader = None,
    loss_function: any = None,
    evaluate: bool = True,
    logging_interval: int = 64,
    log: bool = True,
) -> dict:
    """
    Parameters
    ----------
    epochs : int
    model : any
    optimizer : any
    scheduler: any
    device : torch.device
    training_loader : DataLoader
    validation_loader : DataLoader = None, optional
    loss_function : any = None, optional
    evaluate : bool = True, optional
    logging_interval : int = 64, optional
    log : bool = True, optional

    Returns
    ----------
    dict

    Notes
    ----------
    Main function that trains the neural network and calculates different metrics for it.
    """

    # Dictionary for plots and Data  #
    logging_dict = {
        "train_loss_batch": [],
        "train_loss_epoch": [],
        "train_accuracy_epoch": [],
        "validation_loss_epoch": [],
        "validation_accuracy_epoch": [],
    }

    # Select Loss function  #
    if loss_function is None:
        loss_fn = F.cross_entropy

    # Start of Training #
    start_time = time.time()
    for epoch in range(epochs):
        # Training model start #
        model.train()

        for batch_index, (feature, target) in enumerate(training_loader):
            images = feature.to(device)
            labels = target.to(device)

            # Forward Propagation #
            prediction = model(images)
            loss = loss_fn(prediction, labels)
            optimizer.zero_grad()

            # Backwards Propagation #
            loss.backward()

            # Model Parameters Update #
            optimizer.step()

            # Data Logging#
            logging_dict["train_loss_batch"].append(loss.item())

            if (not batch_index % logging_interval) and (log):
                print(
                    "Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f"
                    % (epoch + 1, epochs, batch_index, len(training_loader), loss)
                )

        # Update Learning Rate #
        scheduler.step()

        if evaluate:
            # Evaluation model start #
            model.eval()

            with torch.set_grad_enabled(False):
                # Calculate Accuracy and loss #
                train_loss = model_loss_epoch(model, training_loader, device)
                train_acc = model_accuracy(model, training_loader, device)

                # Log to console #
                if log:
                    print(
                        "***Epoch: %03d/%03d | Train. Acc.: %.3f%% | Loss: %.3f"
                        % (epoch + 1, epochs, train_acc, train_loss)
                    )

                # Save Data to dictionary #
                logging_dict["train_loss_epoch"].append(train_loss.item())
                logging_dict["train_accuracy_epoch"].append(train_acc.item())

                if validation_loader is not None:
                    # Calculate Accuracy and loss #
                    valid_loss = model_loss_epoch(model, validation_loader, device)
                    valid_acc = model_accuracy(model, validation_loader, device)

                    # Log to console #
                    if log:
                        print(
                            "***Epoch: %03d/%03d | Valid. Acc.: %.3f%% | Loss: %.3f"
                            % (epoch + 1, epochs, valid_acc, valid_loss)
                        )

                    # Save Data to dictionary #
                    logging_dict["validation_loss_epoch"].append(valid_loss.item())
                    logging_dict["validation_accuracy_epoch"].append(valid_acc.item())

        if log:
            print("Time elapsed: %.2f min" % ((time.time() - start_time) / 60))

    if log:
        print("Total Training Time: %.2f min" % ((time.time() - start_time) / 60))

    return logging_dict


def save_model(model: any, path: str = "./") -> None:
    """
    Parameters
    ----------
    model : any
    path : str = "./" , optional

    Returns
    ----------
    None

    Notes
    ----------
    Function for saving model in a given path
    """

    torch.save(model.state_dict(), path)


def load_model(model_Class: any, path: str = "./") -> any:
    """
    Parameters
    ----------
    model_Class:any
    path:str="./"

    Returns
    ----------
    any, returns model object

    Notes
    ----------
    Function for loading model in a given path to its object.
    """
    
    model_Class.load_state_dict(torch.load(path))
    model_Class.eval()
    return model_Class


# Helper for getting model weights #
def get_convolutional_layer_weights(
    model_dict: dict,
    dict_key: str,
    layer_index: int,
    multiple_channels: bool = True,
    multiple_channels_index: int = 0,
    columns: int = 10,
) -> None:
    """
    Parameters
    ----------
    model : dict,
    dict_key: str,
    layer_index : int
    multiple_channels : bool = True , optional
    multiple_channels_index : bool = 0 , optional

    Returns
    ----------
    None.

    Notes
    ----------
    Function for obtaining model weights for a given layer index.
    """

    # Extract layer from models to get its features #
    try:
        weight_tensor = model_dict[
            dict_key + "." + str(layer_index) + ".weight"
        ]
    except:
        print("Cant get weights from non convolutional layer.")
        return

    if multiple_channels:
        plot_filters_multiple_channels(
            weight_tensor.to("cpu"), kernel=multiple_channels_index, columns=columns
        )
    else:
        # Check for single channel kernel #
        if weight_tensor.shape[1] == 3:
            # kernels depth * number of kernels
            plot_filters_single_channel(
                np.array(weight_tensor.to("cpu").numpy(), np.float32), columns=columns
            )

        else:
            print("Cant get weights from multiple layers without multiple channels.")


def get_integrated_gradient(
    model: any,
    dataloader: DataLoader,
    steps: int = 200,
    device: torch.device = torch.device("cuda"),
) -> None:
    """
    Parameters
    ----------
    model : any
    dataloader : Dataloader
    steps : int = 200 , optional
    device : torch.device = torch.device("cpu") , optional

    Returns
    ----------
    None.

    Notes
    ----------
    Function for obtaining integrated gradients for a given tensor.
    """

    # Get random image, index and label #
    tensor, index, label = get_item(dataloader, device)

    # Create integrated gradient object #
    integrated_gradients = IntegratedGradients(model)

    attrib_image = integrated_gradients.attribute(tensor, target=index, n_steps=steps)

    # Convert tensor to numpy array for plotting#
    image = tensor.to("cpu").numpy()[0]

    # Specify target label #
    print("Target label: " + label)

    plot_tensor_array(attrib_image, image, "Integrated Gradient")


def get_occlusion(
    model: any,
    dataloader: DataLoader,
    device: torch.device = torch.device("cuda"),
) -> None:
    """
    Parameters
    ----------
    model : any
    dataloader : Dataloader
    device : torch.device = torch.device("cpu") , optional

    Returns
    ----------
    None.

    Notes
    ----------
    Function for obtaining integrated gradients for a given tensor.
    """

    # Get random image, index and label #
    tensor, index, label = get_item(dataloader, device)

    # Create integrated gradient object #
    occlusion = Occlusion(model)

    attrib_image = occlusion.attribute(
        tensor,
        target=index,
        strides=(3, 8, 8),
        sliding_window_shapes=(3, 15, 15),
        baselines=0,
    )

    # Convert tensor to numpy array for plotting#
    image = tensor.to("cpu").numpy()[0]

    # Specify target label #
    print("Target label: " + label)

    plot_tensor_array(tensor - attrib_image, image, "Negative Occlusion")
    plot_tensor_array(tensor + attrib_image, image, "Positive Occlusion")



def get_outputs(
    model: any,
    dataloader: DataLoader,
    device: torch.device = torch.device("cuda"),
) -> None:
    """
    Parameters
    ----------
    model : any
    dataloader : Dataloader
    device : torch.device = torch.device("cpu") , optional

    Returns
    ----------
    None.

    Notes
    ----------
    Function for obtaining integrated gradients for a given tensor.
    """

    # Get random image, index and label #
    tensor, index, label = get_item(dataloader, device)

    # Use model to classify it #
    # Evaluation of Model #
    model.eval()
    with torch.no_grad():
        output = model(tensor)

    # Plot Image #
    image = tensor.to("cpu").numpy()[0]
    plot_image(image,label, "Original Image")