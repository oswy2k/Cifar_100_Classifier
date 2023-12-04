# External Libraries Imports #
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as datasets
import torchvision.transforms as T

import numpy as np

# Main dataset Configuration #
DATA_PATH = "./Cifar100_Data"
NUM_TRAIN = 50000
NUM_VAL = 5000
NUM_TEST = 5000


def cifar100_dataset_statistics() -> [list[float],list[float]]:
    '''
    Parameters
    ----------
    None.

    Returns
    ----------
    [list[float],list[float]]:

    Notes
    ----------
    Calculates the mean and standard deviation of the cifar100 dataset.
    '''
    # Get mean #
    train_data = datasets.CIFAR100('./', train=True, download=True)

    # Concatenate all images into one numpy array
    x = np.concatenate([np.asarray(train_data[i][0]) for i in range(len(train_data))])

    # Calculate the mean and standard deviation along the axes
    mean = np.mean(x, axis=(0, 1))/255
    std = np.std(x, axis=(0, 1))/255

    # Set the mean and std to lists
    mean=mean.tolist()
    std=std.tolist()

    return [mean,std]

def cifar100_dataset_Generator(
    transforms: T.Compose,
    minibatch_size: int = 32,
    download: bool = True,
    log: bool = True,
) -> [DataLoader, DataLoader, DataLoader]:
    """
    Parameters
    ----------
    transforms : T.Compose
    minibatch_size : int = 32, optional
    download : bool = True, optional
    log : bool = True, optional
    Returns
    ----------
    [DataLoader, DataLoader, DataLoader]

    Notes
    ----------
    Downloads or unpacks Cifar10 dataset and makes the different dataloaders based on
    given transforms.
    """

    # Download Train dataset
    train_data_cifar100 = datasets.CIFAR100(
        DATA_PATH, train=True, download=download, transform=transforms
    )
    # Download Validation set
    validation_data_cifar100 = datasets.CIFAR100(
        DATA_PATH, train=False, download=download, transform=transforms
    )
    # Download Test set
    test_data_cifar100 = datasets.CIFAR100(
        DATA_PATH, train=False, download=download, transform=transforms
    )

    # Create Dataloader for Training
    train_dataLoader = DataLoader(
        train_data_cifar100,
        batch_size=minibatch_size,
        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)),
    )

    # Create Dataloader for Validation
    validation_dataLoader = DataLoader(
        validation_data_cifar100,
        batch_size=minibatch_size,
        sampler=sampler.SubsetRandomSampler(range(NUM_VAL)),
    )

    # Create Dataloader for Testing
    test_dataLoader = DataLoader(
        test_data_cifar100,
        batch_size=minibatch_size,
        sampler=sampler.SubsetRandomSampler(range(NUM_VAL, len(test_data_cifar100))),
    )

    if log:
        verify_data(
            train_data_cifar100, "\n", validation_data_cifar100, "\n", test_data_cifar100
        )

    return [train_dataLoader, validation_dataLoader, test_dataLoader]


def verify_data(*argv) -> None:
    """
    Parameters
    ----------
    argv : any

    Returns
    ----------
    None

    Notes
    ----------
    Takes all values given and prints them to the console.
    """

    for arguments in argv:
        print(arguments)


# Get random image and label from dataloader #
def get_item(dataloader: DataLoader, device: torch.device) -> [torch.Tensor, int, str]:
    """
    Parameters
    ----------
    dataloader : DataLoader

    Returns
    ----------
    [torch.Tensor, int, str]

    Notes
    ----------
    Get a random image, label and index from the dataloader given.
    """

    # Get random index #
    random_Index = np.random.randint(len(dataloader))

    # Get random image and label from dataloader #
    data_Classes = dataloader.dataset.classes

    # Get Random image and Label #
    label = data_Classes[dataloader.dataset[random_Index][1]]
    index = data_Classes.index(data_Classes[dataloader.dataset[random_Index][1]])
    image = torch.tensor(np.array([dataloader.dataset[random_Index][0]])).to(device)

    return [image, index, label]
