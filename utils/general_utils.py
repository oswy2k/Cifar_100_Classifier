# External Libraries Imports #
import numpy as np
import torch

# System Imports #
import random
import time
import os


# Gets main processing device for training #
def get_available_devices() -> torch.device:
    """
    Parameters
    ----------
    None.

    Returns
    ----------
    device: torch.device

    Notes
    ----------
    Get available device for training. If GPU is available, it will be used, else CPU will be used.
    """
    if torch.cuda.is_available():
        # setting device on GPU if available, else CPU
        device = torch.device("cuda")
        print(torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


# Sets seed for all randomizers used in the code #
def set_all_seeds(seed: int) -> None:
    """
    Parameters
    ----------
    seed : int

    Returns
    ----------
    None

    Notes
    ----------
    Set all seeds for randomizers used in the code.
    """

    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
