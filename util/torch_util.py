import torch

def get_device():
    """
    Get local device
    Returns: device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
