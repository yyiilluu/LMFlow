import torch

def get_device():
    """
    Get local device
    Returns: device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_max_memory():
    free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024 ** 3)
    max_memory = f"{free_in_GB - 2}GB"
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory