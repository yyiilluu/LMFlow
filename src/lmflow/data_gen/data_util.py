import json
from typing import List, Dict, Any



def read_jsonl_file(text_file_path: str) -> List[str]:
    """

    Args:
        text_file_path (str): path to file

    Returns:
        List[str]: list of read lines from `text_file_path`
    """
    with open(text_file_path, "r") as f:
        data = f.readlines()
    return data


def from_jsonl(data_file_path: str):
    """class method to generate data directly from a jsonl file

    Args:
        data_file_paths (list of str): A list of dataset file paths
        transforms (Callable): a method for transforming X i.e. input features.
            used by children classes.

    Returns:
        list (data examples as list)
    """
    data = read_jsonl_file(data_file_path)
    data = list(map(json.loads, data))
    return data


def write_to_jsonl(fname: str, data: List[Dict[str, Any]]) -> None:
    """
    Write a list of dictionaries to jsonl file
    """
    with open(fname, "w") as outfile:
        for d in data:
            json.dump(d, outfile)
            outfile.write("\n")


def write_json(fname: str, data: Dict[str, Any]) -> None:
    with open(fname, "w") as outfile:
        json.dump(data, outfile)