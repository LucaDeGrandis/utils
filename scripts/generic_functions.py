"""Generic functions for projects in google colab.

This file contains functions to load/write files in various formats. There is also a function to create directories.

Covered formats:
    - json
    - jsonl
    - txt
"""

from typing import List, Dict, Any, Union
import matplotlib.pyplot as plt
from pathlib import Path
import json
import os



def load_json_file(
	filepath: str
) -> List[Any]:
    """Load a json into a list
    *arguments*
    *filepath* path to the file
    """
    with open(filepath, 'r', encoding='utf8') as reader:
        json_data = json.load(reader)
    return json_data


def write_json_file(
    filepath: str,
    input_dict: Dict[str, Any],
    overwrite: bool =False
) -> None:
    """Write a dictionary into a json
    *arguments*
    *filepath* path to save the file into
    *input_dict* dictionary to be saved in the json file
    *overwrite* whether to force overwriting a file.
        Default is false so you don't delete an existing file.
    """
    if not overwrite:
        assert not os.path.exists(filepath)
    with open(filepath, 'w', encoding='utf8') as writer:
        json.dump(input_dict, writer, indent=4, ensure_ascii=False)


def load_jsonl_file(
	filepath: str
) -> List[Dict[str, Any]]:
    """Load a json into a list
    *arguments*
    *filepath* path to the file
    """
    data = []
    with open(filepath, "r", encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line.strip()))
    return data


def load_jsonl_to_generator(
	filepath: str
) -> List[Dict[str, Any]]:
    """Load a json into a list
    *arguments*
    *filepath* path to the file
    """
    with open(filepath, "r", encoding='utf8') as f:
        lines = f.readlines()
    for line in lines:
        yield json.loads(line.strip())


def write_jsonl_file(
    filepath: str,
    input_list: List[Any],
    mode: str ='a+',
    overwrite: bool =False
) -> None:
    """Write a list into a jsonl
    *arguments*
    *filepath* path to save the file into
    *input_list* list to be saved in the json file, must be made of json iterable objects
    *overwrite* whether to force overwriting a file.
        When set to False you will append the new items to an existing jsonl file (if the file already exists).
    """
    if overwrite:
        try:
            os.remove(filepath)
        except:
            pass
    with open(filepath, mode, encoding='utf8') as writer:
        for line in input_list:
            writer.write(json.dumps(line) + '\n')


def load_txt_file(
	filepath: str,
	join: bool = False,
) -> List[str]:
    """Load a json into a list
    *arguments*
    *filepath* path to the file
    """
    data = []
    with open(filepath, 'r', encoding='utf8') as reader:
        lines = reader.readlines()
        for line in lines:
            data.append(line)
    if join:
        data = "".join(data)
    return data


def write_txt_file(
    filepath: str,
    input_list: List[str],
    mode: str ='a+',
    overwrite: bool =False
) -> None:
    """Write a list into a txt
    *arguments*
    *filepath* path to save the file into
    *input_list* list to be saved in the json file, must be made of strings
    *overwrite* whether to force overwriting a file.
        When set to False you will append the new items to an existing jsonl file (if the file already exists).
    """
    if overwrite:
        try:
            os.remove(filepath)
        except:
            pass
    with open(filepath, mode, encoding='utf8') as writer:
        for line in input_list:
            writer.write(line + '\n')


def create_line_plot(
    data: List[Union[float, int]],
    title: str ='',
    xlabel: str ='',
    ylabel: str='',
    legend_label: str ='',
    color: str ='blue'
) -> None:
    """ Create a line plot from a list of numbers.
    *arguments*
    *data* the data points to be plotted
    *title* the title for the plot
    *xlabel* the label of the x axis
    *ylabel* the label of the y axis
    *legend_label* the label of the legend
    *color* the line color
    """
    plt.figure(figsize=(10, 6))
    plt.plot(data, label=legend_label, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def make_dir(
  path: str
) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
