"""Generic functions for projects in google colab.

This file contains functions to pre/post-process data for OpenAI and files from OpenAI servers.

Covered:
    - transform text in dataframe:
        works if lines are separated by "\n" and cells by ","
    - count tokens from OpenAI models
"""

from typing import List, Dict, Any, Union
import pandas as pd
import json
import tiktoken


def text_to_dataframe(text: str, header: bool = False, index: bool = False) -> pd.DataFrame:
    """
    Transform text data into a Pandas DataFrame.
    *arguments*:
    *text* the input text data with lines separated by "\n" and entries within lines separated by ","
    """
    lines = text.strip().split("\n")
    data = [line.split(",") for line in lines]

    if index:
        index_column = []
        for line in data:
            index_column.append(line.pop(0))
    else:
        index_column = None

    if header:
        header_row = data.pop(0)
        if index:
            index_column.pop(0)
    else:
        header_row = None

    # Create a Pandas DataFrame from the data
    df = pd.DataFrame(data, index=index_column, columns=header_row)

    return df
    

def bytes_to_list(bytes_str):
    spl_string = bytes_str.decode('utf8').split('\n{"messages":')
    res = [json.loads(spl_string[0])]
    if len(spl_string)>1:
        for _el in spl_string[1:]:
            res.append(json.loads('{"messages":' + _el))
    return res


def num_tokens_from_messages(
    messages: Dict[str, str],
    model: str="gpt-3.5-turbo-0613"
) -> int:
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

