"""Generic functions for projects in google colab.

This file contains functions to work with LLMs.
"""

from typing import Dict, List, Any, Union
from copy import deepcopy


def complete_template(
    template: str,
    input_variables: Dict[str, str]
) -> str:
    temp_c = deepcopy(template)
    for key, val in input_variables.items():
        temp_c = temp_c.replace(
            "{" + f"{key}" + "}",
            val
        )
    return temp_c
