"""Generic functions for projects in google colab.

This file contains functions to handle xml files.

Covered functions:
    - read xml file
    - transform the xml file into a dictionary
"""

from typing import Dict, List, Any, Union
import xml.etree.ElementTree as ET


def xml_to_dict(
    element: ET.Element
) -> Dict[str, Any]:
    """ Recursively convert an ElementTree element and its children into a dictionary.
    *arguments*
    *element* The root Element of the XML tree.
    """
    element_dict = dict(element.attrib)

    for child in element:
        child_dict = xml_to_dict(child)
        # Check if the child has text content
        if child.text and child.text.strip():
            child_dict["_text"] = child.text.strip()
        if child.tag in element_dict:
            # Convert to list if the tag is already in the dictionary
            if not isinstance(element_dict[child.tag], list):
                element_dict[child.tag] = [element_dict[child.tag]]
            element_dict[child.tag].append(child_dict)
        else:
            element_dict[child.tag] = child_dict

    return element_dict


def read_xml_to_dict(
    file_path: str
) -> Dict[str, Any]:
    """ Read an XML file and convert its content into a dictionary.
    *args*
    *file_path* The path to the XML file.
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        return {root.tag: xml_to_dict(root)}
    except ET.ParseError as e:
        print(f"Error parsing XML file: {str(e)}")
        return None
