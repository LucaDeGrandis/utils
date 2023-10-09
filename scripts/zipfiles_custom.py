import zipfile


def zip_files(file_paths, zip_path):
    """Create a zip archive of the specified files.

    Args:
        file_paths (list): A list of file paths to include in the zip archive.
        zip_path (str): The path to the output zip archive.

    Returns:
        None
    """
    with zipfile.ZipFile(zip_path, 'w') as zip_file:
        for file_path in file_paths:
            zip_file.write(file_path)
