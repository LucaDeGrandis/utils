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


def unzip_file(file_path, destination_folder):
    """
    Extracts all files from a zip archive to a specified folder.

    Args:
        file_path (str): The path to the zip archive.
        destination_folder (str): The path to the folder where the files will be extracted.

    Returns:
        None
    """
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(destination_folder)
