""" """
import os
import sys
from typing import Dict, Protocol, Union, Optional, Any, Tuple
from dataclasses import dataclass
import logging
import pathlib
import gdown
import requests
import zipfile


SplitPath = Tuple[pathlib.Path, Optional[pathlib.Path]]


file_paths = {
    "https://drive.google.com/file/d/1-kTUqYyhJFG8OaPj_wOcSmCA54F0pNn9/view??usp=share_link": 
        {
            "folder": "bin",
            "file": "bin.zip"
        },
    "https://drive.google.com/file/d/1ZHV9iQA_i0CwjIptIFtnypa51Vg0ymWj/view?usp=share_link":
        {
            "folder": "intermediate",
            "file": "intermediate.zip"
        },
    "https://drive.google.com/file/d/1cd-M6Z7y_otLqXDQCdMvezVgaCT386Ot/view?usp=share_link":
        {
            "folder": "outputs",
            "file": "outputs.zip"
        },
    "https://drive.google.com/file/d/1IRUxRSmqlaC7k1jq7zqOeMnPaCn3CL-l/view?usp=share_link":
        {
            "folder": ".",
            "file": "root.zip"
        },
    "https://drive.google.com/file/d/1aj0eRndJt0o9MQvSbopmJYE2qYeJpdWj/view?usp=share_link":
        {
            "folder": "saved_models",
            "file": "saved_models.zip"
        },
    "https://drive.google.com/file/d/1k5cZvEQMqEGfIZjmjehODirXJvGBGmH_/view?usp=share_link":
        {
            "folder": "moo_solver_CPAIOR/outputs/",
            "file": "moo_outputs.zip"
        },
}


global_formatter = logging.Formatter(
    '%(asctime)s : %(levelname)s : %(name)s : %(message)s')
global_filehandler = logging.FileHandler(
    os.path.join(".", 'download.log'),
    mode='w')
global_streamhandler = logging.StreamHandler(sys.stdout)


def create_logger(
        logger_name: str,
        formatter: logging.Formatter = global_formatter,
        handlers: Tuple[logging.Handler, ...] = (
            global_filehandler, global_streamhandler),
        logging_level: Optional[Union[str, int]] = None) -> logging.Logger:
    """Create and setup a logger using global settings

    Args:
        logger_name: Name of the logger, usually file name given in
            variable `__name__`

    Returns:
        initialized logging.Logger object
    """
    log = logging.getLogger(logger_name)
    # Get a global logging level
    log.setLevel(logging.DEBUG)
    # Set logging level to the value given in the argument
    if logging_level is not None:
        try:
            log.setLevel(logging_level)
        except (ValueError, TypeError):
            pass
    for handler in handlers:
        handler.setFormatter(formatter)
        log.addHandler(handler)
    return log
    
    
def split_path(input_path: Union[str, pathlib.Path]) -> SplitPath:
    """Split path into the parent directory tree and the file name."""
    if isinstance(input_path, str):
        input_path = pathlib.Path(input_path)
    if not input_path.suffix:
        # Provided input path is a path of directories (assuming each file 
        # needs an extension - not True in Unix but we make this assumption here
        return (input_path, None)
    return (input_path.parent, input_path.name)
    
    
def is_directory(input_path: pathlib.Path) -> bool:
    """Check if the path is a directory by checking if split_path returns file
    """
    return (split_path(input_path)[1] is None)


def create_directory_tree(path: pathlib.Path, verbose: bool = True) -> None:
    """For a given path, checks if the directory structure is present. If it is
    not, create the directory path from the first directory (counting from top)
    that is absent"""
    if verbose:
        logger.info("Creating folder structure in %s", path.as_posix())
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        if verbose:
            logger.info("Folder structure already exists.")


class URL_Downloader(Protocol):
    """ """
    def __call__(self, url: str, output_path: Union[pathlib.Path, str]) -> None:
        """Download data from url to output destination (path) """


@dataclass
class GDriveFileDownloader:
    """ """
    quiet: bool = False
    fuzzy: bool = True
    resume: bool = False

    def __call__(self, url: str, output_path: Union[pathlib.Path, str]) -> None:
        """Download file from Google Drive using a URL to output destination 
        (path) """
        if isinstance(output_path, pathlib.Path):
            output_path = output_path.as_posix()
        if is_directory(output_path):
            logger.info(
                "Downloader expects the path to a file, not to a directory.")
            return
        dir_tree, _ = split_path(output_path)
        create_directory_tree(dir_tree, verbose=False)
        gdown.download(
            url, output=output_path, quiet=self.quiet, fuzzy=self.fuzzy,
            resume=self.resume)


@dataclass
class GDriveCachedFileDownloader:
    """ """
    quiet: bool = False
    md5: Optional[Any] = None
    extract: bool = False

    def __call__(self, url: str, output_path: Union[pathlib.Path, str]) -> None:
        """Download cached file from Google Drive using a URL to output 
        destination (path) """
        if isinstance(output_path, pathlib.Path):
            output_path = output_path.as_posix()
        if self.extract:
            postprocess = gdown.extractall
        else:
            postprocess = None
        if is_directory(output_path):
            logger.info(
                "Downloader expects path pointing to a file not to a directory.")
            return
        gdown.cached_download(
            url=url, path=output_path, md5=self.md5, quiet=self.quiet,
            postprocess=postprocess)


def download_from_url(
        url: str, output_path: pathlib.Path, downloader: URL_Downloader, 
        update: bool = True, relative_path: bool = False,
        checksum: Optional[Any] = None,
        verbose: bool = False,
        post_checksum_check: bool = False) -> None:
    """Download data from a URL
    Args:
        url: url pointing to date, e.g. share link from Google Drive
        output: directory/file relative to package root directory
        update: updates old data when checksum does not match
        relative_path: if True the path provided will be relative to the package
            root folder
        verbose: print more detailed output
        checksum: If given validate the file against the md5 sum
        post_checksum_check: Check the checksum once again, if given
    """
    # Create a separate function logger
    local_logger = create_logger(logger_name="Download_URL")
    if verbose:
        local_logger.setLevel(logging.DEBUG)
    else:
        local_logger.setLevel(logging.INFO)
    if relative_path:
        output_path = get_package_file(output_path)
    else:
        output_path = pathlib.Path(output_path)
    is_destination_dir: bool = is_directory(output_path)
    destination_exists: bool = pathlib.Path.exists(output_path)
    local_logger.info("Downloading from url %s", url)
    if destination_exists and not is_destination_dir:
        if checksum is not None:
            if file_valid(output_path, checksum):
                local_logger.debug(
                    "Data from url %s already exists and file is valid", url)
                return
            if not update:
                local_logger.debug(
                    "Data from url %s alread exists and is outdated", url)
                local_logger.debug(
                    "To ovewrite, run the function with output flag set to true")
                return
            local_logger.debug(
                "Data from url %s alread exists but file is outdated", url)
            local_logger.debug("Updating the file with the new version")
            downloader(url, output_path)    
        else:
            if not update:
                local_logger.debug(
                    "Data from url %s alread exists", url)
                local_logger.debug(
                    "To overwrite, run the function with output flag set to true")
                return                  
            local_logger.debug(
                "Data from url %s alread exists, ovewriting", url)
            downloader(url, output_path)
    else:
        local_logger.info("Downloading from url %s", url)
        downloader(url, output_path)

    if post_checksum_check and checksum and not is_destination_dir:
        # Check the checksum of the downloaded file (does not work for folders)
        if file_valid(output_path, checksum):
            local_logger.info("File matches checksum, OK.")
        else:
            local_logger.warning("Downloaded file does not match the checksum.")


def download_and_unzip_file(url: str, output_folder: str) -> None:
    """ """
    # Extract the file ID from the Google Drive URL
    file_id = url.split('/')[-2]
    
    # Construct the download URL
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    # Download the file
    response = requests.get(download_url, stream=True)
    zip_file_path = os.path.join(output_folder, "downloaded_file.zip")
    
    with open(zip_file_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=128):
            file.write(chunk)
    
    # Unzip the file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(output_folder)
    
    # Delete the original zip file
    os.remove(zip_file_path)
    print(f"File(s) downloaded, unzipped to {output_folder}, and original zip file deleted.")


def prompt_user_confirmation():
    response = input(
        "Warning, the download script will overwrite existing files. Do you want to continue (yes/no): ").strip().lower()
    while response not in {"yes", "no"}:
        response = input("Please enter 'yes' or 'no': ").strip().lower()
    return response == "yes"


if __name__ == "__main__":
    #logger = create_logger(logger_name=__name__)
    if not prompt_user_confirmation():
        print("Download aborted by user.")
        sys.exit(0)
    for ix, (url, folder_dict) in enumerate(file_paths.items()):
        if ix == 1:
            break
        folder = folder_dict['folder']
        filename = folder_dict['file']
        filepath = os.path.join(folder,filename)
        download_from_url(
            url, filepath, downloader=GDriveFileDownloader(), 
            relative_path=False, verbose=True)
    
        # Unzip the file
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(".")
        
        # Delete the original zip file
        os.remove(filepath)
        print(f"File(s) downloaded, unzipped to {folder}, and original zip file deleted.")
