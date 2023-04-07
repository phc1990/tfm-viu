"""Utility module."""

from typing import List
import os
import tarfile
import re


def build_path(part1: str, part2: str, extension: str = None) -> str:
    """Builds a filesystem path.

    Args:
        part1 (str): first part of the path (e.g. base directory)
        part2 (str): second part of the path (e.g. file name or directory)
        extension (str, optional): file extension. Defaults to None.

    Returns:
        str: the file system path
    """
    path = os.path.join(part1, part2)
    
    if extension:
        path += '.' + extension
    
    return path


def find_members_in_tar(tar: tarfile.TarFile, regex_pattern: str) -> List[tarfile.TarInfo]:
    """Explores the given tar file searchinf for members matching the specified regular expression.
    This function does not open nor close the tar file.

    Args:
        tar (tarfile.TarFile): the tarfile to explore
        regex_pattern (str): the regular expression pattern to be used to match the tar members

    Returns:
        List[tarfile.TarInfo]: the list of matches
    """
    matches = []

    for member in tar:
        
        if re.match(pattern=regex_pattern, string=member.name):
            matches.append(member)

    return matches


def extract_tar_member_to_dir(tar: tarfile.TarFile, member: tarfile.TarInfo, output_dir: str) -> str:
    """Extracts a tar member onto a target output directory, preserving its base name.
    This function does not open nor close the tar file.

    Args:
        tar (tarfile.TarFile): the tar file containing the member
        member (tarfile.TarInfo): the member to extract
        output_dir (str): the target directory where the file will be placed

    Returns:
        str: the full path of the extracted file
    """
    # Modify the member's name so that extraction does not replicate nested folders
    original_name = member.name
    base_name = os.path.basename(member.name)
    member.name = base_name
    
    tar.extract(member, output_dir)
    
    # Restore the property
    member.name = original_name
    
    return build_path(part1=output_dir,
                      part2=base_name)
            