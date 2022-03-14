"""XMM-Newton Science Archive (XSA) module.

http://nxsa.esac.esa.int/nxsa-web/#home
"""

from io import FileIO
import tempfile
from typing import Dict, List
import requests
import src.utils as utils
import tarfile

class XsaDownloader:
    """XMM-Newton Science Archive downloader interface.
    """
    
    def download(self, output_dir, observation_id: str, filters: List[str]) -> Dict[str, List[str]]:
        """Downloads images of the specified observation and filters into a target output directory.
        Files will be placed follwoing a {output_dir/filter/file} pattern.

        Args:
            output_dir (_type_): output directory where files will be placed
            observation_id (str): observation identifier
            filters (List[str]): list of filters to search for

        Returns:
            Dict[str, List[str]]: a key-value structure containing filter as keys and the list of corresponding
            downloaded file paths as a list. Filters with no files will not be included in the results (no key).
        """
        pass

class XsaHttpDownloader(XsaDownloader):
    """XMM-Newton Science Archive (XSA) downloader via HTTP get request.
    
    http://nxsa.esac.esa.int/nxsa-web/#aio
    """
    
    def __init__(self, base_url: str, regex_patern: str) -> None:
        """Constructor.

        Args:
            base_url (str): XMM-Newton Science Archive (XSA) base URL
            regex_patern (str): regular expression used to matched amongst downloaded files
        """
        super().__init__()
        self.base_url = base_url
        self.regex_pattern = regex_patern
    
    def _build_query_string(self, observation_id: str, filter: str) -> Dict[str, str]:
        return {'obsno': observation_id,    
                'instname': 'OM',
                'level': 'PPS',
                'extension': 'FTZ',
                'filter': filter}
        
    def _download_single_filter_as_tar(self, file: FileIO, observation_id: str, filter: str):      
        response = requests.get(self.base_url, params=self._build_query_string(observation_id,
                                                                               filter))
        file.write(response.content)
        
    def download(self, output_dir, observation_id: str, filters: List[str]) -> Dict[str, List[str]]:
        """See base class."""
        results = {}
        
        for filter in filters:
            extracted_list = []
            
            with tempfile.TemporaryFile(mode='w+b') as temp_file:
                self._download_single_filter_as_tar(file=temp_file,
                                                    observation_id=observation_id,
                                                    filter=filter)
                
                # Point back at the begining of the file before reading it
                temp_file.seek(0)
                
                with tarfile.open(fileobj=temp_file, mode='r') as tar_file:
                    filter_dir = utils.build_path(part1=output_dir,
                                                  part2=filter)
                    
                    for member in utils.find_members_in_tar(tar=tar_file,
                                                            regex_pattern=self.regex_pattern):
                        
                        extracted_file_path = utils.extract_tar_member_to_dir(tar=tar_file,
                                                                            member=member,
                                                                            output_dir=filter_dir)
                        extracted_list.append(extracted_file_path)
                    
                if len(extracted_list) > 0:
                    results[filter] = extracted_list
                    
        return results

    