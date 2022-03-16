"""XMM-Newton Science Archive (XSA) module.

http://nxsa.esac.esa.int/nxsa-web/#home
"""

from io import FileIO
import tempfile
from typing import Dict, List
import requests
import tarfile
import src.utils as utils
import subprocess


class Crawler:
    """XMM-Newton Science Archive (XSA) crawler interface.
    """
    
    def crawl(self, observation_id: str, filters: List[str]) -> Dict[str, List[str]]:
        """Obtains the images of the specified observation and filters.

        Args:
            output_dir (_type_): output directory where files will be placed
            observation_id (str): observation identifier
            filters (List[str]): list of filters to search for

        Returns:
            Dict[str, List[str]]: a key-value structure containing filter as keys and the list of corresponding
            file paths as a list. Filters with no files will not be included in the results (no key).
        """
        pass


class HttpCrawler(Crawler):
    """XMM-Newton Science Archive (XSA) crawler via HTTP GET request. 
    Downloaded files will be placed following a {download_dir/filter/file} pattern.
    
    http://nxsa.esac.esa.int/nxsa-web/#aio
    """
    
    def __init__(self, download_dir, base_url: str, regex_patern: str):
        """Constructor.

        Args:
            download_dir (_type_): directory where files will be downloaded
            base_url (str): XMM-Newton Science Archive (XSA) base URL
            regex_patern (str): regular expression used to matched against downloaded files
        """
        super().__init__()
        self.download_dir = download_dir
        self.base_url = base_url
        self.regex_pattern = regex_patern
    
    def _build_query_params(self, observation_id: str, filter: str) -> Dict[str, str]:
        return {'obsno': observation_id,    
                'instname': 'OM',
                'level': 'PPS',
                'extension': 'FTZ',
                'filter': filter}
        
    def _download_single_filter_as_tar(self, file: FileIO, observation_id: str, filter: str):      
        """Downloads data for the specified filter and observation to the specified file.

        Args:
            file (FileIO): file where data will be downloaded to
            observation_id (str): observation identifier
            filter (str): filter
        """
        pass
        
    def crawl(self, observation_id: str, filters: List[str]) -> Dict[str, List[str]]:
        """See base class."""
        results = {}
        observation_dir = utils.build_path(part1=self.download_dir,
                                           part2=observation_id)
        
        for filter in filters:
            extracted_list = []
            
            with tempfile.NamedTemporaryFile(mode='w+b') as temp_file:
                self._download_single_filter_as_tar(file=temp_file,
                                                    observation_id=observation_id,
                                                    filter=filter)
                
                # Point back at the begining of the file before reading it
                temp_file.seek(0)
                
                with tarfile.open(fileobj=temp_file, mode='r') as tar_file:
                    filter_dir = utils.build_path(part1=observation_dir,
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
    

class HttpPythonRequestsCrawler(HttpCrawler):
    """HTTPCrawler implementation using Python's 'requests' module.

    https://docs.python-requests.org/en/latest/
    """
    
    def __init__(self, download_dir, base_url: str, regex_patern: str):
        """See base class."""
        super().__init__(download_dir, base_url, regex_patern)
    
    def _download_single_filter_as_tar(self, file: FileIO, observation_id: str, filter: str):
        """See base class."""      
        response = requests.get(self.base_url, params=self._build_query_params(observation_id,
                                                                               filter))
        file.write(response.content)
        

class HttpCurlCrawler(HttpCrawler):
    """HTTPCrawler implementation using curl via command line.

    https://curl.se/docs/manpage.html
    """
    
    def __init__(self, download_dir, base_url: str, regex_patern: str):
        """See base class."""
        super().__init__(download_dir, base_url, regex_patern)
        
    def _build_query_string(self, observation_id: str, filter: str) -> str:
        params = []
        for key, value in self._build_query_params(observation_id=observation_id, filter=filter).items():
            params.append('='.join([key,value]))
        
        if len(params) > 0:
            return '?' + '&'.join(params)
        return ''
    
    def _download_single_filter_as_tar(self, file: FileIO, observation_id: str, filter: str):
        """See base class."""
        args = ['curl', '-o', file.name, self.base_url + self._build_query_string(observation_id=observation_id, filter=filter)]      
        subprocess.Popen(args=args).wait()