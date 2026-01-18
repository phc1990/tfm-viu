"""XMM-Newton Science Archive (XSA) module.

http://nxsa.esac.esa.int/nxsa-web/#home
"""

from io import FileIO
import tempfile
from typing import Dict, List
import requests
import tarfile
import subprocess
import src.screening.utils as utils
import os

def convert_filter_name_to_xsa_name(filter: str) -> str:
    if filter == 'S':
        return 'UVW2'
    elif filter == 'M':
        return 'UVM2'
    elif filter == 'L':
        return 'UVW1'
    elif filter == 'U':
        return 'U'
    elif filter == 'B':
        return 'B'
    elif filter == 'V':
        return 'V'
    else:
        raise Exception('Filter "' + filter + '" not recognised.')


class Crawler:
    """XMM-Newton Science Archive (XSA) crawler interface.
    """
    
    def crawl(
        self,
        observation_id: str,
        filters: list[str],
    ) -> Dict[str, List[str]]:
        """Obtains the images of the specified observation and filters.

        Args:
            observation_id (str): the ID of the observation to be obtained
            filters (list[str]): the filters of interest

        Returns:
            Dict[str, List[str]]: a key-value structure containing filter as keys and the list of corresponding
            file paths as a list. Filters with no files will not be included in the results (no key).
        """
        pass


class HttpCrawler(Crawler):
    """XMM-Newton Science Archive (XSA) crawler via HTTP GET request. 
    Downloaded files will be placed following a {download_directory/filter/file} pattern.
    
    http://nxsa.esac.esa.int/nxsa-web/#aio
    """
    
    def __init__(self, download_directory, base_url: str, regex_pattern: str):
        """Constructor.

        Args:
            download_directory (_type_): directory where files will be downloaded
            base_url (str): XMM-Newton Science Archive (XSA) base URL
            regex_pattern (str): regular expression used to matched against downloaded files
        """
        super().__init__()
        self.download_directory = download_directory
        self.base_url = base_url
        self.regex_pattern = regex_pattern
    
    def _build_query_params(self, observation_id: str, filter: str) -> Dict[str, str]:
        """Build the query parameters.

        Args:
            observation_id (str): observation identifier.
            filter (str): filter value (S,M,L,U,B,V which will be mapped according to 
            https://xmm-tools.cosmos.esa.int/external/xmm_user_support/documentation/uhb/omfilters.html)

        Raises:
            Exception: if the filter is not recognised

        Returns:
            Dict[str, str]: a list of query parameters key-value pairs.
        """
        return {'obsno': observation_id,    
                'instname': 'OM',
                'level': 'PPS',
                'extension': 'FTZ',
                'filter': convert_filter_name_to_xsa_name(filter=filter)}
        
    def _download_single_filter_as_tar(self, file: FileIO, observation_id: str, filter: str):      
        """Downloads data for the specified filter and observation to the specified file.

        Args:
            file (FileIO): file where data will be downloaded to
            observation_id (str): observation identifier
            filter (str): filter
        """
        pass
        
    def crawl(
        self,    
        observation_id: str,
        filters: list[str],
    ) -> Dict[str, List[str]]:
        """See base class."""
        results = {}
        observation_dir = utils.build_path(part1=self.download_directory,
                                           part2=observation_id)
        
        for filter in filters:
            extracted_list = []
            
            with tempfile.NamedTemporaryFile(mode='w+b') as temp_file:
                self._download_single_filter_as_tar(file=temp_file,
                                                    observation_id=observation_id,
                                                    filter=filter)

                # 1) comprobar tamaño
                temp_file.flush()
                temp_file.seek(0, os.SEEK_END)
                size = temp_file.tell()
                if size == 0:
                    print(f"[XSA] WARN: empty download (obs={observation.id} filter={filter}); skipping.")
                    continue

                # 2) intentar abrir como tar, y si falla, log de diagnóstico               
                # Point back at the begining of the file before reading it
                temp_file.seek(0)
                try:
                        
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

                except tarfile.ReadError:
                    temp_file.seek(0)
                    head = temp_file.read(300)
                    print(f"[XSA] WARN: payload is not a tar archive (obs={observation.id} filter={filter}). "
                        f"First bytes: {head!r}")
                    continue    
                                    
        return results
    

class HttpPythonRequestsCrawler(HttpCrawler):
    """HTTPCrawler implementation using Python's 'requests' module.

    https://docs.python-requests.org/en/latest/
    """
    
    def __init__(self, download_directory, base_url: str, regex_pattern: str):
        """See base class."""
        super().__init__(download_directory, base_url, regex_pattern)
    
    # def _download_single_filter_as_tar(self, file: FileIO, observation_id: str, filter: str):
    #     """See base class."""      
    #     response = requests.get(self.base_url, params=self._build_query_params(observation_id,
    #                                                                            filter))
    #     file.write(response.content)


def _download_single_filter_as_tar(self, file, observation_id: str, filter: str):
    url = self.base_url + self._build_query_string(observation_id=observation_id, filter=filter)
    args = ["curl", "-L", "-f", "-sS", "--retry", "3", "--retry-delay", "2", "-o", file.name, url]
    res = subprocess.run(args, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"curl failed ({res.returncode}): {res.stderr.strip()}")
        

class HttpCurlCrawler(HttpCrawler):
    """HTTPCrawler implementation using curl via command line.

    https://curl.se/docs/manpage.html
    """
    
    def __init__(self, download_directory, base_url: str, regex_pattern: str):
        """See base class."""
        super().__init__(download_directory, base_url, regex_pattern)
        
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
        