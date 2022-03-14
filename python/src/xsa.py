from io import FileIO
from typing import Dict, List
import src.utils as utils
import requests

class XsaDownloader:
    
    def download(self, dir, observation_id: str, filters: List[str]) -> Dict[str, FileIO]:
        pass

class XsaHttpDownloader(XsaDownloader):
    
    def __init__(self, baseUrl: str) -> None:
        super().__init__()
        self.baseUrl = baseUrl
    
    def _build_query_string(self, observation_id: str, filter: str) -> Dict[str, str]:
        
        return {'obsno': observation_id,    
                'instname': 'OM',
                'level': 'PPS',
                'extension': 'FTZ',
                'filter': filter}
        
    def _download_single_filter(self, filepath: str, observation_id: str, filter: str):
        response = requests.get(self.baseUrl, params=self._build_query_string(observation_id,
                                                                              filter))
        with open(filepath,'wb') as file:
            file.write(response.content)
        
    def download(self, dir, observation_id: str, filters: List[str]) -> Dict[str, FileIO]:
        
        pass

    