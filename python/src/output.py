"""Analysis output module."""

from typing import List
import src.observation as observation
import src.input as input
import os


class Recorder:
    """Analysis output recorder interface."""

    def prepare_observation_record(self, observation: observation.Observation):
        """Prepares the next observation record.

        Args:
            observation (observation.Observation): the observation instance
        """
        pass
    
    def record_filter_input(self, filter: str, input: input.Input):
        """Records the specified input for a specified filter.

        Args:
            filter (str): the filter (e.g. 'U')
            input (input.Input): the analysis input
        """
        pass
    
    def record_observation(self, filters: List[str]):
        """Records the current observation.

        Args:
            filters (List[str]): list of filters that have been considered.
        """
        pass
    

class CsvRecorder(Recorder):
    """Comma Separated Value (CSV) implementation of the recorder interface.
    This implementation will generate a (or append to an existing) CSV file, 
    with records following this schema: 
    
    {observation,object,filter_1,filter_2,...,filter_n}
    
    Where:
    - observation: observation identifier
    - object: name of the potentially observed object
    - filter_i: will contain the analysis input:
        - 'Y': for detections
        - 'N': for no detections
        - 'D': for dubious detections
        - '': am empty string for filters with no analysis input
        - 'ERROR': if any other input was given
    """
    
    def __init__(self, csv_path: str, include_headers: bool = True):
        """Constructor.

        Args:
            csv_path (str): Comma Separated Value (CSV) file path
            include_headers (bool, optional): flag indicating whether
            headers are to be included in the output file. Defaults to True.
        """
        super().__init__()
        self.csv_path               = csv_path
        self.include_headers        = include_headers
        self.current_observation    = None
        self.current_record         = None
    
    def prepare_observation_record(self, observation: observation.Observation):
        """See base class."""
        self.current_observation = observation
        self.current_record = {}
        
    def record_filter_input(self, filter: str, input: input.Input):
        """See base class."""
        self.current_record[filter] = input
    
    def _input_to_str(self, input_value: input.Input) -> str:
        if (input_value == input.Input.DETECTED):
            return 'Y'
        if (input_value == input.Input.NOT_DETECTED):
            return 'N'
        if (input_value == input.Input.DUBIOUS):
            return 'D'
        return 'ERROR'
    
    def _create_csv_row(self, filters: List[str]) -> str:
        cols = [self.current_observation.id,
                self.current_observation.object]
        
        for filter in filters:   
            if filter in self.current_record:
                cols.append(self._input_to_str(self.current_record[filter]))
            else:
                cols.append('')
                        
        return ','.join(cols)
        
    def record_observation(self, filters: List[str]):
        """See base class."""
        with open(self.csv_path, mode='a') as file:
            if (self.include_headers):
                headers = ['observation',
                            'object']
                
                for filter in filters:
                    headers.append(filter)
                    
                file.write(','.join(headers) + os.linesep)
                self.include_headers = False
                
            file.write(self._create_csv_row(filters=filters) + os.linesep)
            