"""Observations-related module."""


from typing import Iterator
import csv


class Observation:
    """Represents an scientific observation."""
    
    def __init__(self, id: str, object: str,
                 ra1: float, dec1: float,
                 ra2: float, dec2: float):
        """Constructor.

        Args:
            id (str): identifier
            object (str): name of the potentially observed object (not the target)
            ra1 (float): right ascension at the start of the observation [deg]
            dec1 (float): declination at the start of the observation [deg]
            ra2 (float): right ascension at the end of the observation [deg]
            dec2 (float): declination at the end of the observation [deg]
        """
        self.id     = id
        self.object = object
        self.ra1    = ra1
        self.dec1   = dec1
        self.ra2    = ra2
        self.dec2   = dec2
        pass
    

class Repository:
    """Observation repository interface."""
    
    def get_iter(self) -> Iterator[Observation]:
        """Returns the observations iterator."""     
        pass
    

class CsvRepository(Repository):
    """Observation repository using a Comma Separated Value (CSV) file.
    The CSV structure must be as follows: 
    
    {id,object,ra1,dec1,ra2,dec2}
    
    Where:
    - id: observation identifier
    - object: name of the potentially observed object
    - ra1: right ascension at the start of the observation [deg]
    - dec1: declination at the start of the observation [deg]
    - ra2: right ascension at the end of the observation [deg]
    - dec2: declination at the end of the observation [deg]
    """
    
    def __init__(self, csv_path: str, ignore_top_n_lines: int = 0):
        """Constructor.

        Args:
            csv_path (str): Comma Separated Value (CSV) file path
            ignore_top_n_lines (int, optional): number of lins to be ignored
            (e.g. 1 for headers). Defaults to 0.
        """
        super().__init__()
        observations = []
        
        with open(file=csv_path, mode='r') as csv_file:
            reader = csv.reader(csv_file)
            current_line = 1
            
            for line in reader:
                if current_line > ignore_top_n_lines:
                    observation = Observation(id        =line[0],
                                              object    =line[1],
                                              ra1       =float(line[2]),
                                              dec1      =float(line[3]),
                                              ra2       =float(line[4]),
                                              dec2      =float(line[5]))
                    observations.append(observation)
                    
                current_line += 1
                
        self.iter = iter(observations)
        
    def get_iter(self) -> Iterator[Observation]:
        """See base class."""     
        return self.iter
        