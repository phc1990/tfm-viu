"""Flexible Image Transport System (FITS) module.

https://fits.gsfc.nasa.gov/
"""


import tempfile
from typing import List
import subprocess
import src.screening.observation as observation
import os
    

class Interface:
    """Flexible Image Transport System (FITS) user interface (UI)."""
    
    def display(self, fits_paths: List[str], observation: observation.Observation = None):
        """Displays the specified FITS corresponding to the specified observation.

        Args:
            fits_paths (List[str]): list of FITS paths
            observation (observation.Observation): the corresponding observation. 
            Defaults to None.
        """
        pass
    
    def close_current_display(self):
        """Closes the current display."""
        pass
    

class Ds9Interface(Interface):
    """Flexible Image Transport System (FITS) user interface (UI) using SAO Image DS9 ('DS9'). 

    https://sites.google.com/cfa.harvard.edu/saoimageds9/home
    """
    def __init__(self, ds9_path: str, zoom: str = None, zscale: bool = False):
        """Constructor.

        Args:
            ds9_path (str): location of DS9 binary
            zoom (str, optional): zoom options (http://ds9.si.edu/doc/ref/command.html#zoom). Defaults to None.
            zscale (bool, optional): scale options (http://ds9.si.edu/doc/ref/command.html#zscale). Defaults to False.
        """
        super().__init__()
        self.ds9_path = ds9_path
        self.current_process = None
        self.zoom = zoom
        self.zscale = zscale

    def _append_arg(self, args:List[str], arg: str):
        for word in arg.split():
            args.append(word)
    
    def _append_fits_args(self, fits_path: str, args: List[str]):
        args.append(fits_path)
            
        if self.zoom:
            self._append_arg(args, '-zoom')
            self._append_arg(args, self.zoom)
        
        if self.zscale:
            self._append_arg(args, '-zscale')
    
    def _generate_region_command(self, ra: float, dec: float, radius: float, color: str) -> str:
        return ''.join(['fk5;circle(',
                        str(ra),
                        ',',
                        str(dec),
                        ',',
                        str(radius),
                        ') # color=',
                        color])
        
    def display(self, fits_paths: List[str], observation: observation.Observation = None):
        """See base class."""       
        self.close_current_display()
        
        args = [self.ds9_path]
        
        for fits_path in fits_paths:
            self._append_fits_args(fits_path=fits_path, args=args)
        
        if observation:
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_reg_file:
                temp_reg_file.write(self._generate_region_command(observation.ra1, observation.dec1, 0.01, 'green'))
                temp_reg_file.write(os.linesep)
                temp_reg_file.write(self._generate_region_command(observation.ra2, observation.dec2, 0.01, 'red'))
                
                args.extend(['-regions',
                             'load',
                             'all',
                             temp_reg_file.name])
        
            self.current_process = subprocess.Popen(args=args)
        
    def close_current_display(self):
        """See base class."""
        if self.current_process:
            self.current_process.terminate()
            
        self.current_process = None
