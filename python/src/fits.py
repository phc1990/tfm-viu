"""Flexible Image Transport System (FITS) module.

https://fits.gsfc.nasa.gov/
"""

from typing import List
import subprocess


class Region:
    """A spatial region to be overlaid on a FITS."""
    
    def __init__(self, label: str, ra: float, dec: float, rad: float, color: str):
        self.label = label
        self.ra = ra
        self.dec = dec
        self.rad = rad
        self.color = color
        pass
    

class FitsUi:
    """Flexible Image Transport System (FITS) user interface (UI)."""
    
    def display(self, fits_paths: List[str], regions: List[Region]=[]):
        """Displays the specified FITS with the specified regions overlaid on it.

        Args:
            fits_paths (List[str]): list of FITS paths
            regions (List[Region], optional): list of regiosn to overlaid. Defaults to [].
        """
        pass
    
    def close_current_display(self):
        """Closes the current display."""
        pass
    

class FitsDs9Ui(FitsUi):
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
        
    def display(self, fits_paths: List[str], regions: List[Region]=[]):
        """See base class."""       
        self.close_current_display()
        
        args = [self.ds9_path]
        
        for fits_path in fits_paths:
            self._append_fits_args(fits_path=fits_path, args=args)
        
        self.current_process = subprocess.Popen(args=args)
        
    def close_current_display(self):
        """See base class."""
        if self.current_process:
            self.current_process.terminate()

        
