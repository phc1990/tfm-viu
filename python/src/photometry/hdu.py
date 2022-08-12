"""Header Data Units (HDUs) module.

https://fits.gsfc.nasa.gov/fits_primer.html
"""


import logging

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS

from photutils import DAOStarFinder


class HDUW:
    """Header Data Unit(s) wrapper for XMM-OMM FITS images.
    """

    def __init__(self, file, sigma: float = 3.0) -> None:
        """Constructor. It will open a FITS image at the given file location.

        Args:
            file (_type_): FITS image file path.
            sigma (float, optional): sigma value to be used when computing background stats. Defaults to 3.0.
        """
        logging.debug(f'Opening image {file}')
        _file = fits.open(file)
        self.hdu = _file[0]
        
        self.epoch              = self.hdu.header.get('DATE-OBS')
        self.filter_name        = self.hdu.header.get('FILTER')
        self.naxis1             = self.hdu.header.get('NAXIS1')
        self.naxis2             = self.hdu.header.get('NAXIS2')
        self.observation_id     = self.hdu.header.get('OBS_ID')
        logging.debug(f'Created HDU {self.naxis1}x{self.naxis2} taken on {self.epoch}, with the {self.filter_name} filter.')
        
        self.wcs = WCS(self.hdu.header)
        logging.debug(f'Extracted WCS from image.')

        self.bkg_mean, self.bkg_median, self.bkg_sigma = sigma_clipped_stats(self.hdu.data, sigma=3.0)
        logging.debug(f'HDU background metrics: mean={self.bkg_mean}, median={self.bkg_median}, sigma={self.bkg_sigma}')
        
        self.exposure_sec = float(self.hdu.header.get('EXPOSURE'))
        logging.debug(f'Extracted exposure (s): {self.exposure_sec}')
        
    def find_sources(self, fwhm: float):
        """Finds a returns a list of sources.

        Args:
            fwhm (float): Full Width Half Maximum (FWHM) to use.

        Returns:
            _type_: an array of sources.
        """
        daofind = DAOStarFinder(    fwhm=fwhm,
                                    #TODO should this 3.0 be consistent with the sigma used in the background calculations?
                                    threshold=self.bkg_median + 3.0 * self.bkg_sigma)
        
        # TODO: remove sources close to the edges
        return daofind(self.hdu.data)
    