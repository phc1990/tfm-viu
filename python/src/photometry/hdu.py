"""
Header Data Units (HDUs) module.

https://fits.gsfc.nasa.gov/fits_primer.html
"""

import logging
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
# old: from photutils import DAOStarFinder
try:
    from photutils.detection import DAOStarFinder  # photutils ≥ 1.0
except Exception:
    from photutils import DAOStarFinder            # very old photutils


class HDUW:
    """Header Data Unit(s) wrapper for XMM-OM FITS images."""

    def __init__(self, file, sigma: float = 3.0) -> None:
        """Constructor. Opens a FITS image and extracts metadata and image data.

        Args:
            file (str): FITS image file path.
            sigma (float, optional): Sigma value for background stats. Defaults to 3.0.
        """
        logging.debug(f'Opening image {file}')
        _file = fits.open(file)
        self.hdu = _file[0]

        # ✅ Add image data attribute
        self.data = self.hdu.data

        # Metadata
        self.epoch = self.hdu.header.get('DATE-OBS')
        self.filter_name = self.hdu.header.get('FILTER')
        self.naxis1 = self.hdu.header.get('NAXIS1')
        self.naxis2 = self.hdu.header.get('NAXIS2')
        self.observation_id = self.hdu.header.get('OBS_ID')
        self.texp = float(self.hdu.header.get('EXPOSURE', 1.0))  # fallback to 1.0 if missing

        logging.debug(f'Created HDU {self.naxis1}x{self.naxis2} taken on {self.epoch}, with the {self.filter_name} filter.')
        logging.debug(f'Extracted exposure (s): {self.texp}')

        # WCS and background stats
        self.wcs = WCS(self.hdu.header)
        logging.debug('Extracted WCS from image.')

        self.bkg_mean, self.bkg_median, self.bkg_sigma = sigma_clipped_stats(self.data, sigma=sigma)
        logging.debug(f'HDU background metrics: mean={self.bkg_mean}, median={self.bkg_median}, sigma={self.bkg_sigma}')

    def find_sources(self, fwhm: float):
        """Finds and returns a list of sources using DAOStarFinder.

        Args:
            fwhm (float): Full Width Half Maximum (FWHM) to use.

        Returns:
            Table: Detected sources.
        """
        daofind = DAOStarFinder(
            fwhm=fwhm,
            threshold=self.bkg_median + 3.0 * self.bkg_sigma  # TODO: make sigma configurable
        )

        # TODO: remove sources close to the edges
        return daofind(self.data)