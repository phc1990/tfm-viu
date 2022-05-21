import sys
import configparser

from astropy.io import fits, ascii
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS

import matplotlib.pyplot as plt

from photutils import DAOStarFinder
from photutils import (aperture_photometry,
                       CircularAperture,
                       CircularAnnulus)

if __name__ == '__main__':

    
    print("**** START ****")
    args = sys.argv 
    
    if len(args) < 2:
        raise Exception('Missing .ini configuration file location.')
    elif len(args) > 2:
        raise Exception('Too many arguments.')
    
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config.read(args[1])
    required = config['REQUIRED']

    hdu = fits.open(required['IMAGE'])
    hdu = hdu[0]
    hdu.header
    epoch = hdu.header.get('DATE-OBS')
    filter_name = hdu.header.get('FILTER')
    naxis1 = hdu.header.get('NAXIS1')
    naxis2 = hdu.header.get('NAXIS2')
    
    print(f'Image {naxis1}x{naxis2} taken on {epoch}, with the {filter_name} filter.')
    bkg_mean, bkg_median, bkg_sigma = sigma_clipped_stats(hdu.data, sigma=3.0)
    wcs = WCS(hdu.header)
    print(f'bkg mean {bkg_mean}, bkg_med {bkg_median}, bkg_sigma{bkg_sigma}')
    
    fig = plt.figure()
    ax = plt.subplot(projection=wcs)
    im = ax.imshow( hdu.data,
                    cmap='Greys',
                    origin='lower',
                    vmin=bkg_median-3*bkg_sigma,
                    vmax=bkg_median+3*bkg_sigma)
    fig.colorbar(im, ax=ax)
    plt.show()


    fwhm=float(required['SOURCE_DETECTION_FWHM'])

    daofind = DAOStarFinder(fwhm=fwhm,
                            threshold=bkg_median+3.*bkg_sigma)

    sources = daofind(hdu.data)

    print('Number of sources detected: {:d}'.format(len(sources)))
    print('\n')
    print("**** END ****")


