import sys
import configparser
import logging
from typing import Any
from xmlrpc.client import Boolean

from src.config import get_field_or_fail, raise_unrecognised_field_value

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from photutils import DAOStarFinder
from photutils import (aperture_photometry,
                       CircularAperture,
                       CircularAnnulus,
                       RectangularAperture,
                       RectangularAnnulus)

from sklearn import linear_model

from scipy import stats

import numpy as np

from astroquery.mast import Catalogs


class HDUW:

    def __init__(self, file) -> None:
        logging.debug(f'Opening image {file}')
        _file = fits.open(file)
        self.hdu = _file[0]

        self.epoch       = self.hdu.header.get('DATE-OBS')
        self.filter_name = self.hdu.header.get('FILTER')
        self.naxis1      = self.hdu.header.get('NAXIS1')
        self.naxis2      = self.hdu.header.get('NAXIS2')
        logging.debug(f'Created HDU {self.naxis1}x{self.naxis2} taken on {self.epoch}, with the {self.filter_name} filter.')
        
        self.wcs = WCS(self.hdu.header)
        logging.debug(f'Extracted WCS from image.')

        self.bkg_mean, self.bkg_median, self.bkg_sigma = sigma_clipped_stats(self.hdu.data, sigma=3.0)
        logging.debug(f'HDU background metrics: mean={self.bkg_mean}, median={self.bkg_median}, sigma={self.bkg_sigma}')
      


class UI:

    def __init__(self, hduw: HDUW) -> None:
        plt.ion()

        self.hduw = hduw        
        self.fig = plt.figure()
        self.ax = plt.subplot(projection=hduw.wcs)
        self.im = self.ax.imshow( hduw.hdu.data,
                        cmap='Greys',
                        origin='lower',
                        vmin=hduw.bkg_median-3*hduw.bkg_sigma,
                        vmax=hduw.bkg_median+3*hduw.bkg_sigma)
        self.fig.colorbar(self.im, ax=self.ax)

    def add_sources(self, sources):
        self.ax.scatter( sources['xcentroid'],
                    sources['ycentroid'],
                    color='yellow',
                    alpha=0.4)
        self.update()

    def select_trail(self, selector):
        # Build with event handling: https://matplotlib.org/stable/users/explain/event_handling.html
        self.fig.canvas.mpl_connect('button_press_event', selector.onclick)
        while selector.is_done() == False:
            plt.waitforbuttonpress()
                                                
        self.ax.add_patch(selector.rectangular_aperture._to_patch(  fill=True,
                                                                    color='blue',
                                                                    alpha=0.5))

        self.ax.add_patch(selector.rectangular_annulus._to_patch(   fill=True,
                                                                    color='red',
                                                                    alpha=0.2))

        self.update()
        plt.waitforbuttonpress()

    def update(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

class TrailSelector:

    def __init__(self, height: int, semi_out: int) -> None:
        self.clicks = 0
        self.height = height
        self.semi_out = semi_out

    def onclick(self, event):
        if self.clicks == 0:
            self.event1 = event
        elif self.clicks == 1:
            self.event2 = event
            self._compute_properties()
        self.clicks +=1

    def _compute_properties(self):
        self.centre =    [0.5*(self.event1.xdata + self.event2.xdata),
                          0.5*(self.event1.ydata + self.event2.ydata)]
        self.width = np.sqrt(np.power(self.event1.xdata - self.event2.xdata, 2) + np.power(self.event1.ydata - self.event2.ydata, 2))
        self.theta = np.arctan((self.event1.ydata - self.centre[1])/(self.event1.xdata - self.centre[0]))
        self.rectangular_aperture = RectangularAperture(positions=self.centre,
                                                        w=self.width,
                                                        h=self.height,
                                                        theta=self.theta)
        self.rectangular_annulus = RectangularAnnulus(  positions=self.centre,
                                                        w_in=self.width,
                                                        w_out=self.width + 2 * self.semi_out,
                                                        h_in=self.height,
                                                        h_out=self.height + 2 * self.semi_out,
                                                        theta=self.theta)

    def is_done(self) -> Boolean:
        if self.clicks >= 2:
            return True
        return False
            

if __name__ == '__main__':
    args = sys.argv 
    
    if len(args) < 2:
        raise Exception('Missing .ini configuration file location.')
    elif len(args) > 2:
        raise Exception('Too many arguments.')
    
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config.read(args[1])
    required = get_field_or_fail(config, 'REQUIRED')

    log_level_config = get_field_or_fail(required, 'LOG_LEVEL')
    log_level = None
    if log_level_config == 'DEBUG':
        log_level = logging.DEBUG
    elif log_level_config == 'INFO':
        log_level = logging.INFO
    else:
        raise_unrecognised_field_value('LOG_LEVEL', log_level_config)
    
    logging.basicConfig(level=log_level, format='%(asctime)s  %(levelname)s %(message)s')
    logging.info('Starting photometry script...')

    required = get_field_or_fail(config, 'REQUIRED')
    image_file = get_field_or_fail(required, 'IMAGE')
    
    logging.info(f'Loading image from {image_file}')
    hduw = HDUW(file=image_file)
    
    
    ui = UI(hdu=hdu)    

    # TODO: obtain from??
    # FIXME: articulos sobre OM deberia de aparecer
    fwhm=float(required['SOURCE_DETECTION_FWHM'])

    daofind = DAOStarFinder(fwhm=fwhm,
                            threshold=bkg_median+3.*bkg_sigma)

    sources = daofind(hdu.data)
    print('Number of sources detected: {:d}'.format(len(sources)))
    
    ui.add_sources(sources=sources)

    
    # TODO: REMOVE SOURCES CLOSE TO EDGES
    
    
    aperture_radius = 2.0 * fwhm
    annulus_radius = [aperture_radius+2, aperture_radius+5]

    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    apertures = CircularAperture(   positions,
                                    r=aperture_radius)

    # aperture_photometry: https://photutils.readthedocs.io/en/stable/api/photutils.aperture.aperture_photometry.html#photutils.aperture.aperture_photometry
    # QTable: https://docs.astropy.org/en/stable/api/astropy.table.QTable.html#astropy.table.QTable
    phot_table = aperture_photometry(   hdu.data, 
                                        apertures=apertures)

    annulus_aperture = CircularAnnulus( positions,
                                        r_in=annulus_radius[0],
                                        r_out=annulus_radius[1])

    annulus_masks = annulus_aperture.to_mask(method='center')

    bkg_median_arr = []
    for mask in annulus_masks:
        annulus_data = mask.multiply(hdu.data)
        #TODO understand this
        annulus_data_1d = annulus_data[mask.data > 0]
        _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
        bkg_median_arr.append(median_sigclip)

    bkg_median_arr = np.array(bkg_median_arr)
    phot_table['annulus_median'] = bkg_median_arr
    phot_table['aper_bkg'] = bkg_median_arr * apertures.area
    phot_table['aper_sum_bkgsub'] = phot_table['aperture_sum'] - phot_table['aper_bkg']

    # TODO why is aper_sum_bkgsub a noise contribution? Why are SNR > 1?
    phot_table['noise'] = np.sqrt(phot_table['aper_sum_bkgsub'] + phot_table['aper_bkg'])
    phot_table['SNR'] = phot_table['aper_sum_bkgsub'] / phot_table['noise']

    field = SkyCoord(wcs.wcs.crval[0], wcs.wcs.crval[1], unit=u.deg)

    # TODO understand this params
    # For sure radius to make sure we don't miss out on any stars
    print('Performing catalog')
    """FIXME
    - Maybe we can use PANSTAARS -> No
    - Elena to send the query to obtain biproduct of catalog
    - Use XMM SUSS
    """
    catalog = Catalogs.query_criteria(coordinates=field.to_string(), radius=0.25,
                                  catalog="PANSTARRS", table="mean", data_release="dr2",
                                  nStackDetections=[("gte", 1)],
                                  iMeanPSFMag=[("lt", 21), ("gt", 1)], 
                                  iMeanPSFMagErr=[("lt", 0.02), ("gt", 0.0)], 
                                  columns=['objName','raMean', 'decMean','nDetections',
                                           'iMeanPSFMag', 'iMeanPSFMagErr' ])


    coord_apertures = apertures.to_sky(wcs).positions
    coord_catalog = SkyCoord(   ra=catalog['raMean']*u.deg,
                                dec=catalog['decMean']*u.deg)


    # https://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html#astropy.coordinates.SkyCoord.match_to_catalog_sky    
    # xm_id: indices of the catalog that have matched
    # xm_ang_distance: angular separation (Angle)
    # _: would have been 3D distance
    xm_id, xm_ang_distance, _ = coord_apertures.match_to_catalog_sky(coord_catalog, nthneighbor=1)

    # TODO need to understand what the equivalent for PIXSCALX is in our system
    #max_sep = hdu.header.get('PIXSCALX') * fwhm * u.arcsec
    max_sep = 2.5 * u.arcsec

    sep_constraint = xm_ang_distance < max_sep
    coord_matches = coord_apertures[sep_constraint]
    
    catalog_matches = catalog[xm_id[sep_constraint]]
    coord_catalog_matches = coord_catalog[xm_id[sep_constraint]]
    
    # Record the RA/Dec of apertures
    phot_table['ra'] = coord_apertures.ra.value
    phot_table['dec'] = coord_apertures.dec.value

    # Exposure seems to be in seconds
    exptime = float(hdu.header.get('EXPOSURE'))

    ins_mag = -2.5*np.log10( phot_table[sep_constraint]['aper_sum_bkgsub']/exptime )
    # TODO this obtians the magnitude regardless of the wavelenght? Should this not be based on wavelength?
    cat_mag = catalog['iMeanPSFMag'][xm_id[sep_constraint]]

    # TODO looks like there was a typo here where it had '--'
    ins_err = ins_mag - 2.5*np.log10( (phot_table[sep_constraint]['aper_sum_bkgsub']+phot_table[sep_constraint]['noise'])/exptime )
    cat_err = catalog['iMeanPSFMagErr'][xm_id[sep_constraint]]

    # TODO needed to add .0 here otherwise ins_mag was stored as an int
    phot_table['ins_mag'] = 0.0
    phot_table['ins_mag'][sep_constraint] = ins_mag

    
    fig, ax = plt.subplots(1,2,figsize=(15,5))

    # Instrumental vs catalog mag
    ax[0].errorbar(ins_mag, cat_mag, xerr=ins_err, yerr=cat_err, marker='.', linestyle='none', label='All sources')
    ax[0].set_xlabel('Instrument magnitude')
    ax[0].set_ylabel('Catalog magnitude')
    ax[0].legend(loc='best')
    ax[0].set_aspect('equal')

    # Magnitude difference as function of catalog magnitude
    ax[1].errorbar(cat_mag, cat_mag-ins_mag, xerr=cat_err, yerr=(cat_err+ins_err), marker='.', linestyle='none', label='All sources')
    ax[1].set_xlabel('Catalog magnitude')
    ax[1].set_ylabel('Instrument - Catalog magnitude')
    ax[1].legend(loc='best')
    

    # TODO maybe from config? Does this depend on filter?
    # Selection from magnitude range
    mag_min, mag_max = 15.0, 18.0
    cond = (cat_mag>mag_min) & (cat_mag<mag_max) & \
        (~cat_mag.mask) & (~np.isnan(ins_mag))

    # Create two mock arrays for linear regression
    X = ins_mag[cond].reshape(-1, 1)
    y = cat_mag[cond].reshape(-1, 1)

    # Simple linear regression
    linear = linear_model.LinearRegression()
    linear.fit(X, y)
    
    # sigma clipping pour choisir le threshold
    MAD = stats.median_abs_deviation(X-y)
    _, _, sig = sigma_clipped_stats(X-y)


    # RANSAC linear regressions
    ransac = linear_model.RANSACRegressor(residual_threshold=3*MAD)
    ransac.fit(X, y)

    # Results
    # TODO we do not get a slope of 1
    print('Photometric calibration:')
    print( f'  Linear Slope: {linear.coef_[0][0]:.3f}')
    print( f'  Linear ZP   : {linear.intercept_[0]:.3f}\n')
    print( f'  RANSAC Slope: {ransac.estimator_.coef_[0][0]:.3f}')
    print( f'  RANSAC ZP   : {ransac.estimator_.intercept_[0]:.3f}')

    # Positive values
    positive = np.where( phot_table['aper_sum_bkgsub']>0 )

    # Compute calibrated mag
    phot_table['mag'] = 0.
    phot_table['mag'][positive] = ransac.predict( (-2.5*np.log10( phot_table[positive]['aper_sum_bkgsub']/exptime)).data.reshape(-1,1)).flatten()

    cond = phot_table['mag']>0
    phot_table2 = phot_table[cond]

    # TODO select height & semi_out
    selector = TrailSelector(height=7, semi_out=2)
    ui.select_trail(selector)

    # TODO handle points outside the image
    aperture_target = selector.rectangular_aperture
    annuluse_target = selector.rectangular_annulus
    phot_target = aperture_photometry(hdu.data, aperture_target)


    annulus_mask_target = annuluse_target.to_mask(method='center')
    annulus_data_target = annulus_mask_target.multiply(hdu.data)
    annulus_data_target_1d = annulus_data_target[annulus_mask_target.data > 0]
    _, median_sigclip_target, _ = sigma_clipped_stats(annulus_data_target_1d)
    bkg_median_target = median_sigclip_target

    phot_target['annulus_median'] = bkg_median_target
    phot_target['aper_bkg'] = bkg_median_target * aperture_target.area
    phot_target['aper_sum_bkgsub'] = phot_target['aperture_sum'] - phot_target['aper_bkg']

    cal_mag = ransac.predict( (-2.5*np.log10( phot_target['aper_sum_bkgsub']/exptime)).data.reshape(-1,1))
    phot_target['mag'] = cal_mag


    print(phot_target)
    