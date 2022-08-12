"""Photometry module.
"""

from re import A
import numpy as np

from photutils import (aperture_photometry,
                       CircularAperture,
                       CircularAnnulus,
                       RectangularAperture,
                       RectangularAnnulus)

from astropy import units as u
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord

from astroquery.mast import Catalogs

from sklearn import linear_model

from scipy import stats

from src.photometry.hdu import HDUW


class _LinearModel:
    def __init__(self, a: float, b: float) -> None:
        self.a = a
        self.b = b
    
    def predict(self, x: float) -> float:
        return self.a * x + self.b


class PhotTable:
    """Photometry Table wrapper. It contains a native astropy QTable.
    
    https://docs.astropy.org/en/stable/api/astropy.table.QTable.html#astropy.table.QTable
    """
    
    def __init__(self, hduw: HDUW) -> None:
        """Constructor.

        Args:
            hduw (HDUW): Header Data Unit(s) (HDU) wrapper.
        """
        self.hduw = hduw
    
    # TODO remove sources close to edges
    def add_sources_apertures(  self, sources,
                                aperture_radius: int,
                                annular_aperture_start_offset: int,
                                annular_aperture_end_offset: int) -> None:
        """Performs and adds sources aperture information to the native table.

        Args:
            sources (_type_): _description_
            aperture_radius (int): _description_
            annular_aperture_start_offset (int): _description_
            annular_aperture_end_offset (int): _description_
        """
            
        annulus_radius = [aperture_radius + annular_aperture_start_offset, 
                          aperture_radius + annular_aperture_end_offset]

        positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
        self.source_apertures = CircularAperture(   positions,
                                                    r=aperture_radius)

        # https://photutils.readthedocs.io/en/stable/api/photutils.aperture.aperture_photometry.html#photutils.aperture.aperture_photometry
        self.qtable = aperture_photometry(  self.hduw.hdu.data, 
                                            apertures=self.source_apertures)

        annulus_aperture = CircularAnnulus( positions,
                                            r_in=annulus_radius[0],
                                            r_out=annulus_radius[1])

        annulus_masks = annulus_aperture.to_mask(method='center')

        bkg_median_arr = []
        for mask in annulus_masks:
            annulus_data = mask.multiply(self.hduw.hdu.data)
            #TODO understand this
            annulus_data_1d = annulus_data[mask.data > 0]
            _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
            bkg_median_arr.append(median_sigclip)

        bkg_median_arr = np.array(bkg_median_arr)
        self.qtable['annulus_median'] = bkg_median_arr
        self.qtable['aper_bkg'] = bkg_median_arr * self.source_apertures.area
        self.qtable['aper_sum_bkgsub'] = self.qtable['aperture_sum'] - self.qtable['aper_bkg']

        # TODO why is aper_sum_bkgsub a noise contribution? Why are SNR > 1?
        self.qtable['noise'] = np.sqrt(self.qtable['aper_sum_bkgsub'] + self.qtable['aper_bkg'])
        self.qtable['SNR'] = self.qtable['aper_sum_bkgsub'] / self.qtable['noise']
        
    
    def calibrate_against_source_list(self, source_list_file, filter) -> None:
        hdul = fits.open(source_list_file)
        zero_point, slope = hdul[1].header['ABM0'+filter], hdul[1].header['ABF0'+filter]
        self.fitting_model = _LinearModel(slope, zero_point)
        
        
    def calibrate_against_catalogue(self, start_magnitude: float, end_magnitude: float) -> None:
        field = SkyCoord(self.hduw.wcs.wcs.crval[0], self.hduw.wcs.wcs.crval[1], unit=u.deg)

        # TODO understand this params
        # For sure radius to make sure we don't miss out on any stars
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


        source_apertures_coordinates = self.source_apertures.to_sky(self.hduw.wcs).positions
        catalogue_coordinates = SkyCoord(   ra=catalog['raMean'] * u.deg,
                                            dec=catalog['decMean'] * u.deg)


        # https://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html#astropy.coordinates.SkyCoord.match_to_catalog_sky    
        # xm_id: indices of the catalog that have matched
        # xm_ang_distance: angular separation (Angle)
        # _: would have been 3D distance
        cross_matching_indices, cross_matching_angular_separation, _ = source_apertures_coordinates.match_to_catalog_sky(catalogue_coordinates, nthneighbor=1)

        # TODO need to understand what the equivalent for PIXSCALX is in our system
        #max_sep = hdu.header.get('PIXSCALX') * fwhm * u.arcsec
        maximum_separation = 2.5 * u.arcsec
        
        # Record the RA/Dec of apertures
        self.qtable['ra'] = source_apertures_coordinates.ra.value
        self.qtable['dec'] = source_apertures_coordinates.dec.value

        separation_constraint = cross_matching_angular_separation < maximum_separation
        
        ins_mag = -2.5*np.log10( self.qtable[separation_constraint]['aper_sum_bkgsub']/self.hduw.exposure_sec )
        # TODO this obtains the magnitude regardless of the wavelenght? Should this not be based on wavelength?
        # TODO not used?
        cat_mag = catalog['iMeanPSFMag'][cross_matching_indices[separation_constraint]]

        # TODO looks like there was a typo here where it had '--'
        ins_err = ins_mag - 2.5*np.log10( (self.qtable[separation_constraint]['aper_sum_bkgsub'] + self.qtable[separation_constraint]['noise'])/self.hduw.exposure_sec )
        cat_err = catalog['iMeanPSFMagErr'][cross_matching_indices[separation_constraint]]

        # TODO needed to add .0 here otherwise ins_mag was stored as an int
        self.qtable['ins_mag'] = 0.0
        self.qtable['ins_mag'][separation_constraint] = ins_mag

        fitting_condition = (cat_mag>start_magnitude) & (cat_mag<end_magnitude) & (~cat_mag.mask) & (~np.isnan(ins_mag))

        # Create two mock arrays for linear regression
        X = ins_mag[fitting_condition].reshape(-1, 1)
        y = cat_mag[fitting_condition].reshape(-1, 1)

        # Simple linear regression
        linear = linear_model.LinearRegression()
        linear.fit(X, y)
        
        # sigma clipping pour choisir le threshold
        MAD = stats.median_abs_deviation(X-y)
        # TODO not used
        _, _, sig = sigma_clipped_stats(X-y)

        # RANSAC linear regressions
        ransac = linear_model.RANSACRegressor(residual_threshold=3*MAD)
        ransac.fit(X, y)

        # Choose the model with the slope closest to 1
        if (abs(linear.coef_[0][0] - 1.0) <= abs(ransac.estimator_.coef_[0][0] - 1.0)):
            self.fitting_model = linear
        else:
            self.fitting_model = ransac

        # Positive values
        positive = np.where( self.qtable['aper_sum_bkgsub']>0 )

        # Compute calibrated mag
        self.qtable['mag'] = 0.
        self.qtable['mag'][positive] = self.fitting_model.predict( (-2.5*np.log10( self.qtable[positive]['aper_sum_bkgsub']/self.hduw.exposure_sec)).data.reshape(-1,1)).flatten()

        # Remove points not used for the fitting
        fitting_condition = self.qtable['mag'] > 0
        self.qtable = self.qtable[fitting_condition]

    def perform_trail_photometry(self, rectangular_aperture: RectangularAperture, rectangular_annulus: RectangularAnnulus) -> float:
        target_qtable = aperture_photometry(self.hduw.hdu.data, rectangular_aperture)

        annulus_mask_target = rectangular_annulus.to_mask(method='center')
        annulus_data_target = annulus_mask_target.multiply(self.hduw.hdu.data)
        annulus_data_target_1d = annulus_data_target[annulus_mask_target.data > 0]
        _, median_sigclip_target, _ = sigma_clipped_stats(annulus_data_target_1d)
        bkg_median_target = median_sigclip_target

        target_qtable['annulus_median'] = bkg_median_target
        target_qtable['aper_bkg'] = bkg_median_target * rectangular_aperture.area
        target_qtable['aper_sum_bkgsub'] = target_qtable['aperture_sum'] - target_qtable['aper_bkg']

        cal_mag = self.fitting_model.predict( (-2.5*np.log10( target_qtable['aper_sum_bkgsub']/self.hduw.exposure_sec)).data.reshape(-1,1))
        target_qtable['mag'] = cal_mag

        return cal_mag
    
