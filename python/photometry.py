import sys

import tempfile

from src.config import config_from_sys_ars
from src.photometry.hdu import HDUW
from src.photometry.phot import PhotTable
from src.photometry.ui import UI, TrailSelector

from astroquery.esa.xmm_newton import XMMNewton
            

def _calibrate_using_source_list(phot_table: PhotTable):
    hduw = phot_table.hduw
    with tempfile.NamedTemporaryFile(mode='w+b') as source_list_file:
        XMMNewton.download_data(hduw.observation_id, filename=source_list_file.name, instname='OM', name='OBSMLI', level='PPS', extension='FTZ')
        phot_table.calibrate_against_source_list(source_list_file=source_list_file.name+'.FTZ',
                                                 filter=hduw.filter_name)

def _calibrate_using_cataloge(phot_table: PhotTable):
    config = config_from_sys_ars()
    required = config.child('REQUIRED')
    
    fwhm = required.get_float('SOURCE_DETECTION_FWHM')
    sources = hduw.find_sources(fwhm=fwhm)
        
    phot_table.add_sources_apertures(sources=sources,
                                     aperture_radius=2*fwhm,
                                     annular_aperture_start_offset=required.get_float('SOURCE_ANNULAR_APERTURE_START_OFFSET'),
                                     annular_aperture_end_offset=required.get_float('SOURCE_ANNULAR_APERTURE_END_OFFSET'))
    
    # TODO this to be probably picked by user
    phot_table.calibrate_against_catalogue(start_magnitude=15.0,
                                           end_magnitude=18.0)
    
    ui.add_sources(sources=sources)


if __name__ == '__main__':

    hduw = HDUW(file=sys.argv[1])
    ui = UI(hduw=hduw)
    phot_table = PhotTable(hduw=hduw)

    _calibrate_using_source_list(phot_table=phot_table)
    
    # TODO this params
    trail_selector = TrailSelector(height=7, semi_out=2)
    ui.select_trail(selector=trail_selector)

    magnitude = phot_table.perform_trail_photometry( rectangular_aperture=trail_selector.rectangular_aperture,
                                                     rectangular_annulus=trail_selector.rectangular_annulus)

    print(magnitude)