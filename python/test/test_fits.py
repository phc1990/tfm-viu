import unittest
import src.fits as fits
import test.helper as helper
import time


class TestDs9Ui(unittest.TestCase):
    
    def test_display_1_fits(self):
        ds9_ui = fits.Ds9Interface(ds9_path=helper.get_test_data_file_path('ds9'))
        
        ds9_ui.display(fits_paths=[helper.get_test_data_file_path('fits1.fits')])
        time.sleep(15)
        ds9_ui.close_current_display()
        
    def test_display_1_fits_with_zoom_to_fit_and_zscale(self):
        ds9_ui = fits.Ds9Interface(ds9_path=helper.get_test_data_file_path('ds9'),
                                   zoom='to fit',
                                   zscale=True)
        
        ds9_ui.display(fits_paths=[helper.get_test_data_file_path('fits1.fits')])
        time.sleep(15)
        ds9_ui.close_current_display()
        
    def test_display_1_ftz(self):
        ds9_ui = fits.Ds9Interface(ds9_path=helper.get_test_data_file_path('ds9'))
        
        ds9_ui.display(fits_paths=[helper.get_test_data_file_path('fits1.ftz')])
        time.sleep(15)
        ds9_ui.close_current_display()
        
    def test_display_1_ftz_with_zoom_to_fit_and_zscale(self):
        ds9_ui = fits.Ds9Interface(ds9_path=helper.get_test_data_file_path('ds9'),
                                   zoom='to fit',
                                   zscale=True)
        
        ds9_ui.display(fits_paths=[helper.get_test_data_file_path('fits1.ftz')])
        time.sleep(15)
        ds9_ui.close_current_display()
        
    def test_display_1_fits_and_1_ftz_with_zoom_to_fit_and_zscale(self):
        ds9_ui = fits.Ds9Interface(ds9_path=helper.get_test_data_file_path('ds9'),
                                   zoom='to fit',
                                   zscale=True)
        
        ds9_ui.display(fits_paths=[
            helper.get_test_data_file_path('fits1.fits'),
            helper.get_test_data_file_path('fits1.ftz')])
        
        time.sleep(15)
        ds9_ui.close_current_display()
        
        