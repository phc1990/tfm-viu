import unittest
import tempfile
import src.xsa as xsa
import os


class TestHttpDownloader(unittest.TestCase):
 
    def test_download_single_filter_as_tar(self): 
        downloader = xsa.HttpDownloader(base_url='http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio',
                                           regex_patern='^.*?FSIMAG.*?\.FTZ$')
        
        with tempfile.TemporaryFile(mode='wb') as file:
            downloader._download_single_filter_as_tar(file=file,
                                                      observation_id='0781040101',
                                                      filter='UVW1')
        
            self.assertTrue(os.path.exists(file.name))
            self.assertEqual(os.path.getsize(file.name), 12943360)
            
    def test_download(self): 
        downloader = xsa.HttpDownloader(base_url='http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio',
                                        regex_patern='^.*?FSIMAG.*?\.FTZ$')
        
        with tempfile.TemporaryDirectory() as output_dir:
            
            results = downloader.download(output_dir=output_dir,
                                          observation_id='0781040101',
                                          filters=['UVW1','UVM2'])

            self.assertEqual(len(results['UVW1']), 2)
            self.assertEqual(len(results['UVM2']), 4)
            self.assertFalse('UVW2' in results)
            