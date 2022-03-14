import unittest
import tempfile
import src.xsa as xsa
import os

class TestXsaHttpDownloader(unittest.TestCase):

    def test_download_single_filter(self):
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = temp_dir + 'a.tar'
            downloader = xsa.XsaHttpDownloader(baseUrl='http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio')
            downloader._download_single_filter(filepath=filepath,
                                               observation_id='0781040101',
                                               filter='UVW1')
            self.assertTrue(os.path.exists(filepath))
            self.assertEqual(os.path.getsize(filepath), 12943360)
            
            
        