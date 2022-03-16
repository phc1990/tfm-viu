import unittest
import tempfile
import src.xsa as xsa
import os


class TestHttpPythonCrawler(unittest.TestCase):
 
    def test_download_single_filter_as_tar(self): 
        with tempfile.TemporaryDirectory() as download_dir:
            crawler = xsa.HttpPythonRequestsCrawler(download_dir=download_dir,
                                                    base_url='http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio',
                                                    regex_patern='^.*?FSIMAG.*?\.FTZ$')
            
            with tempfile.NamedTemporaryFile(mode='wb') as temp_file:
                crawler._download_single_filter_as_tar(file=temp_file,
                                                       observation_id='0781040101',
                                                       filter='UVW1')
            
                self.assertTrue(os.path.exists(temp_file.name))
                self.assertEqual(os.path.getsize(temp_file.name), 12943360)

    def test_crawl(self): 
        with tempfile.TemporaryDirectory() as download_dir:
            crawler = xsa.HttpPythonRequestsCrawler(download_dir=download_dir,
                                                    base_url='http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio',
                                                    regex_patern='^.*?FSIMAG.*?\.FTZ$')
            
            results = crawler.crawl(observation_id='0781040101',
                                       filters=['UVW1','UVM2'])

            self.assertEqual(len(results['UVW1']), 2)
            self.assertEqual(len(results['UVM2']), 4)
            self.assertFalse('UVW2' in results)
            
            
class TestHttpCurlCrawler(unittest.TestCase):
 
    def test_download_single_filter_as_tar(self): 
        with tempfile.TemporaryDirectory() as download_dir:
            crawler = xsa.HttpCurlCrawler(download_dir=download_dir,
                                          base_url='http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio',
                                          regex_patern='^.*?FSIMAG.*?\.FTZ$')
            
            with tempfile.NamedTemporaryFile(mode='wb') as temp_file:
                crawler._download_single_filter_as_tar(file=temp_file,
                                                       observation_id='0781040101',
                                                       filter='UVW1')
            
                self.assertTrue(os.path.exists(temp_file.name))
                self.assertEqual(os.path.getsize(temp_file.name), 12943360)
                
    def test_crawl(self): 
        with tempfile.TemporaryDirectory() as download_dir:
            
            crawler = xsa.HttpCurlCrawler(download_dir=download_dir,
                                          base_url='http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio',
                                          regex_patern='^.*?FSIMAG.*?\.FTZ$')
            
            results = crawler.crawl(observation_id='0781040101',
                                       filters=['UVW1','UVM2'])

            self.assertEqual(len(results['UVW1']), 2)
            self.assertEqual(len(results['UVM2']), 4)
            self.assertFalse('UVW2' in results)
 