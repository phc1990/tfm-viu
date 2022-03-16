import os
import tempfile
import unittest
import src.utils as utils
import test.helper as helper
import tarfile


class TestUtils(unittest.TestCase):
    
    def test_find_in_tar_0_matches(self):
        tar_path = helper.get_test_data_file_path(file_name='tar1.tar')
        
        with tarfile.open(name=tar_path, mode='r') as tar: 
            matches = utils.find_members_in_tar(tar=tar,
                                                regex_pattern='^.*?CRAZY_PATTERN_TO_MATCH$')
        
            self.assertEqual(len(matches),0)
    
    def test_find_in_tar_2_matches(self):
        tar_path = helper.get_test_data_file_path(file_name='tar1.tar')
        
        with tarfile.open(name=tar_path, mode='r') as tar: 
            matches = utils.find_members_in_tar(tar=tar,
                                                regex_pattern='^.*?FSIMAG.*?\.FTZ$')
            
            self.assertEqual(len(matches),2)
        
    def test_extract_tar_members_to_dir(self):
        with tempfile.TemporaryDirectory() as output_dir:
            tar_path = helper.get_test_data_file_path(file_name='tar1.tar')
            
            with tarfile.open(name=tar_path, mode='r') as tar: 
                members = utils.find_members_in_tar(tar=tar,
                                                    regex_pattern='.')
            
                for member in members:
                    utils.extract_tar_member_to_dir(tar=tar,
                                                    member=member,
                                                    output_dir=output_dir)
                    
                self.assertEqual(len(os.listdir(output_dir)), 6)
                    
                
            