import unittest
import src.observation as observation
import test.helper as helper


class TestObservation(unittest.TestCase):
    
    def test_constructor(self):
        o = observation.Observation(id      = 'id',
                                    object  = 'object',
                                    ra1     = 1.0,
                                    dec1    = -2.0,
                                    ra2     = 0.0,
                                    dec2    = 123)
        
        self.assertEqual(o.id, 'id')
        self.assertEqual(o.object, 'object')
        self.assertEqual(o.ra1, 1.0)
        self.assertEqual(o.dec1, -2.0)
        self.assertEqual(o.ra2, 0.0)
        self.assertEqual(o.dec2, 123.0)


class TestCsvRepository(unittest.TestCase):
    
    def test_load_no_headers(self):
        repo = observation.CsvRepository(csv_path=helper.get_test_data_file_path('csv1.csv'),
                                         ignore_n_top_lines=0)
        
        iter = repo.get_iter()
        
        o1 = next(iter)
        self.assertEqual(o1.id, 'Q1')
        self.assertEqual(o1.object, 'A A')
        self.assertEqual(o1.ra1, 3.5)
        self.assertEqual(o1.dec1, -7.0)
        self.assertEqual(o1.ra2, 8.0)
        self.assertEqual(o1.dec2, 0.0)
        
        o2 = next(iter)
        self.assertEqual(o2.id, 'Z1')
        self.assertEqual(o2.object, 'b')
        self.assertEqual(o2.ra1, -4.2)
        self.assertEqual(o2.dec1, -7.0)
        self.assertEqual(o2.ra2, 0.0)
        self.assertEqual(o2.dec2, 1.0)
        
        o3 = next(iter, None)
        self.assertIsNone(o3)
    
    def test_load_ignore_headers(self):
        repo = observation.CsvRepository(csv_path=helper.get_test_data_file_path('csv2.csv'),
                                         ignore_n_top_lines=1)
        
        iter = repo.get_iter()
        
        o1 = next(iter)
        self.assertEqual(o1.id, 'Q1')
        self.assertEqual(o1.object, 'A A')
        self.assertEqual(o1.ra1, 3.5)
        self.assertEqual(o1.dec1, -7.0)
        self.assertEqual(o1.ra2, 8.0)
        self.assertEqual(o1.dec2, 0.0)
        
        o2 = next(iter)
        self.assertEqual(o2.id, 'Z1')
        self.assertEqual(o2.object, 'b')
        self.assertEqual(o2.ra1, -4.2)
        self.assertEqual(o2.dec1, -7.0)
        self.assertEqual(o2.ra2, 0.0)
        self.assertEqual(o2.dec2, 1.0)
        
        o3 = next(iter, None)
        self.assertIsNone(o3)