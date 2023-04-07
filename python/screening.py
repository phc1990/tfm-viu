from typing import List
import src.screening.observation as observation
import src.screening.xsa as xsa
import src.screening.fits as fits
import src.screening.input as input
import src.screening.output as output
import sys
import configparser
from src.config import Config

class Screening:
    
    def __init__(self,
                 observation_respository: observation.Repository,
                 xsa_crawler: xsa.Crawler,
                 fits_interface: fits.Interface,
                 input_interface: input.Interface,
                 output_recorder: output.Recorder):
        
        self.observation_repository = observation_respository
        self.xsa_crawler            = xsa_crawler
        self.fits_interface         = fits_interface
        self.input_interface        = input_interface
        self.output_recorder        = output_recorder
        
    def _iteration(self, filter: str, fits: List[str], observation: observation.Observation) -> input.Input:
        self.fits_interface.close_current_display()
        
        self.input_interface.message('Displaying filter         : ' + filter)
        self.fits_interface.display(fits, observation)
        
        self.input_interface.message('Awaiting input...')
        input = self.input_interface.listen()        
        
        if input == input.REPEAT:
            self.input_interface.message('Repeating filter...       : ' + filter)
            return self._iteration(filter=filter,
                                   fits=fits,
                                   observation=observation)
        
        return input
    
    def start(self):
        """Starts the screening process."""
        self.input_interface.message('Starting screening...')
        close = False
        
        for observation in self.observation_repository.get_iter():
            self.output_recorder.prepare_observation_record(observation=observation)
            self.input_interface.message('**************************')
            self.input_interface.message('Observation ID            : ' + observation.id)
            self.input_interface.message('Potential object          : ' + observation.object)
            self.input_interface.message('Filters                   : ' + str(observation.filters))
            self.input_interface.message('Obtaining data...')
            data = self.xsa_crawler.crawl(observation=observation)

            number_of_detections = 0

            if len(data) > 0:
                for filter,fits in data.items():
                    self.input_interface.message('')
                    input_value = self._iteration(filter, fits, observation)
                    
                    if (input_value == input.Input.CLOSE):
                        close = True
                        break
                    elif input_value == input.Input.DETECTED:
                        number_of_detections = number_of_detections+1
                        
                    self.input_interface.message('Registering filter input  : ' + str(input_value))
                    self.output_recorder.record_filter_input(filter, input_value)
            
            if close:
                break
            
            self.input_interface.message('')
            self.input_interface.message('Registering observation   : ' + str(number_of_detections) + ' detections')
            self.output_recorder.record_observation()
            self.input_interface.message('')
        
        if close:
            self.input_interface.message('Closing...')
        else:
            self.input_interface.message('Finished: no more observations to screen.')
        
        self.fits_interface.close_current_display()
    

def create_observation_repository(config) -> observation.Repository:
    required = config.child('REQUIRED')
    type = required.get_field('OBSERVATIONS_REPOSITORY')
    
    if type == 'CSV':
        section = config.child('OBSERVATIONS_REPOSITORY_CSV')
        return observation.CsvRepository(csv_path           = section.get_field('CSV_FILEPATH'),
                                         ignore_top_n_lines = section.get_int('IGNORE_TOP_N_LINES'))
            

def create_xsa_crawler(config) -> xsa.Crawler:
    required = config.child('REQUIRED')
    type = required.get_field('XSA_CRAWLER')
    if type == 'HTTP_PYTHON':
        section = config.child('XSA_CRAWLER_HTTP')
        return xsa.HttpPythonRequestsCrawler(download_dir   = section.get_field('DOWNLOAD_DIRECTORY'),
                                             base_url       = section.get_field('BASE_URL'),
                                             regex_patern   = section.get_field('REGEX'))
    elif type == 'HTTP_CURL':
        section = config.child('XSA_CRAWLER_HTTP')
        return xsa.HttpCurlCrawler(download_dir     = section.get_field('DOWNLOAD_DIRECTORY'),
                                   base_url         = section.get_field('BASE_URL'),
                                   regex_patern     = section.get_field('REGEX'))
    
    
def create_fits_interface(config) -> fits.Interface:
    required = config.child('REQUIRED')
    type = required.get_field('FITS_INTERFACE')
    if type == 'DS9':
        section = config.child('FITS_INTERFACE_DS9')
        return fits.Ds9Interface(ds9_path   = section.get_field('DS9_BINARY_FILEPATH'),
                                 zoom       = section.get_field('ZOOM'),
                                 zscale     = section.get_boolean('ZSCALE'))
        

def create_input_interface(config) -> input.Interface:
    required = config.child('REQUIRED')
    type = required.get_field('INPUT_INTERFACE')
    if type == 'STDIO':
        return input.StdIoInterface()


def create_output_recorder(config) -> output.Recorder:
    required = config.child('REQUIRED')
    type = required.get_field('OUTPUT_RECORDER')
    
    if type == 'CSV':
        section = config.child('OUTPUT_RECORDER_CSV')
        return output.CsvRecorder(csv_path          = section.get_field('CSV_FILEPATH'),
                                  include_headers   = section.get_boolean('INCLUDE_HEADERS'))


def create_screening(config) -> Screening:
    return Screening(observation_respository    = create_observation_repository(config),
                     xsa_crawler                = create_xsa_crawler(config),
                     fits_interface             = create_fits_interface(config),
                     input_interface            = create_input_interface(config),
                     output_recorder            = create_output_recorder(config))
   
   
if __name__ == '__main__':
    
    args = sys.argv
    
    if len(args) < 2:
        raise Exception('Missing .ini configuration file location.')
    elif len(args) > 2:
        raise Exception('Too many arguments.')
    
    data = configparser.ConfigParser(inline_comment_prefixes='#')
    data.read(args[1])
    config = Config(data=data)
    screening = create_screening(config)
    screening.start()
    