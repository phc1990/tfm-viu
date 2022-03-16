from typing import List
import src.observation as observation
import src.xsa as xsa
import src.fits as fits
import src.input as input
import src.output as output
import sys
import configparser


class Screening:
    
    def __init__(self,
                 filters: List[str],
                 observation_respository: observation.Repository,
                 xsa_crawler: xsa.Crawler,
                 fits_interface: fits.Interface,
                 input_interface: input.Interface,
                 output_recorder: output.Recorder):
        
        self.filters                = filters
        self.observation_repository = observation_respository
        self.xsa_crawler            = xsa_crawler
        self.fits_interface         = fits_interface
        self.input_interface        = input_interface
        self.output_recorder        = output_recorder
        
    def _iteration(self, filter: str, fits: List[str]) -> input.Input:
        self.fits_interface.close_current_display()
        
        self.input_interface.message('Displaying filter         : ' + filter)
        self.fits_interface.display(fits)
        
        self.input_interface.message('Awaiting input...')
        input = self.input_interface.listen()        
        
        if input == input.REPEAT:
            self.input_interface.message('Repeating filter...       : ' + filter)
            return self._iteration(filter=filter,
                                   fits=fits)
        
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
            self.input_interface.message('Obtaining data...')
            data = self.xsa_crawler.crawl(observation_id=observation.id,
                                          filters=self.filters)
            n = len(data)
            m = 0
            self.input_interface.message('Number of filters         : ' + str(n))
            
            if len(data) > 0:
                for filter,fits in data.items():
                    self.input_interface.message('')
                    input_value = self._iteration(filter, fits)
                    
                    if (input_value == input.Input.CLOSE):
                        close = True
                        break
                    elif input_value == input.Input.DETECTED:
                        m = m+1
                        
                    self.input_interface.message('Registering filter input  : ' + str(input_value))
                    self.output_recorder.record_filter_input(filter, input_value)
            
            if close:
                break
            
            self.input_interface.message('')
            self.input_interface.message('Registering observation   : ' + str(m) + ' detections')
            self.output_recorder.record_observation(filters=self.filters)
            self.input_interface.message('')
        
        if close:
            self.input_interface.message('Closing...')
        else:
            self.input_interface.message('Finished: no more observations to screen.')
        
        self.fits_interface.close_current_display()


def get_field_or_fail(config, field_name: str):
    if field_name not in config:
        raise Exception('Missing field "' + field_name + '"')
    elif config[field_name] == '':
        raise Exception('Empty field "' + field_name + '"')
    
    return config[field_name]


def unrecognised_field_value(field_name: str, field_value: str):
    raise Exception('Unrecognised value "' + field_value + '" for field "' + field_name + '"')
    

def get_boolean_or_fail(config, field_name: str) -> bool:
    value = get_field_or_fail(config, field_name)
    if value == 'TRUE':
        return True
    elif value == 'FALSE':
        return False
    
    unrecognised_field_value(field_name, value)
    

def create_observation_repository(config) -> observation.Repository:
    required = get_field_or_fail(config, 'REQUIRED')
    type = get_field_or_fail(required, 'OBSERVATIONS_REPOSITORY')
    
    if type == 'CSV':
        section = get_field_or_fail(config,'OBSERVATIONS_REPOSITORY_CSV')
        return observation.CsvRepository(csv_path=get_field_or_fail(section,'CSV_FILEPATH'),
                                         ignore_top_n_lines=int(get_field_or_fail(section,'IGNORE_TOP_N_LINES')))
        
    unrecognised_field_value('OBSERVATION_REPOSITORY', type)
    

def create_xsa_crawler(config) -> xsa.Crawler:
    required = get_field_or_fail(config, 'REQUIRED')
    type = get_field_or_fail(required, 'XSA_CRAWLER')
    if type == 'HTTP':
        section = get_field_or_fail(config, 'XSA_CRAWLER_HTTP')
        return xsa.HttpCrawler(download_dir=get_field_or_fail(section, 'DOWNLOAD_DIRECTORY'),
                               base_url=get_field_or_fail(section, 'BASE_URL'),
                               regex_patern=get_field_or_fail(section, 'REGEX'))
        
    unrecognised_field_value('XSA_CRAWLER', type)
    
    
def create_fits_interface(config) -> fits.Interface:
    required = get_field_or_fail(config, 'REQUIRED')
    type = get_field_or_fail(required, 'FITS_INTERFACE')
    if type == 'DS9':
        section = get_field_or_fail(config, 'FITS_INTERFACE_DS9')
        return fits.Ds9Interface(ds9_path=get_field_or_fail(section, 'DS9_BINARY_FILEPATH'),
                                 zoom=get_field_or_fail(section,'ZOOM'),
                                 zscale=get_boolean_or_fail(section, 'ZSCALE'))
        
    unrecognised_field_value('FITS_INTERFACE', type)


def create_input_interface(config) -> input.Interface:
    required = get_field_or_fail(config, 'REQUIRED')
    type = get_field_or_fail(required, 'INPUT_INTERFACE')
    if type == 'STDIO':
        return input.StdIoInterface()
    
    unrecognised_field_value('INPUT_INTERFACE', type)


def create_output_recorder(config) -> output.Recorder:
    required = get_field_or_fail(config, 'REQUIRED')
    type = get_field_or_fail(required, 'OUTPUT_RECORDER')
    
    if type == 'CSV':
        section = get_field_or_fail(config,'OUTPUT_RECORDER_CSV')
        return output.CsvRecorder(csv_path=get_field_or_fail(section, 'CSV_FILEPATH'),
                                  include_headers=get_boolean_or_fail(section, 'INCLUDE_HEADERS'))
        
    unrecognised_field_value('OUTPUT_RECORDER', type)


def create_screening(config) -> Screening:
    required = get_field_or_fail(config, 'REQUIRED')
    filters = get_field_or_fail(required,'FILTERS').split(',')
    return Screening(filters=filters,
                     observation_respository=create_observation_repository(config),
                     xsa_crawler=create_xsa_crawler(config),
                     fits_interface=create_fits_interface(config),
                     input_interface=create_input_interface(config),
                     output_recorder=create_output_recorder(config))
   
   
if __name__ == '__main__':
    
    args = sys.argv
    
    if len(args) < 2:
        raise Exception('Missing .ini configuration file location.')
    elif len(args) > 2:
        raise Exception('Too many arguments.')
    
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config.read(args[1])
    screening = create_screening(config)
    screening.start()
    