from fileinput import filename
import sys

from src.screening import xsa, utils
            
from astroquery.esa.xmm_newton import XMMNewton

if __name__ == '__main__':

    download_dir = sys.argv[1]
    observation_id = sys.argv[2]
    filter = sys.argv[3]
    
    crawler = xsa.HttpPythonRequestsCrawler(  download_dir=download_dir,
                                              base_url='http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio',
                                              regex_patern='^.*?FSIMAG.*?\.FTZ$')
    
    with open(utils.build_path(download_dir, observation_id + filter + '.tar'), mode = 'w+b') as file: 
        XMMNewton.download_data(observation_id=observation_id,
                                filename=file.name, instname='OM', name='OBSMLI', level='PPS', extension='FTZ')
          