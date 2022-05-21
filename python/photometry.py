from tabnanny import verbose
import astropy.io.fits as pyf
from trippy import scamp

inputFile = '/home/pau/git/tfm-viu/temp/0303561001/U/P0303561001OMS006SIMAGE0000.FTZ'

with pyf.open(inputFile) as han:
    data = han[0].data
    header = han[0].header
    
scamp.makeParFiles.writeSex('example.sex',
                    minArea=3.,
                    threshold=5.,
                    zpt=27.8,
                    aperture=20.,
                    min_radius=2.0,
                    catalogType='FITS_LDAC',
                    saturate=55000)
scamp.makeParFiles.writeConv()
scamp.makeParFiles.writeParam(numAps=1) #numAps is thenumber of apertures that you want to use. Here we use 1

scamp.runSex('example.sex',
             inputFile,
             options={'CATALOG_NAME':'example.cat'},
             verbose=True)