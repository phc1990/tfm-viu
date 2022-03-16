# tfm-viu

Trabajo Fin de Master (TFM) for the Valencian International University (VIU).

## Layout

The repository presents the following layout:

```
/
    .gitignore      
    /python             Python3 source code and test data
    /temp               files in this directory will be ignored by Git
```

## `/python`

Written in [Python3](https://www.python.org/downloads/), the main functionailty of this module is to assist the user with by running **(semi)automated observation screening process**:

```
1. Load observations metadata
2. For each observation...

    2.1 Obtain the relevant FITS
    2.2 For each FIT...
        
        2.2.1 For each filter...
    
            2.2.1 Display the images
            2.2.2 Capture the analysis input (e.g. 'detection', 'no detection', etc.)
    
    2.3 Record the analysis input for the observation
```

Several **abstraction layers** have been defined so that the steps above can be carried out by different implementations:

- `src.observation.Repository` loads a set of observation metadata.
- `src.xsa.Crawler` obtains real data for a given observation.
- `src.fits.Interface` used for displaying and analyzing observations images.
- `src.input.Interface` used to capture the analysis input.
- `src.output.Recorder` used to register the analysis results.

### Layout

The `python` module presents the following layout:

```
/configs                configuration templates
/src                    source code
    fits.py             interfaces to display and/or analyze FITS
    input.py            capturing the analysis input
    observation.py      reading and loading observations
    output.py           registering and recording the analysis results
    utils.py            tools and utilities
    xsa.py              obtaining data from XMM-Newton Science Archive (XSA)
/test                   test code
    /data               test data
    test_*.py           test file for module "*"
screening.py            puts together the screening process
README.md               this file
```

### Running the screening

The file `/python/screening.py` will run the screening process. It only requires 1 argument pointing to a configuration file. A template of the configuration file can be found in `/python/configs/screening.ini`.

You can **run the screening using test data as an example** from the repository base directory (i.e. this directory):

```
python3 python/screening.py python/test/data/screening.ini
```

### Style

Both source and test files follow the [PEP 8 Style Guide](https://peps.python.org/pep-0008/#introduction).

### Tests

Test files use the [unittest](https://docs.python.org/3/library/unittest.html) framework.

Tests can be **run from the `/python` directory**:

```
python3 -m unittest discover test/
```
