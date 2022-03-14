# tfm-viu

Trabajo Fin de Master (TFM) for the Valencian International University (VIU).

## Layout

The repository presents the following layout:

```
/
    .gitignore      
    python          Python3 source code and test data
```

## `/python`

The `/python` module contains [Python3](https://www.python.org/downloads/) source code and test data:

```
/src                source code
/test               test code
    /data           test data
```

The main functionalities of the module are:
- Downloading data from [XMM-Newton Science Archive (XSA)](http://nxsa.esac.esa.int/nxsa-web/#home).
- Spawning user interfaces to view [Flexible Image Transport System (FITS)](https://fits.gsfc.nasa.gov/) files.

### Style

Both source and test files follow the [PEP 8 Style Guide](https://peps.python.org/pep-0008/#introduction).

### Tests

Test files use the [unittest](https://docs.python.org/3/library/unittest.html) framework.

Tests can be run from the `/python` directory:

```
python3 -m unittest discover test/
```


