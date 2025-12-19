#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
screening.py
INI-driven screening workflow for asteroid candidates.

[INPUT]
CSV_FILEPATH = /path/to/screening_input.csv

[DOWNLOAD]
DOWNLOAD_DIRECTORY = /path/where/XSA_CRAWLER/puts/files
REGEX = .*(?:/|^)({observation_id})(?:/).*?(?:_|-){filter}(?:_|-).*\\.(?:fits|ftz)$

[DS9]
DS9_BINARY_FILEPATH = /Applications/SAOImageDS9.app/Contents/MacOS/ds9
ZOOM   = to fit
ZSCALE = TRUE

[SCREENING_RESULTS]
CSV_FILEPATH    = /path/to/screening_results.csv
INCLUDE_HEADERS = TRUE
"""

from __future__ import annotations
import argparse
import configparser
import csv
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, List
from collections.abc import Sequence

# Make XQuartz available (no XPA required in this build)
os.environ.setdefault('DISPLAY', ':0')

# Column names
OBS_ID_COLS   = ('observation_id', 'obsid', 'observation')
TARGET_COLS   = ('sso_name', 'target_name', 'target')
FILTER_COLS   = ('filter', 'om_filter', 'band')

POS1_RA_COLS  = ('position_1_ra', 'ra_deg_1')
POS1_DEC_COLS = ('position_1_dec', 'dec_deg_1')
POS2_RA_COLS  = ('position_2_ra', 'ra_deg_2')
POS2_DEC_COLS = ('position_2_dec', 'dec_deg_2')

# -----------------------
# Helpers
# -----------------------

def _run(cmd, timeout: float = 20.0) -> Tuple[int, str, str]:
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy(), timeout=timeout)
        return out.returncode, out.stdout, out.stderr
    except subprocess.TimeoutExpired as e:
        return 124, e.stdout or "", e.stderr or "timeout"

def first_present(row: Dict[str, str], names: Iterable[str]) -> Optional[str]:
    for n in names:
        if n in row and str(row[n]).strip():
            return row[n].strip()
    return None

# def read_input_csv(path: Path):
#     with path.open('r', newline='') as f:
#         sample = f.read(4096)
#         f.seek(0)
#         try:
#             dialect = csv.Sniffer().sniff(sample, delimiters=";,")
#             delimiter = dialect.delimiter
#         except csv.Error:
#             delimiter = ';' if ';' in sample and ',' not in sample else ','
#         reader = csv.DictReader(f, delimiter=delimiter, quotechar='"')
#         for row in reader:
#             yield row

def read_input_csv(path: Path) -> tuple[list[dict], list[str]]:
    """
    Read repository CSV into memory and return (rows, fieldnames).

    - Auto-detects delimiter between ';' and ','.
    - Returns a list of dict rows + the list of column names.
    """
    with path.open('r', newline='') as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ';' if ';' in sample and ',' not in sample else ','

        reader = csv.DictReader(f, delimiter=delimiter, quotechar='"')
        rows = list(reader)
        fieldnames = list(reader.fieldnames or ([] if not rows else rows[0].keys()))

    return rows, fieldnames


def parse_row_positions(row: Dict[str, str]) -> Tuple[float, float, float, float]:
    ra1 = first_present(row, POS1_RA_COLS)
    dec1 = first_present(row, POS1_DEC_COLS)
    ra2 = first_present(row, POS2_RA_COLS)
    dec2 = first_present(row, POS2_DEC_COLS)
    if not all([ra1, dec1, ra2, dec2]):
        raise KeyError('Missing RA/Dec columns: need ra_deg_1/dec_deg_1/ra_deg_2/dec_deg_2 (or *_1/_2 equivalents).')
    return float(ra1), float(dec1), float(ra2), float(dec2)

# INI

def _find_ini_key(cfg: configparser.ConfigParser, key: str, sections: Iterable[str]) -> Optional[str]:
    for sec in sections:
        if sec in cfg and key in cfg[sec] and cfg[sec][key].strip():
            return cfg[sec][key].strip()
    for sec in cfg.sections():
        if key in cfg[sec] and cfg[sec][key].strip():
            return cfg[sec][key].strip()
    return None

# def read_ini_settings(ini_path: Path):
#     cfg = configparser.ConfigParser()
#     cfg.read(ini_path)

#     input_csv = _find_ini_key(cfg, 'CSV_FILEPATH', sections=('INPUT',))
#     if not input_csv:
#         raise KeyError('INPUT.CSV_FILEPATH not found in INI.')

#     download_dir = _find_ini_key(cfg, 'DOWNLOAD_DIRECTORY', sections=('DOWNLOAD','XSA_CRAWLER'))
#     if not download_dir:
#         raise KeyError('DOWNLOAD_DIRECTORY not found in INI (in [DOWNLOAD] or [XSA_CRAWLER]).')

#     regex_pat = _find_ini_key(cfg, 'REGEX', sections=('DOWNLOAD','XSA_CRAWLER'))

#     ds9_bin = _find_ini_key(cfg, 'DS9_BINARY_FILEPATH', sections=('DS9',)) or 'ds9'
#     zoom     = _find_ini_key(cfg, 'ZOOM', sections=('DS9',)) or 'to fit'
#     zscale_s = (_find_ini_key(cfg, 'ZSCALE', sections=('DS9',)) or 'TRUE').strip().upper()
#     zscale   = (zscale_s == 'TRUE')

#     screening_csv = _find_ini_key(cfg, 'CSV_FILEPATH', sections=('SCREENING_RESULTS',))
#     if not screening_csv:
#         raise KeyError('SCREENING_RESULTS.CSV_FILEPATH not found in INI.')
#     screening_headers_s = (_find_ini_key(cfg, 'INCLUDE_HEADERS', sections=('SCREENING_RESULTS',)) or 'TRUE').strip().upper()
#     screening_include_headers = (screening_headers_s == 'TRUE')

#     return {
#         'input_csv': Path(input_csv).expanduser().resolve(),
#         'download_dir': Path(download_dir).expanduser().resolve(),
#         'regex': regex_pat,
#         'ds9_bin': ds9_bin,
#         'zoom': zoom,
#         'zscale': zscale,
#         'screening_csv': Path(screening_csv).expanduser().resolve(),
#         'screening_include_headers': screening_include_headers,
#     }

def read_ini_settings(ini_path: Path):
    cfg = configparser.ConfigParser()
    cfg.read(ini_path)

    # INPUT CSV: support both old [INPUT] and new [OBSERVATIONS_REPOSITORY_CSV]
    input_csv = _find_ini_key(
        cfg,
        'CSV_FILEPATH',
        sections=('INPUT', 'OBSERVATIONS_REPOSITORY_CSV'),
    )
    if not input_csv:
        raise KeyError(
            'CSV_FILEPATH not found in INI (expected in [INPUT] or [OBSERVATIONS_REPOSITORY_CSV]).'
        )

    download_dir = _find_ini_key(
        cfg,
        'DOWNLOAD_DIRECTORY',
        sections=('DOWNLOAD', 'XSA_CRAWLER', 'XSA_CRAWLER_HTTP'),
    )
    if not download_dir:
        raise KeyError(
            'DOWNLOAD_DIRECTORY not found in INI (in [DOWNLOAD], [XSA_CRAWLER] or [XSA_CRAWLER_HTTP]).'
        )

    regex_pat = _find_ini_key(
        cfg,
        'REGEX',
        sections=('DOWNLOAD', 'XSA_CRAWLER', 'XSA_CRAWLER_HTTP'),
    )

    ds9_bin = _find_ini_key(cfg, 'DS9_BINARY_FILEPATH', sections=('DS9', 'FITS_INTERFACE_DS9')) or 'ds9'
    zoom     = _find_ini_key(cfg, 'ZOOM',                 sections=('DS9', 'FITS_INTERFACE_DS9')) or 'to fit'
    zscale_s = (_find_ini_key(cfg, 'ZSCALE',              sections=('DS9', 'FITS_INTERFACE_DS9')) or 'TRUE').strip().upper()
    zscale   = (zscale_s == 'TRUE')

    # OUTPUT CSV: support old [SCREENING_RESULTS] and new [OUTPUT_RECORDER_CSV]
    screening_csv = _find_ini_key(
        cfg,
        'CSV_FILEPATH',
        sections=('SCREENING_RESULTS', 'OUTPUT_RECORDER_CSV'),
    )
    if not screening_csv:
        raise KeyError(
            'CSV_FILEPATH not found in INI (expected in [SCREENING_RESULTS] or [OUTPUT_RECORDER_CSV]).'
        )
    screening_headers_s = (
        _find_ini_key(
            cfg,
            'INCLUDE_HEADERS',
            sections=('SCREENING_RESULTS', 'OUTPUT_RECORDER_CSV'),
        )
        or 'TRUE'
    ).strip().upper()
    screening_include_headers = (screening_headers_s == 'TRUE')

    return {
        'input_csv': Path(input_csv).expanduser().resolve(),
        'download_dir': Path(download_dir).expanduser().resolve(),
        'regex': regex_pat,
        'ds9_bin': ds9_bin,
        'zoom': zoom,
        'zscale': zscale,
        'screening_csv': Path(screening_csv).expanduser().resolve(),
        'screening_include_headers': screening_include_headers,
    }


# DS9 zoom arg builder — splits correctly
def _zoom_args(zoom: str) -> List[str]:
    z = (zoom or '').strip()
    if z.lower() == 'to fit':
        return ['-zoom', 'to', 'fit']
    parts = z.split()
    if len(parts) > 1:
        return ['-zoom', *parts]
    return ['-zoom', z]

def _regions_text(ra1: float, dec1: float, ra2: float, dec2: float) -> str:
    # Use a minimal, version-tolerant regions syntax
    return (
        'global color=cyan width=2\n'
        'fk5\n'
        f'circle({ra1},{dec1},10") # text={{pos1}}\n'
        f'circle({ra2},{dec2},10") # text={{pos2}}\n'
    )

# def launch_ds9_with_regions(ds9_bin: str,
#                             fits_path: Path,
#                             zoom: str,
#                             zscale: bool,
#                             ra1: float, dec1: float, ra2: float, dec2: float) -> tuple[tempfile.NamedTemporaryFile, subprocess.Popen]:
#     """
#     Launch DS9 with a temp regions file preloaded. Return (tempfile_handle, proc).
#     Caller is responsible for closing/removing the tempfile.
#     """
#     reg_text = _regions_text(ra1, dec1, ra2, dec2)
#     tf = tempfile.NamedTemporaryFile('w', suffix='.reg', delete=False)
#     tf.write(reg_text)
#     tf.flush()  # ensure content is written

#     env = os.environ.copy()
#     env.setdefault('DISPLAY', ':0')

#     cmd = [ds9_bin, str(fits_path), '-scale', 'zscale' if zscale else 'linear']
#     cmd += _zoom_args(zoom)
#     cmd += ['-regions', 'load', tf.name]

#     proc = subprocess.Popen(cmd, env=env)
#     # tiny delay helps DS9 render before user looks
#     time.sleep(0.4)
#     return tf, proc


# def launch_ds9_with_regions(
#     ds9_bin: str,
#     fits_paths: Path | Sequence[Path],
#     zoom: str,
#     zscale: bool,
#     ra1: float,
#     dec1: float,
#     ra2: float,
#     dec2: float,
# ) -> tuple[tempfile.NamedTemporaryFile, subprocess.Popen]:
#     """
#     Launch DS9 with one or many FITS images and a temp regions file preloaded.
#     Return (tempfile_handle, proc). Caller must close/remove tempfile.
#     """
#     # Normalise to a list of Paths
#     if isinstance(fits_paths, (str, os.PathLike, Path)):
#         fits_list = [Path(fits_paths)]
#     else:
#         fits_list = [Path(p) for p in fits_paths]

#     if not fits_list:
#         raise ValueError("launch_ds9_with_regions called with empty fits_paths.")

#     reg_text = _regions_text(ra1, dec1, ra2, dec2)
#     tf = tempfile.NamedTemporaryFile('w', suffix='.reg', delete=False)
#     tf.write(reg_text)
#     tf.flush()

#     env = os.environ.copy()
#     env.setdefault('DISPLAY', ':0')

#     cmd = [ds9_bin]
#     cmd += [str(p) for p in fits_list]
#     if zscale:
#         cmd += ['-scale', 'zscale']
#     else:
#         cmd += ['-scale', 'linear']
#     cmd += _zoom_args(zoom)
#     cmd += ['-regions', 'load', tf.name]

#     proc = subprocess.Popen(cmd, env=env)
#     time.sleep(0.4)  # small delay so DS9 shows up
#     return tf, proc


def launch_ds9_with_regions(
    ds9_bin: str,
    fits_paths: Path | Sequence[Path],
    zoom: str,
    zscale: bool,
    ra1: float,
    dec1: float,
    ra2: float,
    dec2: float,
) -> tuple[tempfile.NamedTemporaryFile, subprocess.Popen]:
    """
    Launch DS9 with one or many FITS images and a temp regions file preloaded
    on *all* frames. Return (tempfile_handle, proc). Caller must close/remove
    the tempfile.
    """

    # --- normalise to a list of Paths ---
    if isinstance(fits_paths, (str, os.PathLike, Path)):
        fits_list = [Path(fits_paths)]
    else:
        fits_list = [Path(p) for p in fits_paths]

    if not fits_list:
        raise ValueError("launch_ds9_with_regions called with empty fits_paths.")

    # --- build temporary regions file ---
    reg_text = _regions_text(ra1, dec1, ra2, dec2)
    tf = tempfile.NamedTemporaryFile('w', suffix='.reg', delete=False)
    tf.write(reg_text)
    tf.flush()

    env = os.environ.copy()
    env.setdefault('DISPLAY', ':0')

    cmd: list[str] = [ds9_bin]

    # 1) load all images → DS9 creates one frame per image
    cmd += [str(p) for p in fits_list]

    # 2) set scale on current frame, then match + lock across frames
    if zscale:
        # mode zscale + copy to others + keep locked
        cmd += [
            '-scale', 'mode', 'zscale',
            '-scale', 'match',
            '-scale', 'lock', 'yes',
        ]
    else:
        # linear, but still match/lock so all frames behave the same
        cmd += [
            '-scale', 'linear',
            '-scale', 'match',
            '-scale', 'lock', 'yes',
        ]

    # 3) zoom (to fit / numeric etc.)
    cmd += _zoom_args(zoom)

    # 4) load regions on *all* frames
    cmd += ['-regions', 'load', 'all', tf.name]

    proc = subprocess.Popen(cmd, env=env)
    time.sleep(0.4)  # small delay so DS9 has time to render

    return tf, proc

def close_ds9(proc: Optional[subprocess.Popen]):
    """Close DS9 best-effort without XPA."""
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
        except Exception:
            pass

def prompt_user(label: str) -> str:
    while True:
        ans = input(f'{label} Detection? [Y]es / [N]o / [D]ubious: ').strip().upper()
        if ans in ('Y', 'N', 'D'):
            return ans
        print('Please type Y, N, or D.')

# FITS search

def _compile_regex_from_ini(regex_tmpl: str,
                            observation_id: str,
                            filt: str,
                            target_name: str) -> re.Pattern:
    filled = regex_tmpl.format(observation_id=re.escape(str(observation_id)),
                               filter=re.escape(str(filt)),
                               target_name=re.escape(str(target_name)))
    return re.compile(filled, re.IGNORECASE)

def _list_fits_like(obs_dir: Path) -> List[Path]:
    exts = ('*.fits', '*.fit', '*.FITS', '*.FIT', '*.ftz', '*.FTZ', '*.fz', '*.FZ')
    files: List[Path] = []
    for ext in exts:
        files.extend(obs_dir.rglob(ext))
    return files

def find_fits_with_regex(download_root: Path,
                         observation_id: str,
                         filt: str,
                         target_name: str,
                         regex_tmpl: str) -> Optional[Path]:
    obs_dir = download_root / str(observation_id)
    if not obs_dir.exists():
        cands = [c for c in download_root.glob(f'*{observation_id}*') if c.is_dir()]
        if not cands:
            return None
        obs_dir = cands[0]
    files = _list_fits_like(obs_dir)
    if not files:
        return None
    pattern = _compile_regex_from_ini(regex_tmpl, observation_id, filt, target_name)
    matches = [p for p in files if pattern.search(p.as_posix())]
    if not matches:
        return None
    def sort_key(p: Path):
        ext = p.suffix.lower()
        score = 0 if ext in ('.fits', '.fit') else 1
        return (score, len(p.as_posix()))
    matches.sort(key=sort_key)
    return matches[0]

def looks_like_filter_in_name(name: str, filt: str) -> bool:
    n = name.upper()
    f = filt.upper()
    pats = [
        f'_{f}_', f'-{f}-', f'_{f}-', f'-{f}_',
        f'.{f}.', f'.{f}_', f'_{f}.',
        f'OM{f}', f'{f}OM',
        f'_{f}.FITS', f'_{f}.FTZ', f'-{f}.FITS', f'-{f}.FTZ'
    ]
    return any(p in n for p in pats)

def find_fits_fallback(download_root: Path,
                       observation_id: str,
                       filt: str) -> Optional[Path]:
    obs_dir = download_root / str(observation_id)
    if not obs_dir.exists():
        cands = [c for c in download_root.glob(f'*{observation_id}*') if c.is_dir()]
        if not cands:
            return None
        obs_dir = cands[0]
    files = _list_fits_like(obs_dir)
    if not files:
        return None
    matches = [p for p in files if observation_id in p.as_posix() and (looks_like_filter_in_name(p.name, filt) or filt.upper() in p.name.upper())]
    if not matches:
        return files[0] if len(files) == 1 else None
    def sort_key(p: Path):
        ext = p.suffix.lower()
        score = 0 if ext in ('.fits', '.fit') else 1
        return (score, len(p.as_posix()))
    matches.sort(key=sort_key)
    return matches[0]

def find_all_fits_with_regex(
    download_root: Path,
    observation_id: str,
    filt: str,
    target_name: str,
    regex_tmpl: str,
) -> list[Path]:
    """
    Return ALL FITS files under <download_root>/<obsid> matching the regex template.
    Sorted to put "real" FITS before compressed, and shorter paths first.
    """
    obs_dir = download_root / str(observation_id)
    if not obs_dir.exists():
        cands = [c for c in download_root.glob(f'*{observation_id}*') if c.is_dir()]
        if not cands:
            return []
        obs_dir = cands[0]

    files = _list_fits_like(obs_dir)
    if not files:
        return []

    pattern = _compile_regex_from_ini(regex_tmpl, observation_id, filt, target_name)
    matches = [p for p in files if pattern.search(p.as_posix())]
    if not matches:
        return []

    def sort_key(p: Path):
        ext = p.suffix.lower()
        score = 0 if ext in ('.fits', '.fit') else 1
        return (score, len(p.as_posix()))

    matches.sort(key=sort_key)
    return matches


def find_all_fits_fallback(
    download_root: Path,
    observation_id: str,
    filt: str,
) -> list[Path]:
    """
    Fallback: all FITS in <download_root>/<obsid> that look like they belong to a filter.

    Uses heuristics from looks_like_filter_in_name; if nothing matches, returns [].
    """
    obs_dir = download_root / str(observation_id)
    if not obs_dir.exists():
        cands = [c for c in download_root.glob(f'*{observation_id}*') if c.is_dir()]
        if not cands:
            return []
        obs_dir = cands[0]

    files = _list_fits_like(obs_dir)
    if not files:
        return []

    matches = [
        p
        for p in files
        if observation_id in p.as_posix()
        and (looks_like_filter_in_name(p.name, filt) or filt.upper() in p.name.upper())
    ]
    if not matches:
        return []

    def sort_key(p: Path):
        ext = p.suffix.lower()
        score = 0 if ext in ('.fits', '.fit') else 1
        return (score, len(p.as_posix()))

    matches.sort(key=sort_key)
    return matches




# Screening CSV

# Screening CSV (exposure-based)

def ensure_screening_headers(csv_path: Path, include_headers: bool, fieldnames: list[str]):
    """
    Create the output CSV with headers if requested and file does not exist.

    Headers = all input metadata columns + FITS_FILE + DECISION
    """
    if not include_headers:
        return
    if csv_path.exists():
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    full_fields = list(fieldnames) + ['FITS_FILE', 'DECISION']
    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=full_fields)
        w.writeheader()


def append_screening_row(
    csv_path: Path,
    input_fieldnames: list[str],
    row: dict,
    fits_path: Path | None,
    decision: str,
):
    """
    Append one exposure row to the output CSV:

    - All columns from the repository CSV row
    - FITS_FILE : basename of the exposure (or placeholder if None)
    - DECISION  : 'Y', 'D', 'N'
    """
    full_fields = list(input_fieldnames) + ['FITS_FILE', 'DECISION']

    rec = {k: row.get(k, "") for k in input_fieldnames}
    rec['FITS_FILE'] = fits_path.name if fits_path is not None else ''
    rec['DECISION'] = decision

    with csv_path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=full_fields)
        w.writerow(rec)


# def ensure_screening_headers(csv_path: Path, include_headers: bool):
#     if not include_headers:
#         return
#     if not csv_path.exists():
#         csv_path.parent.mkdir(parents=True, exist_ok=True)
#         with csv_path.open('w', newline='') as f:
#             w = csv.writer(f)
#             w.writerow(['observation_id','target_name','filter','fits_filename','decision'])

# def append_screening_row(csv_path: Path,
#                          observation_id: str,
#                          target_name: str,
#                          filt: str,
#                          fits_path: Path,
#                          decision: str):
#     with csv_path.open('a', newline='') as f:
#         w = csv.writer(f)
#         w.writerow([observation_id, target_name, filt, fits_path.name, decision])

# Photometry (launch in PARALLEL on Y/D)

def run_photometry_if_needed(decision: str,
                             fits_path: Path,
                             target_name: str,
                             observation_id: str,
                             pos1_ra: float,
                             pos1_dec: float,
                             pos2_ra: float,
                             pos2_dec: float,
                             filter: str,
                             ini_path: Path):
    if decision not in ('Y', 'D'):
        return None
    photometry_py = Path(__file__).with_name('expo_photometry.py')
    if not photometry_py.exists():
        print(f'[WARN] expo_photometry.py not found next to screening.py; skipping photometry.', file=sys.stderr)
        return None
    cmd = [
        sys.executable, str(photometry_py),
        '--fits', str(fits_path),
        '--target-name', str(target_name),
        '--observation-id', str(observation_id),
        '--pos1-ra', str(pos1_ra), '--pos1-dec', str(pos1_dec),
        '--pos2-ra', str(pos2_ra), '--pos2-dec', str(pos2_dec),
        '--filter',str(filter),
        '--ini', str(ini_path)
    ]
    try:
        print(f'[INFO] Launching photometry (parallel): {" ".join(shlex.quote(x) for x in cmd)}')
        return subprocess.Popen(cmd, env=os.environ.copy())
    except Exception as e:
        print(f'[WARN] photometry launch failed for {fits_path}: {e}', file=sys.stderr)
        return None

# add near the top-level helpers
def _s(x):
    # safe strip/normalize for optional CSV fields
    if x is None:
        return ""
    x = str(x)
    return x.strip()

def _s_or_none(x):
    if x is None:
        return None
    x = str(x).strip()
    return x if x else None

# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ini', required=True, help='Path to screening.ini (contains INPUT/SCREENING_RESULTS/DOWNLOAD/DS9).')
    args = ap.parse_args()

    ini_path  = Path(args.ini).expanduser().resolve()
    if not ini_path.exists():
        print(f'[ERROR] INI not found: {ini_path}', file=sys.stderr)
        sys.exit(2)

    try:
        settings = read_ini_settings(ini_path)
    except Exception as e:
        print(f'[ERROR] {e}', file=sys.stderr)
        sys.exit(2)

    input_csv: Path      = settings['input_csv']
    download_root: Path  = settings['download_dir']
    regex_tmpl: Optional[str] = settings['regex']
    ds9_bin: str         = settings['ds9_bin']
    zoom: str            = settings['zoom']
    zscale: bool         = settings['zscale']
    screening_csv: Path  = settings['screening_csv']
    screening_headers: bool = settings['screening_include_headers']

    # if not input_csv.exists():
    #     print(f'[ERROR] Input CSV not found: {input_csv}', file=sys.stderr)
    #     sys.exit(2)
    
        # --- load repository CSV into memory ---
    rows, input_fieldnames = read_input_csv(input_csv)

    if not input_csv.exists():
        print(f'[ERROR] Input CSV not found: {input_csv}', file=sys.stderr)
        sys.exit(2)

    # Prepare output headers using the same metadata columns
    ensure_screening_headers(screening_csv, screening_headers, input_fieldnames)

    from src.screening.xsa import HttpCurlCrawler

    crawler = HttpCurlCrawler(
        download_dir=download_root,
        base_url="https://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio",
        regex_patern="^.*?SIMAG.*?\.FTZ$"
    )

    try:
        for row in rows:
            try:
                observation_id = first_present(row, OBS_ID_COLS)
                target_name    = first_present(row, TARGET_COLS) or 'unknown'
                filt           = first_present(row, FILTER_COLS)

                if not observation_id:
                    print('[WARN] Row missing observation_id; skipping.', file=sys.stderr)
                    continue
                if not filt:
                    print(f'[WARN] Row missing filter for obs={observation_id}; skipping.', file=sys.stderr)
                    continue

                ra1, dec1, ra2, dec2 = parse_row_positions(row)

                # --- find ALL exposures for this observation/filter ---
                fits_paths: list[Path] = []
                if regex_tmpl:
                    fits_paths = find_all_fits_with_regex(
                        download_root,
                        observation_id,
                        filt,
                        target_name,
                        regex_tmpl,
                    )

                # if not fits_paths:
                #     fits_paths = find_all_fits_fallback(download_root, observation_id, filt)
                import src.screening.observation as observation

                if not fits_paths:
                    crawler.crawl(
                        observation=observation.Observation(
                            id=observation_id,
                            object="ASDF",
                            ra1=ra1,
                            dec1=dec1,
                            ra2=ra2,
                            dec2=dec2,
                            filters=[filt],
                        )
                    )

                fits_paths = find_all_fits_with_regex(
                        download_root,
                        observation_id,
                        filt,
                        target_name,
                        regex_tmpl,
                    )
                
                if not fits_paths:
                    print(
                        f'[WARN] No FITS found under {download_root}/* for '
                        f'obs={observation_id}, filter={filt}; marking decision=N.'
                    )
                    append_screening_row(
                        screening_csv,
                        input_fieldnames,
                        row,
                        fits_path=None,
                        decision='N',
                    )
                    continue

                label = f'{target_name} | obs={observation_id} | filter={filt}'
                print(f'\n=== {label} ===')
                print('[INFO] Frames loaded:')
                for i, p in enumerate(fits_paths, start=1):
                    print(f'  [{i}] {p.name}')

                # --- open DS9 with ALL frames + regions on pos1/pos2 ---
                tf, ds9_proc = launch_ds9_with_regions(
                    ds9_bin,
                    fits_paths,
                    zoom,
                    zscale,
                    ra1,
                    dec1,
                    ra2,
                    dec2,
                )

                # ---- interactive detection loop for this observation ----
                while True:
                    decision = prompt_user(label)  # Y / N / D

                    # Close DS9 after the first definitive decision
                    close_ds9(ds9_proc)
                    try:
                        tf.close()
                        Path(tf.name).unlink(missing_ok=True)
                    except Exception:
                        pass

                    if decision in ('Y', 'D'):
                        # Ask which exposures have the detection
                        while True:
                            raw = input(
                                'Which exposure frame(s) show the detection? '
                                '(comma-separated indices, e.g. 1,3,4): '
                            ).strip()
                            if not raw:
                                print('Please provide at least one index (e.g. 1 or 1,3).')
                                continue
                            try:
                                indices = sorted(
                                    {int(x.strip()) for x in raw.split(',') if x.strip()}
                                )
                            except ValueError:
                                print('Invalid list of indices; please use numbers like 1,2,3.')
                                continue

                            bad = [i for i in indices if i < 1 or i > len(fits_paths)]
                            if bad:
                                print(
                                    f'Invalid indices {bad}; valid range is 1..{len(fits_paths)}.'
                                )
                                continue

                            # Record each selected exposure as its own row
                            for i in indices:
                                fits_path = fits_paths[i - 1]
                                append_screening_row(
                                    screening_csv,
                                    input_fieldnames,
                                    row,
                                    fits_path=fits_path,
                                    decision=decision,
                                )

                                # Launch photometry in parallel for each Y/D exposure
                                _ = run_photometry_if_needed(
                                    decision=decision,
                                    fits_path=fits_path,
                                    target_name=target_name,
                                    observation_id=str(observation_id),
                                    pos1_ra=ra1,
                                    pos1_dec=dec1,
                                    pos2_ra=ra2,
                                    pos2_dec=dec2,
                                    filter=str(filt),
                                    ini_path=ini_path,
                                )

                            break  # indices accepted

                        # After a Y/D detection, ask if there is another independent detection
                        more = input(
                            'Any other detection in this observation? '
                            '[Y]es / [D]ubious / [N]o (default N): '
                        ).strip().upper() or 'N'

                        if more in ('Y', 'D'):
                            # Re-open DS9 so user can inspect frames again
                            tf, ds9_proc = launch_ds9_with_regions(
                                ds9_bin,
                                fits_paths,
                                zoom,
                                zscale,
                                ra1,
                                dec1,
                                ra2,
                                dec2,
                            )
                            decision = more
                            # and loop back to the top of the while True (decision loop)
                            continue

                        # No more detections in this observation → go to next observation row
                        break

                    else:
                        # decision == 'N' (or anything not Y/D, since prompt_user restricts)
                        append_screening_row(
                            screening_csv,
                            input_fieldnames,
                            row,
                            fits_path=None,
                            decision='N',
                        )
                        break  # go to next observation row

            # except KeyboardInterrupt:
            #     print('\n[INFO] Aborted by user.')
            #     break
            except Exception as e:
                print(f'[ERROR] Failed on row: {e}', file=sys.stderr)
                raise e
    except KeyboardInterrupt:
        # Global Ctrl+C: keep everything that was already written to CSV
        print('\n[INFO] Aborted by user.')
        print(f'[INFO] Partial screening results preserved in: {screening_csv}')

    else:
        # Only printed if the loop finishes normally (no Ctrl+C)
        print('\n[INFO] Screening completed.')
        print(f'[INFO] Screening results written to: {screening_csv}')

    # ensure_screening_headers(screening_csv, screening_headers)

    # for row in read_input_csv(input_csv):
    #     try:
    #         observation_id = first_present(row, OBS_ID_COLS)
    #         target_name    = first_present(row, TARGET_COLS) or 'unknown'
    #         filt           = first_present(row, FILTER_COLS)

    #         if not observation_id:
    #             print('[WARN] Row missing observation_id; skipping.', file=sys.stderr)
    #             continue
    #         if not filt:
    #             print(f'[WARN] Row missing filter for obs={observation_id}; skipping.', file=sys.stderr)
    #             continue

    #         ra1, dec1, ra2, dec2 = parse_row_positions(row)

    #         # Find FITS path
    #         fits_path: Optional[Path] = None
    #         if regex_tmpl:
    #             fits_path = find_fits_with_regex(download_root, observation_id, filt, target_name, regex_tmpl)
    #         if not fits_path:
    #             fits_path = find_fits_fallback(download_root, observation_id, filt)
    #         if not fits_path:
    #             print(f'[WARN] FITS not found under {download_root}/** for obs={observation_id}, filter={filt}; marking decision=N.')
    #             append_screening_row(screening_csv, str(observation_id), target_name, filt, Path(f'{observation_id}_{filt}.fits'), 'N')
    #             continue

    #         label = f'{target_name} | obs={observation_id} | filter={filt}'
    #         print(f'\n=== {label} ===')
    #         print(f'FITS: {fits_path}')

    #         # Launch DS9 ONCE with regions preloaded (no XPA needed)
    #         tf, ds9_proc = launch_ds9_with_regions(ds9_bin, fits_path, zoom, zscale, ra1, dec1, ra2, dec2)

    #         # Prompt
    #         decision = prompt_user(label)

    #         # Close DS9 immediately after answer
    #         close_ds9(ds9_proc)
    #         try:
    #             tf.close()
    #             Path(tf.name).unlink(missing_ok=True)
    #         except Exception:
    #             pass

    #         # Record screening
    #         append_screening_row(screening_csv, str(observation_id), target_name, filt, fits_path, decision)

    #         # Photometry on Y/D — launch in PARALLEL
    #         _ = run_photometry_if_needed(
    #             decision=decision,
    #             fits_path=fits_path,
    #             target_name=target_name,
    #             observation_id=str(observation_id),
    #             pos1_ra=ra1, pos1_dec=dec1,
    #             pos2_ra=ra2, pos2_dec=dec2,
    #             filter=str(filt),
    #             ini_path=ini_path
    #         )

    #         # Continue to next row immediately

    #     except KeyboardInterrupt:
    #         print('\n[INFO] Aborted by user.')
    #         break
    #     except Exception as e:
    #         print(f'[ERROR] Failed on row: {e}', file=sys.stderr)
    #         continue



    print('\n[INFO] Screening completed.')

if __name__ == '__main__':
    main()
