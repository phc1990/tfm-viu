#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from configparser import ConfigParser
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


from common import OBS_ID_COLS, TARGET_COLS, FILTER_COLS, DECISION_COLS, FITS_FILE_COLS, DETECTION_VALS
from common import read_csv, extract_row_value, extract_matching_rows, find_fits_files 
from action_download import action_download
from action_screening import action_screening
from action_photometry import action_photometry

# Make XQuartz available (no XPA required in this build)
os.environ.setdefault('DISPLAY', ':0')


def _find_next_action(
    input_filepath: str,
    download_directory: str,
    screening_filepath: str,
    photometry_filepath: str,
    skip_screening: bool = False,
    skip_photometry: bool = False,
) -> tuple[str, Any]:
    if not os.path.exists(input_filepath):
        raise FileNotFoundError(f"CSV file not found: {input_filepath}")

    _, input_rows = read_csv(
        filepath=input_filepath,
        raise_file_not_found=True,
    )

    for input_row in input_rows:
        observation_id: str = extract_row_value(
            row=input_row,
            columns=OBS_ID_COLS,
        )
    
        if not find_fits_files(
            folder=download_directory,
            observation_id=observation_id,
            filter=extract_row_value(
                row=input_row,
                columns=FILTER_COLS,
            )
        ):           
            print(f"No FITS files found for Observation {observation_id}, starting download...")
            return ("Download", input_row)

        # Filter by observation id
        matching_screening_rows: list[dict] = extract_matching_rows(
            filepath_or_rows=screening_filepath,
            columns=OBS_ID_COLS,
            value=observation_id
        )

        target_name: str = extract_row_value(
            row=input_row,
            columns=TARGET_COLS,
        )

        # Filter by target name
        matching_screening_rows = extract_matching_rows(
            filepath_or_rows=matching_screening_rows,
            columns=TARGET_COLS,
            value=target_name,
        )

        # If there are no screening rows, it needs to be screened
        if len(matching_screening_rows) == 0 and not skip_screening:
            print(f"No screenings found for Observation {observation_id}, starting screening...")
            return ("Screening", input_row)
        
        detection_rows: list[dict] = []
        for screening_row in matching_screening_rows:
            if extract_row_value(
                row=screening_row,
                columns=DECISION_COLS,
            ) in DETECTION_VALS:
                detection_rows.append(screening_row)

        # If there are no detections, we forget about photometry
        if not detection_rows:
            continue

        # Check if if photometry took place
        for screening_row in detection_rows:
            fits_file: str = extract_row_value(
                row=screening_row,
                columns=FITS_FILE_COLS,
            )

            matching_photometry_rows: list[dict] = extract_matching_rows(
                filepath_or_rows=photometry_filepath,
                columns=FITS_FILE_COLS,
                value=fits_file,
            )

            # If there was no photometry:
            if not matching_photometry_rows and not skip_photometry:
                print(f"No photometry found for FITS {fits_file}, starting photometry...")
                return ("Photometry", screening_row)
            
            if len(matching_photometry_rows) > 1:
                raise RuntimeError(f"More than one photometry row for {fits_file}")
    
    print("No further actions :)")
    return None, None


def main(
    config: ConfigParser
) -> None:
    while True:
        next_action, details = _find_next_action(
            input_filepath=config['INPUT']['FILEPATH'],
            download_directory=config['INPUT']['DOWNLOAD_DIRECTORY'],
            screening_filepath=config['SCREENING']['FILEPATH'],
            photometry_filepath=config['PHOTOMETRY']['FILEPATH'],
            skip_screening=config.getboolean('SCREENING', 'SKIP'),
            skip_photometry=config.getboolean('PHOTOMETRY', 'SKIP'),
        )

        # No next actions, we're done
        if next_action is None:
            break

        elif next_action == 'Download':
            action_download(
                config=config,
                input_row=details,
            )

        elif next_action == 'Screening':
            action_screening(
                config=config,
                input_row=details,
            )

        elif next_action == 'Photometry':
            action_photometry(
                config=config,
                screening_row=details,
            )

        else:
            raise RuntimeError(f"Unrecognized action: {next_action}")


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


if __name__ == '__main__':
    config: ConfigParser = ConfigParser()
    config.read(sys.argv[1])
    
    main(config=config)
