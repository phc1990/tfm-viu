#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from configparser import ConfigParser
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


from common import OBS_ID_COLS, FILTER_COLS, DECISION_COLS, FITS_FILE_COLS, DETECTION_VALS
from common import read_csv, extract_row_value, extract_matching_rows, find_fits_files 
from action_download import action_download
from action_screening import action_screening

# Make XQuartz available (no XPA required in this build)
os.environ.setdefault('DISPLAY', ':0')


def find_next_action(
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

        matching_screening_rows: list[dict] = extract_matching_rows(
            filepath=screening_filepath,
            columns=OBS_ID_COLS,
            value=observation_id
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
                filepath=photometry_filepath,
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
        next_action, details = find_next_action(
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
            print(f"Simulating Photometry with detaisl {details}")

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


# Main

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument('--ini', required=True, help='Path to screening.ini (contains INPUT/SCREENING_RESULTS/DOWNLOAD/DS9).')
#     args = ap.parse_args()

#     ini_path  = Path(args.ini).expanduser().resolve()
#     if not ini_path.exists():
#         print(f'[ERROR] INI not found: {ini_path}', file=sys.stderr)
#         sys.exit(2)

#     try:
#         settings = read_ini_settings(ini_path)
#     except Exception as e:
#         print(f'[ERROR] {e}', file=sys.stderr)
#         sys.exit(2)

#     input_csv: Path      = settings['input_csv']
#     download_root: Path  = settings['download_dir']
#     regex_tmpl: Optional[str] = settings['regex']
#     ds9_bin: str         = settings['ds9_bin']
#     zoom: str            = settings['zoom']
#     zscale: bool         = settings['zscale']
#     screening_csv: Path  = settings['screening_csv']
#     screening_headers: bool = settings['screening_include_headers']

#     # if not input_csv.exists():
#     #     print(f'[ERROR] Input CSV not found: {input_csv}', file=sys.stderr)
#     #     sys.exit(2)
    
#         # --- load repository CSV into memory ---
#     rows, input_fieldnames = read_input_csv(input_csv)

#     if not input_csv.exists():
#         print(f'[ERROR] Input CSV not found: {input_csv}', file=sys.stderr)
#         sys.exit(2)

#     try:
#         try:
#             observation_id = first_present(row, OBS_ID_COLS)
#             target_name    = first_present(row, TARGET_COLS) or 'unknown'
#             filt           = first_present(row, FILTER_COLS)

#             if not observation_id:
#                 print('[WARN] Row missing observation_id; skipping.', file=sys.stderr)
#                 continue
#             if not filt:
#                 print(f'[WARN] Row missing filter for obs={observation_id}; skipping.', file=sys.stderr)
#                 continue

#             ra1, dec1, ra2, dec2 = parse_row_positions(row)

#             # --- find ALL exposures for this observation/filter ---
#             fits_paths: list[Path] = []
#             if regex_tmpl:
#                 fits_paths = find_all_fits_with_regex(
#                     download_root,
#                     observation_id,
#                     filt,
#                     target_name,
#                     regex_tmpl,
#                 )

#             # if not fits_paths:
#             #     fits_paths = find_all_fits_fallback(download_root, observation_id, filt)
#             import src.screening.observation as observation

#             if not fits_paths:
#                 crawler.crawl(
#                     observation=observation.Observation(
#                         id=observation_id,
#                         object="ASDF",
#                         ra1=ra1,
#                         dec1=dec1,
#                         ra2=ra2,
#                         dec2=dec2,
#                         filters=[filt],
#                     )
#                 )

#             fits_paths = find_all_fits_with_regex(
#                     download_root,
#                     observation_id,
#                     filt,
#                     target_name,
#                     regex_tmpl,
#                 )
            
#             if not fits_paths:
#                 print(
#                     f'[WARN] No FITS found under {download_root}/* for '
#                     f'obs={observation_id}, filter={filt}; marking decision=N.'
#                 )
#                 append_screening_row(
#                     screening_csv,
#                     input_fieldnames,
#                     row,
#                     fits_path=None,
#                     decision='N',
#                 )
#                 continue

#             label = f'{target_name} | obs={observation_id} | filter={filt}'
#             print(f'\n=== {label} ===')
#             print('[INFO] Frames loaded:')
#             for i, p in enumerate(fits_paths, start=1):
#                 print(f'  [{i}] {p.name}')

#             # --- open DS9 with ALL frames + regions on pos1/pos2 ---
#             tf, ds9_proc = launch_ds9_with_regions(
#                 ds9_bin,
#                 fits_paths,
#                 zoom,
#                 zscale,
#                 ra1,
#                 dec1,
#                 ra2,
#                 dec2,
#             )

#             # ---- interactive detection loop for this observation ----
#             while True:
#                 decision = prompt_user(label)  # Y / N / D

#                 # Close DS9 after the first definitive decision
#                 close_ds9(ds9_proc)
#                 try:
#                     tf.close()
#                     Path(tf.name).unlink(missing_ok=True)
#                 except Exception:
#                     pass

#                 if decision in ('Y', 'D'):
#                     # Ask which exposures have the detection
#                     while True:
#                         raw = input(
#                             'Which exposure frame(s) show the detection? '
#                             '(comma-separated indices, e.g. 1,3,4): '
#                         ).strip()
#                         if not raw:
#                             print('Please provide at least one index (e.g. 1 or 1,3).')
#                             continue
#                         try:
#                             indices = sorted(
#                                 {int(x.strip()) for x in raw.split(',') if x.strip()}
#                             )
#                         except ValueError:
#                             print('Invalid list of indices; please use numbers like 1,2,3.')
#                             continue

#                         bad = [i for i in indices if i < 1 or i > len(fits_paths)]
#                         if bad:
#                             print(
#                                 f'Invalid indices {bad}; valid range is 1..{len(fits_paths)}.'
#                             )
#                             continue

#                         # Record each selected exposure as its own row
#                         for i in indices:
#                             fits_path = fits_paths[i - 1]
#                             append_screening_row(
#                                 screening_csv,
#                                 input_fieldnames,
#                                 row,
#                                 fits_path=fits_path,
#                                 decision=decision,
#                             )

#                             # Launch photometry in parallel for each Y/D exposure
#                             _ = run_photometry_if_needed(
#                                 decision=decision,
#                                 fits_path=fits_path,
#                                 target_name=target_name,
#                                 observation_id=str(observation_id),
#                                 pos1_ra=ra1,
#                                 pos1_dec=dec1,
#                                 pos2_ra=ra2,
#                                 pos2_dec=dec2,
#                                 filter=str(filt),
#                                 ini_path=ini_path,
#                             )

#                         break  # indices accepted

#                     # After a Y/D detection, ask if there is another independent detection
#                     more = input(
#                         'Any other detection in this observation? '
#                         '[Y]es / [D]ubious / [N]o (default N): '
#                     ).strip().upper() or 'N'

#                     if more in ('Y', 'D'):
#                         # Re-open DS9 so user can inspect frames again
#                         tf, ds9_proc = launch_ds9_with_regions(
#                             ds9_bin,
#                             fits_paths,
#                             zoom,
#                             zscale,
#                             ra1,
#                             dec1,
#                             ra2,
#                             dec2,
#                         )
#                         decision = more
#                         # and loop back to the top of the while True (decision loop)
#                         continue

#                     # No more detections in this observation → go to next observation row
#                     break

#                 else:
#                     # decision == 'N' (or anything not Y/D, since prompt_user restricts)
#                     append_screening_row(
#                         screening_csv,
#                         input_fieldnames,
#                         row,
#                         fits_path=None,
#                         decision='N',
#                     )
#                     break  # go to next observation row

#         # except KeyboardInterrupt:
#         #     print('\n[INFO] Aborted by user.')
#         #     break
#         except Exception as e:
#             print(f'[ERROR] Failed on row: {e}', file=sys.stderr)
#             raise e
#     except KeyboardInterrupt:
#         # Global Ctrl+C: keep everything that was already written to CSV
#         print('\n[INFO] Aborted by user.')
#         print(f'[INFO] Partial screening results preserved in: {screening_csv}')

#     else:
#         # Only printed if the loop finishes normally (no Ctrl+C)
#         print('\n[INFO] Screening completed.')
#         print(f'[INFO] Screening results written to: {screening_csv}')

#     ensure_screening_headers(screening_csv, screening_headers)

#     for row in read_input_csv(input_csv):
#         try:
#             observation_id = first_present(row, OBS_ID_COLS)
#             target_name    = first_present(row, TARGET_COLS) or 'unknown'
#             filt           = first_present(row, FILTER_COLS)

#             if not observation_id:
#                 print('[WARN] Row missing observation_id; skipping.', file=sys.stderr)
#                 continue
#             if not filt:
#                 print(f'[WARN] Row missing filter for obs={observation_id}; skipping.', file=sys.stderr)
#                 continue

#             ra1, dec1, ra2, dec2 = parse_row_positions(row)

#             # Find FITS path
#             fits_path: Optional[Path] = None
#             if regex_tmpl:
#                 fits_path = find_fits_with_regex(download_root, observation_id, filt, target_name, regex_tmpl)
#             if not fits_path:
#                 fits_path = find_fits_fallback(download_root, observation_id, filt)
#             if not fits_path:
#                 print(f'[WARN] FITS not found under {download_root}/** for obs={observation_id}, filter={filt}; marking decision=N.')
#                 append_screening_row(screening_csv, str(observation_id), target_name, filt, Path(f'{observation_id}_{filt}.fits'), 'N')
#                 continue

#             label = f'{target_name} | obs={observation_id} | filter={filt}'
#             print(f'\n=== {label} ===')
#             print(f'FITS: {fits_path}')

#             # Launch DS9 ONCE with regions preloaded (no XPA needed)
#             tf, ds9_proc = launch_ds9_with_regions(ds9_bin, fits_path, zoom, zscale, ra1, dec1, ra2, dec2)

#             # Prompt
#             decision = prompt_user(label)

#             # Close DS9 immediately after answer
#             close_ds9(ds9_proc)
#             try:
#                 tf.close()
#                 Path(tf.name).unlink(missing_ok=True)
#             except Exception:
#                 pass

#             # Record screening
#             append_screening_row(screening_csv, str(observation_id), target_name, filt, fits_path, decision)

#             # Photometry on Y/D — launch in PARALLEL
#             _ = run_photometry_if_needed(
#                 decision=decision,
#                 fits_path=fits_path,
#                 target_name=target_name,
#                 observation_id=str(observation_id),
#                 pos1_ra=ra1, pos1_dec=dec1,
#                 pos2_ra=ra2, pos2_dec=dec2,
#                 filter=str(filt),
#                 ini_path=ini_path
#             )

#             # Continue to next row immediately

#         except KeyboardInterrupt:
#             print('\n[INFO] Aborted by user.')
#             break
#         except Exception as e:
#             print(f'[ERROR] Failed on row: {e}', file=sys.stderr)
#             continue



#     print('\n[INFO] Screening completed.')

if __name__ == '__main__':
    config: ConfigParser = ConfigParser()
    config.read(sys.argv[1])
    
    main(config=config)
