from __future__ import annotations
from configparser import ConfigParser
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence, List


from common import OBS_ID_COLS, TARGET_COLS, FILTER_COLS, DECISION_COLS, POS1_DEC_COLS, POS1_RA_COLS, POS2_DEC_COLS, POS2_RA_COLS, FITS_FILE_COLS, DETECTION_VALS, NON_DETECTION_VALS
from common import extract_row_value, append_row, find_fits_files, find_fits_files_without_srclist, close_subprocess, request_user_input 


def _ds9_zoom_args(zoom: str) -> List[str]:
    z = (zoom or '').strip()
    if z.lower() == 'to fit':
        return ['-zoom', 'to', 'fit']
    parts = z.split()
    if len(parts) > 1:
        return ['-zoom', *parts]
    return ['-zoom', z]


def _ds9_launch_with_regions(
    config: ConfigParser,
    fits_paths: Path | Sequence[Path],
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
    reg_text = (
        'global color=cyan width=2\n'
        'fk5\n'
        f'circle({ra1},{dec1},10") # text={{pos1}}\n'
        f'circle({ra2},{dec2},10") # text={{pos2}}\n'
    )
    tf = tempfile.NamedTemporaryFile('w', suffix='.reg', delete=False)
    tf.write(reg_text)
    tf.flush()

    env = os.environ.copy()
    env.setdefault('DISPLAY', ':0')

    cmd: list[str] = [config['SCREENING']['DS9_BINARY_FILEPATH']]

    # 1) load all images → DS9 creates one frame per image
    cmd += [str(p) for p in fits_list]

    # 2) set scale on current frame, then match + lock across frames
    if config.getboolean('SCREENING', 'ZSCALE'):
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
    cmd += _ds9_zoom_args(config['SCREENING']['ZOOM'])

    # 4) load regions on *all* frames
    cmd += ['-regions', 'load', 'all', tf.name]

    proc = subprocess.Popen(cmd, env=env)
    time.sleep(0.4)  # small delay so DS9 has time to render

    return tf, proc


def action_screening(
    config: ConfigParser,
    input_row: dict[str, Any],
) -> None:
    observation_id: str = extract_row_value(input_row, OBS_ID_COLS)
    filter: str = extract_row_value(input_row, FILTER_COLS)
    target: str = extract_row_value(input_row, TARGET_COLS)
    ra1: float = float(extract_row_value(input_row, POS1_RA_COLS))
    dec1: float = float(extract_row_value(input_row, POS1_DEC_COLS))
    ra2: float = float(extract_row_value(input_row, POS2_RA_COLS))
    dec2: float = float(extract_row_value(input_row, POS2_DEC_COLS))

    fits_paths: list[Path] = find_fits_files_without_srclist(
        folder=config['INPUT']['DOWNLOAD_DIRECTORY'],
        observation_id=observation_id,
        filter=filter,
        debug=True
    )

    if not fits_paths:
        raise RuntimeError(f"Could not find FITS files for observation {observation_id}.")    

    label: str = f"{target} | obs={observation_id} | filter={filter}"
    print(f"Screening {label}. Frames loaded:")
    for i, p in enumerate(fits_paths, start=1):
        print(f'  [{i}] {p.name}')

    # --- open DS9 with ALL frames + regions on pos1/pos2 ---
    temp_ds9_region_file, ds9_process = _ds9_launch_with_regions(
        config=config,
        fits_paths=fits_paths,
        ra1=ra1,
        dec1=dec1,
        ra2=ra2,
        dec2=dec2,
    )

    # ---- interactive detection loop for this observation ----
    while True:
        decision: str = request_user_input(
            prompt=f'{label} Detection? [Y]es / [N]o / [D]ubious:',
            valid_inputs=DETECTION_VALS + NON_DETECTION_VALS,
        ) 

        # Close DS9 and temporary file after the first definitive decision
        close_subprocess(ds9_process)
        temp_ds9_region_file.close()
        Path(temp_ds9_region_file.name).unlink(missing_ok=True)

        if decision in DETECTION_VALS:
            # Ask which exposures have the detection
            while True:
                raw = request_user_input(
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
                    append_row(
                        filepath=config['SCREENING']['FILEPATH'],
                        row={
                            **input_row,
                            FITS_FILE_COLS[0]: fits_paths[i-1].name,
                            DECISION_COLS[0]: decision,
                        }
                    )

                break  # indices accepted

            # No more detections in this observation → go to next observation row
            break

        else:
            append_row(
                filepath=config['SCREENING']['FILEPATH'],
                row={
                    **input_row,
                    FITS_FILE_COLS[0]: None,
                    DECISION_COLS[0]: NON_DETECTION_VALS[0],
                }
            )
            break  # go to next observation row
