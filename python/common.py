from __future__ import annotations
import csv
import os
import subprocess
from pathlib import Path
from typing import Any, Optional, Union, Sequence


# Columns
OBS_ID_COLS   = ('observation_id', 'obsid', 'observation')
TARGET_COLS   = ('sso_name', 'target_name', 'target')
FILTER_COLS   = ('filter', 'om_filter', 'band')
DECISION_COLS = ('DECISION',)
POS1_RA_COLS  = ('position_1_ra', 'ra_deg_1')
POS1_DEC_COLS = ('position_1_dec', 'dec_deg_1')
POS2_RA_COLS  = ('position_2_ra', 'ra_deg_2')
POS2_DEC_COLS = ('position_2_dec', 'dec_deg_2')
FITS_FILE_COLS = ('FITS_FILE', 'fits_name')

# Values
DETECTION_VALS = ('Y', 'D')
NON_DETECTION_VALS = ('N',)


def read_csv(
    filepath: str,
    raise_file_not_found: bool = False,
) -> tuple[Sequence[str], list[dict[str, Any]]]:
    """
    Reads a CSV file where values may be separated by ',' or ';'.
    Strips leading/trailing double quotes from each value.
    """
    headers: Sequence[str] = []
    rows = []

    try: 
        with open(filepath, newline='', encoding='utf-8') as f:
            for i, line in enumerate(f):
                # Normalize separators by replacing commas with semicolons
                line = line.replace(',', ';')
                # Split by semicolon and strip quotes/spaces
                values = [v.strip().strip('"') for v in line.split(';')]

                if i == 0:
                    headers = values
                else:
                    rows.append({header: values[j] if j < len(values) else '' for j, header in enumerate(headers)})

    except FileNotFoundError as e:
        if raise_file_not_found:
            raise e

    return headers, rows


def extract_row_value(row: dict[str, Any], columns: Union[str, Sequence[str]]) -> Any:
    """
    Extracts the value of a column from a row dict given one or more column names.

    Args:
        row: dictionary representing a CSV row.
        columns: a single column name or a list/tuple of column names.

    Returns:
        The value of the matching column.

    Raises:
        ValueError: if none of the columns are found or if multiple columns are found.
    """
    if isinstance(columns, str):
        columns = [columns]  # normalize to list

    # Find all matching columns present in the row
    found_columns = [col for col in columns if col in row]
    if len(found_columns) == 0:
        raise ValueError(f"None of the columns {columns} found in row {row}.")
    elif len(found_columns) > 1:
        raise ValueError(f"Multiple columns {found_columns} found in row {row}.")
    
    return row[found_columns[0]]


def extract_matching_rows(
    filepath: str,
    columns: Union[str, Sequence[str]],
    value: str,
) -> list[dict[str, Any]]:
    headers, csv_rows = read_csv(filepath=filepath)

    if len(csv_rows) == 0:
        return []

    # Check that at least a column is present
    if not any(column in headers for column in columns):
        raise ValueError(
            f"None of the columns {columns} found in CSV headers {headers}."
        )
    
    matching_rows: list[dict] = []
    try:
        for csv_row in csv_rows:
            if any(csv_row.get(column) == value for column in columns if column in headers):
                matching_rows.append(csv_row)

    except FileNotFoundError:
        return []
    
    return matching_rows


def append_row(filepath: str, row: dict[str, Any]) -> None:
    """
    Appends a row to a CSV file. If the file does not exist, it is created
    with headers taken from row.keys().

    The provided dict may have keys in any order; they are reordered to match
    the file's header order when appending.
    """

    file_exists = os.path.exists(filepath)

    if not file_exists:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    # File exists: match existing header order
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # first row is header

    # Ensure provided dict doesn’t contain unknown fields
    unknown = set(row.keys()) - set(header)
    if unknown:
        raise ValueError(f"Row contains unknown fields: {unknown}")

    # Build ordered row matching header
    ordered_row = {col: row.get(col, "") for col in header}

    with open(filepath, "a", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writerow(ordered_row)


def find_fits_files(
    folder: str,
    observation_id: str,
    filter: str,
) -> list[Path]:
    """
    Returns all files under folder/observation_id/filter as Path objects.

    Args:
        folder: base folder path
        observation_id: subfolder name for the observation
        filter: subfolder name for the filter

    Returns:
        list of Path objects for all files under the specified folder.
    """
    base_path = Path(folder) / observation_id / filter
    
    if not base_path.exists() or not base_path.is_dir():
        return []  # Return empty list if path does not exist

    # list only files (ignore subdirectories)
    files = [f for f in base_path.iterdir() if f.is_file()]

    return files


def close_subprocess(proc: Optional[subprocess.Popen]):
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
        except Exception:
            pass


def request_user_input(prompt: str, valid_inputs: Sequence[str] | None = None) -> str:
    while True:
        ans = input(prompt).strip().upper()
        if valid_inputs is None:
            return ans
        if ans in valid_inputs:
            return ans
        print(f'Please input a valid option from: {valid_inputs}')
