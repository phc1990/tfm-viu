"""
Observations-related module.

Robust CSV repository that:
- Auto-detects delimiter (supports ';', ',', '\\t', '|').
- Supports both headered CSV (DictReader) and positional CSV when headers are skipped.
- Understands modern column names:
    observation_id; sso_name; ra_deg_1; dec_deg_1; ra_deg_2; dec_deg_2; filter; xmatch_type
  and legacy names:
    id; object; ra1; dec1; ra2; dec2; filters
- Converts a single 'filter' value (e.g., 'L') into a 'filters' list (['L']).
"""

from typing import Iterator, List, Optional
import csv
import io
import logging
import os

# Public order used elsewhere (e.g., CsvRecorder headers)
_FILTERS = ['S', 'M', 'L', 'U', 'B', 'V']

LOGGER = logging.getLogger(__name__)


class Observation:
    """Represents a scientific observation."""
    def __init__(
        self,
        id: str,
        object: str,
        ra1: float,
        dec1: float,
        ra2: float,
        dec2: float,
        filters: List[str],
    ):
        """
        Args:
            id (str): observation identifier
            object (str): name of the potentially observed object
            ra1 (float): right ascension at start [deg]
            dec1 (float): declination at start [deg]
            ra2 (float): right ascension at end [deg]
            dec2 (float): declination at end [deg]
            filters (List[str]): list of filters (e.g. ['L','U'])
        """
        self.id = id
        self.object = object
        self.ra1 = ra1
        self.dec1 = dec1
        self.ra2 = ra2
        self.dec2 = dec2
        self.filters = filters


class Repository:
    """Observation repository interface."""
    def get_iter(self) -> Iterator[Observation]:
        """Returns the observations iterator."""
        raise NotImplementedError


def _to_float(value: str, field: str, line_no: Optional[int] = None) -> float:
    try:
        return float(value)
    except Exception as e:
        where = f" (line {line_no})" if line_no is not None else ""
        raise ValueError(f"Invalid float for '{field}'{where}: {value!r}") from e


def _split_filters(raw: Optional[str]) -> List[str]:
    """
    Convert a raw filter(s) string to a list:
    - If None/empty -> []
    - If it contains common separators (; , | whitespace) -> split on them
    - Else -> single-element list
    """
    if raw is None:
        return []
    text = str(raw).strip().strip('"').strip("'")
    if not text:
        return []
    for sep in [';', ',', '|', ' ']:
        if sep in text:
            parts = [p.strip() for p in text.split(sep) if p.strip()]
            if parts:
                return parts
    return [text]


def _sniff_dialect(sample: str) -> csv.Dialect:
    try:
        return csv.Sniffer().sniff(sample, delimiters=[',', ';', '\t', '|'])
    except Exception:
        class _Default(csv.Dialect):
            delimiter = ','
            quotechar = '"'
            escapechar = None
            doublequote = True
            skipinitialspace = False
            lineterminator = '\n'
            quoting = csv.QUOTE_MINIMAL
        return _Default()


class CsvRepository(Repository):
    """
    Observation repository using a CSV file.

    Supported input schemas:

    A) Modern (semicolon or comma delimited, quoted header):
       observation_id; sso_name; ra_deg_1; dec_deg_1; ra_deg_2; dec_deg_2; filter; xmatch_type

    B) Legacy (comma delimited, header optional, 'filters' column contains ';'-separated list):
       id, object, ra1, dec1, ra2, dec2, filters

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    ignore_top_n_lines : int, default 0
        Number of initial lines to skip (e.g., 1 to skip a header the user knows exists).
        When >0, the repository will parse rows positionally using the "modern" order A
        (falling back to the legacy order if there are exactly 7 columns).
    encoding : str, default 'utf-8'
        File encoding.
    """

    def __init__(self, csv_path: str, ignore_top_n_lines: int = 0, encoding: str = 'utf-8'):
        super().__init__()
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Observations CSV not found: {csv_path}")

        # 1) Read a sample from the beginning to sniff delimiter/quoting.
        with open(csv_path, 'r', encoding=encoding, newline='') as f:
            sample = f.read(4096)
        dialect = _sniff_dialect(sample)

        observations: List[Observation] = []

        # 2) Parse the file content after skipping top lines WITHOUT using the iterator,
        #    and WITHOUT seeking back/forward after iteration.
        with open(csv_path, 'r', encoding=encoding, newline='') as f:
            # Skip requested top lines using readline() (avoid iterator buffering)
            for _ in range(max(0, ignore_top_n_lines)):
                _ = f.readline()

            # Peek a block for header detection, then read the rest and rebuild a full buffer.
            peek = f.read(2048)
            rest = f.read()
            remaining_text = peek + rest  # includes the peeked bytes

        # Decide if a header exists in the remaining text (if user didn’t ask to skip it)
        use_dict_reader = False
        if ignore_top_n_lines == 0:
            try:
                use_dict_reader = csv.Sniffer().has_header(peek)
            except Exception:
                use_dict_reader = False

        buffer = io.StringIO(remaining_text)

        if use_dict_reader:
            reader = csv.DictReader(buffer, dialect=dialect)
            # line numbers will refer to the post-skip content (header consumed by DictReader)
            for i, row in enumerate(reader, start=2):  # +2: header + 1-based
                line_no = ignore_top_n_lines + i
                try:
                    obs = self._row_to_observation_dict(row, line_no=line_no)
                    if obs:
                        observations.append(obs)
                except Exception as e:
                    LOGGER.warning("Skipping malformed row %d: %s", line_no, e)
        else:
            reader = csv.reader(buffer, dialect=dialect)
            for i, row in enumerate(reader, start=1):
                line_no = ignore_top_n_lines + i
                if not row or all((c or "").strip() == "" for c in row):
                    continue
                try:
                    obs = self._row_to_observation_positional(row, line_no=line_no)
                    if obs:
                        observations.append(obs)
                except Exception as e:
                    LOGGER.warning("Skipping malformed row %d: %s", line_no, e)

        self.iter = iter(observations)

    # ---------- row parsers ----------

    def _row_to_observation_dict(self, row: dict, line_no: Optional[int] = None) -> Optional[Observation]:
        # Normalize keys to lowercase without quotes
        norm = {str(k).strip().strip('"').strip("'").lower(): v for k, v in row.items()}

        # Modern names
        id_val = norm.get("observation_id") or norm.get("id")
        obj_val = norm.get("sso_name") or norm.get("object") or ""
        ra1_val = norm.get("ra_deg_1") or norm.get("ra1")
        dec1_val = norm.get("dec_deg_1") or norm.get("dec1")
        ra2_val = norm.get("ra_deg_2") or norm.get("ra2")
        dec2_val = norm.get("dec_deg_2") or norm.get("dec2")

        # Filters: either 'filter' (single) or 'filters' (potential list)
        filt_raw = norm.get("filter")
        filters_raw = norm.get("filters")
        filters = _split_filters(filt_raw if filt_raw is not None else filters_raw)

        if not id_val:
            raise ValueError("missing 'observation_id'/'id'")
        if ra1_val is None or dec1_val is None or ra2_val is None or dec2_val is None:
            raise ValueError("missing one of RA/DEC fields")

        ra1 = _to_float(ra1_val, "ra1", line_no)
        dec1 = _to_float(dec1_val, "dec1", line_no)
        ra2 = _to_float(ra2_val, "ra2", line_no)
        dec2 = _to_float(dec2_val, "dec2", line_no)

        return Observation(
            id=str(id_val).strip().strip('"').strip("'"),
            object=str(obj_val or "").strip().strip('"').strip("'"),
            ra1=ra1,
            dec1=dec1,
            ra2=ra2,
            dec2=dec2,
            filters=filters,
        )

    def _row_to_observation_positional(self, row: List[str], line_no: Optional[int] = None) -> Optional[Observation]:
        # Trim surrounding whitespace/quotes for each cell
        cleaned = [ (c or "").strip().strip('"').strip("'") for c in row ]

        if len(cleaned) >= 8:
            # Modern order with xmatch_type present
            (
                id_val,
                obj_val,
                ra1_val,
                dec1_val,
                ra2_val,
                dec2_val,
                filter_or_filters,
                _xmatch_type,
            ) = cleaned[:8]
        elif len(cleaned) == 7:
            # Legacy order without xmatch_type
            (
                id_val,
                obj_val,
                ra1_val,
                dec1_val,
                ra2_val,
                dec2_val,
                filter_or_filters,
            ) = cleaned
        else:
            raise ValueError(f"expected 7 or 8 columns, got {len(cleaned)}: {cleaned!r}")

        ra1 = _to_float(ra1_val, "ra1", line_no)
        dec1 = _to_float(dec1_val, "dec1", line_no)
        ra2 = _to_float(ra2_val, "ra2", line_no)
        dec2 = _to_float(dec2_val, "dec2", line_no)
        filters = _split_filters(filter_or_filters)

        return Observation(
            id=str(id_val),
            object=str(obj_val or ""),
            ra1=ra1,
            dec1=dec1,
            ra2=ra2,
            dec2=dec2,
            filters=filters,
        )

    # ---------- public ----------

    def get_iter(self) -> Iterator[Observation]:
        """Returns the observations iterator."""
        return self.iter