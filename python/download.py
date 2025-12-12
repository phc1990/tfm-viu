# -*- coding: utf-8 -*-
"""
download.py — NXSA helpers for OM products

Provides:
- ensure_om_srclist_or_obsmlist(obs_id, force=False) -> (Path, "OBSMLI")
- download_om_mosaic_sky_image(obs_id, filter_code, force=False) -> Path
- extract_obsml_sky_image_for_band(obs_id, obsml_path, band) -> Path (optional)
"""

from __future__ import annotations

import tarfile
import tempfile
import shutil
import time
from pathlib import Path
from typing import Optional, Tuple, List

from astropy.io import fits
from astroquery.esa.xmm_newton import XMMNewton


# ---------------------- tiny logger ----------------------

def _log(msg: str) -> None:
    print(msg)


# ---------------------- small utils ----------------------

def _fits_ok(path: Path) -> bool:
    try:
        with fits.open(str(path), memmap=False) as _:
            return True
    except Exception:
        return False


def _member_names(tf: tarfile.TarFile, limit: int = 100) -> List[str]:
    return [m.name for m in tf.getmembers()[:limit]]


def _safe_first_path(maybe_paths) -> Optional[Path]:
    if maybe_paths is None:
        return None
    if isinstance(maybe_paths, (list, tuple)) and maybe_paths:
        try:
            return Path(maybe_paths[0]).resolve()
        except Exception:
            return None
    try:
        return Path(maybe_paths).resolve()
    except Exception:
        return None


def _most_recent_matching(base: Path, patterns: List[str]) -> Optional[Path]:
    """
    Find the most recently modified file under 'base' matching any glob patterns.
    """
    best_p = None
    best_mtime = -1.0
    for pat in patterns:
        for p in base.glob(pat):
            try:
                mtime = p.stat().st_mtime
            except Exception:
                continue
            if mtime > best_mtime:
                best_mtime = mtime
                best_p = p
    return best_p.resolve() if best_p else None

from pathlib import Path
import shutil

def _move_to_dest(src: Path, dest_dir: Path, force: bool = False) -> Path:
    """
    Mueve (o copia si se prefiere) el FITS 'src' al directorio 'dest_dir'.
    Si ya existe y no 'force', devuelve el existente.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dst = dest_dir / src.name
    if dst.exists():
        if force:
            try:
                dst.unlink()
            except Exception:
                pass
        else:
            return dst
    try:
        # mover (más rápido si misma FS); si falla, copiar
        try:
            shutil.move(str(src), str(dst))
        except Exception:
            shutil.copy2(str(src), str(dst))
    except Exception as e:
        print(f"[NXSA] WARN: no se pudo mover {src} → {dst}: {e}. Usando {src}")
        return src
    return dst


# ---------------------- core fetcher ----------------------

def _download_product(obs_id: str, product_name: str, force: bool = False) -> Tuple[Path, bool]:
    """
    Attempt to fetch a single FTZ or a TAR containing the requested product.
    Returns (path, is_tar).

    product_name examples:
      - "OBSMLI"  (OM Observation Source List)
      - "USIMAG"/"LSIMAG"/"HSIMAG"/"RSIMAG" (OM mosaic sky images)
    """
    # Try normal call
    p = None
    try:
        out = XMMNewton.download_data(
            observation_id=str(obs_id),
            instrument="OM",
            level="PPS",
            name=product_name,
            extension="FTZ",
            overwrite=force
        )
        p = _safe_first_path(out)
    except Exception as e:
        _log(f"[NXSA] NOTE: {product_name} (overwrite={force}) raised: {e}")

    # Fallback: astroquery sometimes downloads but returns None — scan CWD
    if p is None:
        # Allow a brief moment for file to flush to disk
        time.sleep(0.1)
        # Common names:
        #   - <obsid>.tar
        #   - <obsid>.FTZ
        #   - GUEST*.tar / *_aio.tar
        # We accept any tar/FTZ that includes obsid token in the filename.
        obs_token = str(obs_id)
        candidates = [
            f"*{obs_token}*.tar",
            f"*{obs_token}*.FTZ",
            f"GUEST*{obs_token}*.tar",
            f"*aio*{obs_token}*.tar",
        ]
        found = _most_recent_matching(Path.cwd(), candidates)
        if found and found.exists():
            p = found.resolve()

    if p and p.exists():
        return p, (p.suffix.lower() == ".tar")

    # Try forcing overwrite as a second attempt
    if not force:
        try:
            out = XMMNewton.download_data(
                observation_id=str(obs_id),
                instrument="OM",
                level="PPS",
                name=product_name,
                extension="FTZ",
                overwrite=True
            )
            p = _safe_first_path(out)
        except Exception as e:
            _log(f"[NXSA] WARN: {product_name} retry failed: {e}")

        if p is None:
            # Scan again after overwrite attempt
            time.sleep(0.1)
            found = _most_recent_matching(Path.cwd(), [
                f"*{obs_token}*.tar",
                f"*{obs_token}*.FTZ",
                f"GUEST*{obs_token}*.tar",
                f"*aio*{obs_token}*.tar",
            ])
            if found and found.exists():
                p = found.resolve()

        if p and p.exists():
            return p, (p.suffix.lower() == ".tar")

    raise FileNotFoundError(f"NXSA product {product_name} not found for obsid={obs_id} in CWD.")


def _extract_first_member_matching(
    tar_path: Path,
    want_tokens: List[str],
    extensions: Tuple[str, ...] = (".ftz", ".fits", ".fit"),
    negative_tokens: Optional[List[str]] = None,
) -> Optional[Path]:
    """
    Extract the best-scoring member whose basename contains all tokens in want_tokens.
    Negative tokens (if provided) will penalize/skip members.

    Returns path to extracted file or None.
    """
    neg = tuple(t.lower() for t in (negative_tokens or []))
    toks = tuple(t.lower() for t in want_tokens)

    with tarfile.open(tar_path, mode="r:*") as tf:
        scored: List[Tuple[int, tarfile.TarInfo]] = []
        for m in tf.getmembers():
            base = Path(m.name).name
            low = base.lower()
            if not low.endswith(extensions):
                continue
            if any(t not in low for t in toks):
                continue
            if neg and any(t in low for t in neg):
                # hard skip
                continue
            score = 0
            if "omx" in low:
                score += 4
            if "oms" in low:
                score += 2
            if "obs" in low:
                score += 2
            if "srcli" in low or "srclist" in low:
                score -= 4
            if "pps" in low:
                score += 1
            scored.append((score, m))

        if not scored:
            return None

        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1]
        out_dir = Path(tempfile.mkdtemp(prefix=f"nxsa_extract_"))
        out_path = (out_dir / Path(best.name).name).resolve()
        src = tf.extractfile(best)
        if src is None:
            return None
        with open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

    if _fits_ok(out_path):
        return out_path
    return None


# ---------------------- OBSMLI only ----------------------

def ensure_om_srclist_or_obsmlist(obs_id: str,
                                  force: bool = False,
                                  dest_root: str | Path | None = None) -> tuple[Path, str]:
    """
    Fetch and return the OM list product used for AB zero-point calibration.

    Only OBSMLI (OM observation source list) is used — it carries ABM0UVW* keys.

    Returns: (path_to_FTZ_or_FITS, "OBSMLI")

    The function handles:
      - Direct FTZ download (path points to the FTZ/FITS).
      - TAR download (GUEST...tar, <obsid>.tar, etc.) and extraction of
        the most plausible '...OBSMLI...' member (usually under pps/).
    """
    #Try first if mosaic already exists (skip new download)
    if dest_root.exists():
        candidates = sorted(dest_root.glob("*OMX*OBSMLI*"))
        if candidates:
            _log(f"[NXSA] OBSMLI candidate exist-no need to re-download: {candidates[0]}")
            return candidates[0], "OBSMLI"


    # Try to obtain OBSMLI
    path, is_tar = _download_product(obs_id, "OBSMLI", force=force)

    if not is_tar:
        # got a direct FTZ/FITS
        return path, "OBSMLI"

    # We have a tar (could be GUEST*.tar). Extract a ...OBSMLI... member.
    # Typical names include pps/P<obsid>OMX000OBSMLI0000.FTZ (or .FIT)
    out = _extract_first_member_matching(
        tar_path=path,
        want_tokens=["obsmli"],  # focus on OBSMLI
        negative_tokens=["srcli", "srclist"]
    )
    if out:
        if dest_root is None:
            return out, "OBSMLI"

        dest_root = Path(dest_root).expanduser().resolve()
        final_path = _move_to_dest(Path(out), dest_root, force=force)
        print(f"[NXSA] Guardado OBSMLI en: {final_path}")
        return final_path, "OBSMLI"

    # Debug preview to help refine matching
    with tarfile.open(path, mode="r:*") as tf:
        preview = "\n  - ".join(_member_names(tf, 80))
    raise FileNotFoundError(
        f"Could not find OBSMLI member in {path.name}.\nMembers:\n  - {preview}"
    )


# ---------------------- OBSMLI → sky image (optional) ----------------------

def extract_obsml_sky_image_for_band(obs_id: str, obsml_path: Path, band: str) -> Path:
    """
    Try to obtain a sky image referenced by an OBSMLI product for the given band (UVW1/UVM2/UVW2/V/B/U).
    If a filename isn't directly present, fall back to scanning the OBSMLI TAR members.

    Returns path to extracted FITS (FTZ).

    NOTE: Your main flow should prefer the mosaiced sky image via download_om_mosaic_sky_image().
    """
    band = (band or "").strip().upper()
    col_name = f"{band}_SKY_IMAGE"

    # Best-effort filename from header/table (may be absent/placeholder)
    sky_basename: Optional[str] = None
    try:
        with fits.open(str(obsml_path), memmap=False) as hdul:
            if len(hdul) > 1 and hasattr(hdul[1], "header"):
                hdr = hdul[1].header
                if col_name in hdr:
                    val = str(hdr[col_name]).strip()
                    if val.lower().endswith((".fit", ".fits", ".ftz")):
                        sky_basename = Path(val).name
            if sky_basename is None and len(hdul) > 1 and hasattr(hdul[1], "data") and hdul[1].data is not None:
                data = hdul[1].data
                if col_name in data.columns.names:
                    for v in data[col_name]:
                        s = (v.decode() if isinstance(v, (bytes, bytearray)) else str(v)).strip()
                        if s.lower().endswith((".fit", ".fits", ".ftz")):
                            sky_basename = Path(s).name
                            break
    except Exception:
        pass

    # Re-download OBSMLI as a TAR to scan members (even if we already have a single FTZ)
    tar_path, is_tar = _download_product(str(obs_id), "OBSMLI", force=False)
    if not is_tar:
        # Try again forcing overwrite; some servers hand back FTZ instead of TAR
        tar_path, is_tar = _download_product(str(obs_id), "OBSMLI", force=True)
        if not is_tar:
            raise FileNotFoundError("OBSMLI returned a single FTZ; cannot scan members for sky image.")

    with tarfile.open(tar_path, mode="r:*") as tf:
        members = tf.getmembers()

        # If a basename was found, try exact match first
        if sky_basename:
            exact = [m for m in members if Path(m.name).name.lower() == sky_basename.lower()]
            if exact:
                m = exact[0]
                out_dir = Path(tempfile.mkdtemp(prefix=f"nxsa_sky_{obs_id}_{band}_"))
                out_path = (out_dir / Path(m.name).name).resolve()
                src = tf.extractfile(m)
                if src is None:
                    raise IOError(f"Could not read member {m.name} from {tar_path}")
                with open(out_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                if not _fits_ok(out_path):
                    raise IOError(f"Extracted sky image is not a valid FITS: {out_path}")
                return out_path
            # else: fall through to heuristics

        # Heuristic: pick a member whose name looks like a sky image for this band
        band_tokens = {
            "UVW1": ("uvw1", "w1"),
            "UVM2": ("uvm2", "m2"),
            "UVW2": ("uvw2", "w2"),
            "V": ("v",),
            "B": ("b",),
            "U": ("u",),
        }.get(band, (band.lower(),))

        def score_member(m: tarfile.TarInfo) -> int:
            name = Path(m.name).name.lower()
            if not name.endswith((".fit", ".fits", ".ftz")):
                return -1
            s = 0
            if "sky" in name:
                s += 10
            if "skyimage" in name or "sky_image" in name:
                s += 8
            if "sw" in name and "sky" in name:
                s += 4
            if "oms" in name:
                s += 2
            if "omx" in name:
                s += 1
            if any(t in name for t in band_tokens):
                s += 7
            # Avoid SRCLIST-ish things
            if "srcli" in name or "srclist" in name or "obsmli" in name:
                s -= 6
            if "reg" in name or "mask" in name:
                s -= 2
            return s

        scored = [(score_member(m), m) for m in members]
        scored.sort(key=lambda x: x[0], reverse=True)
        for sc, m in scored:
            if sc <= 0:
                break
            out_dir = Path(tempfile.mkdtemp(prefix=f"nxsa_sky_{obs_id}_{band}_"))
            out_path = (out_dir / Path(m.name).name).resolve()
            src = tf.extractfile(m)
            if src is None:
                continue
            with open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            if _fits_ok(out_path):
                return out_path

        preview = "\n  - " + "\n  - ".join(_member_names(tf, 80))
        raise FileNotFoundError(
            f"Could not locate a plausible {band} sky image within OBSMLI tar for obsid={obs_id}."
            f"\nPreview of members:{preview}"
        )


# ---------------------- OM mosaic sky image (SIMAG) ----------------------

def _extract_best_simag_from_tar(tar_path: Path, obs_id: str, filt_letter: str) -> Path:
    """
    Select the most appropriate *SIMAG<letter>* FIT(S|Z) inside the TAR.
    Example matches: ...USIMAGL..., ...LSIMAGS..., ...RSIMAGB..., ...HSIMAGU...
    """
    fl = (filt_letter or "").strip().upper()
    token = f"simag{fl}".lower()

    with tarfile.open(tar_path, mode="r:*") as tf:
        candidates = []
        for m in tf.getmembers():
            base = Path(m.name).name.lower()
            if not base.endswith((".ftz", ".fits", ".fit")):
                continue
            if token not in base:
                continue
            score = 0
            if "omx" in base:
                score += 5
            if "oms" in base:
                score += 2
            if "sky" in base:
                score += 2
            if any(k in base for k in ("usimag", "lsimag", "hsimag", "rsimag")):
                score += 3
            candidates.append((score, m))

        if not candidates:
            preview = "\n  - " + "\n  - ".join(_member_names(tf, 80))
            raise FileNotFoundError(
                f"No SIMAG{fl} FITS found in {tar_path.name} for obsid={obs_id}.{preview}"
            )

        candidates.sort(key=lambda x: x[0], reverse=True)
        best = candidates[0][1]
        out_dir = Path(tempfile.mkdtemp(prefix=f"nxsa_om_mosaic_{obs_id}_{fl}_"))
        out_path = (out_dir / Path(best.name).name).resolve()
        src = tf.extractfile(best)
        if src is None:
            raise IOError(f"Could not read member {best.name} from {tar_path}")
        with open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

    if not _fits_ok(out_path):
        raise IOError(f"Extracted mosaic sky image is not a valid FITS: {out_path}")

    return out_path


def _filter_letter_from_code(filter_code: str) -> str:
    """
    Input codes you’re using: V,B,U,L,M,S (L=UVW1, M=UVM2, S=UVW2).
    These are exactly the letters used in SIMAG filenames.
    """
    return (filter_code or "").strip().upper()


def download_om_mosaic_sky_image(obs_id: str,
                                 filter_code: str,
                                 force: bool = False,
                                 dest_root: str | Path | None = None) -> Path:
    """
    Download + extract the OM mosaiced sky image for the given obsid & filter.
    Tries USIMAG, LSIMAG, HSIMAG, RSIMAG (whichever is present for that observation).
    Returns a path to the extracted FIT(S|Z) file.
    """
    fl = _filter_letter_from_code(filter_code)
    # _log(f"[NXSA] intput dest_root: {dest_root}")

    #Try first if mosaic already exists (skip new download)
    if dest_root.exists():
        # OMX*SIMAG*
        candidates = sorted(dest_root.glob("*OMX*SIMAG*"))
        _log(f"[NXSA] intput dest_root candidate exist-no need to re-download: {candidates[0]}")
        if candidates:
            return candidates[0]


    # Try the common order; not all products are present for every obs
    # for product in ("USIMAG", "LSIMAG", "HSIMAG", "RSIMAG"):
    product = f"{fl}SIMAG"
    try:
        path, is_tar = _download_product(obs_id, product_name=product, force=force)
    except Exception as e:
        _log(f"[NXSA] {product}: {e}")
        # continue

    if not is_tar:
        # Single FITS/FTZ returned — verify it matches our band
        p = path
        if f"simag{fl}".lower() in p.name.lower():
            _log(f"[NXSA] Got single mosaic FITS for {product}: {p.name}")
            out = p
            # return p
        else:
            _log(f"[NXSA] Single FITS did not match band={fl}: {p.name}")
            # continue

    # Extract the correct SIMAG<letter> member out of the tar
    try:
        out = _extract_best_simag_from_tar(path, obs_id, fl)
        _log(f"[NXSA] Extracted mosaic sky image: {out.name}")
        # return out
    except Exception as e:
        _log(f"[NXSA] WARN: Couldn’t extract SIMAG{fl} from {product} tar: {e}")
        # continue

    if dest_root is None:
        return out

    dest_root = Path(dest_root).expanduser().resolve()
    # filt = (filter_code or "").strip().upper()
    # dest_dir = dest_root / str(obs_id) / filt
    final_path = _move_to_dest(Path(out), dest_root, force=force)

    print(f"[NXSA] Mosaic stored in: {final_path}")
    return final_path

    raise IOError(f"Could not obtain OM mosaic sky image (SIMAG{fl}) for obsid={obs_id}.")
