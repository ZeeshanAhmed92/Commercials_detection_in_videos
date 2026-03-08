"""
S2_fingerprint_db.py  —  Build / update the ad-fingerprint SQLite database.
All tunable constants live in config.py.
"""

import os
import sqlite3
import hashlib
import time
from tqdm import tqdm
import numpy as np
import librosa
from scipy.ndimage import maximum_filter, binary_erosion, generate_binary_structure

from config import (
    SR_EXPECTED,
    N_FFT, HOP_LENGTH,
    AMP_MIN_DB,
    PEAK_NEIGHBORHOOD_FREQ, PEAK_NEIGHBORHOOD_TIME,
    FAN_VALUE, MAX_TIME_DELTA, HASH_TRUNCATE,
    BATCH_INSERT_SIZE,
    ADS_AUDIO_FOLDER, DB_PATH,
)


# ───────────────────────────── DATABASE ───────────────────────────────────────

def _table_names(language: str) -> tuple[str, str]:
    """
    Return (files_table, fingerprints_table) for a given language.
    E.g. 'Hindi' → ('files_Hindi', 'fingerprints_Hindi')
    Table names are sanitised so only alphanumerics and underscores are kept.
    """
    safe = "".join(c if c.isalnum() else "_" for c in language)
    return f"files_{safe}", f"fingerprints_{safe}"


def init_db(db_path: str, language: str) -> None:
    """
    Create per-language tables and indexes if they don't already exist.
    Each language gets its own isolated pair of tables:
        files_<lang>         — one row per ad file
        fingerprints_<lang>  — all hash rows for that language
    This eliminates cross-language hash collisions and makes every query
    cheaper because the table/index is much smaller.
    """
    files_tbl, fp_tbl = _table_names(language)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    cur = conn.cursor()

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {files_tbl} (
            ad_id        TEXT PRIMARY KEY,
            filename     TEXT,
            duration     REAL,
            content_hash TEXT UNIQUE
        )
    """)
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {fp_tbl} (
            hash        TEXT,
            ad_id       TEXT,
            time_offset REAL
        )
    """)

    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{fp_tbl}_hash ON {fp_tbl}(hash)")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{fp_tbl}_ad   ON {fp_tbl}(ad_id)")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{files_tbl}_hash ON {files_tbl}(content_hash)")

    conn.commit()
    conn.close()


def list_db_languages(db_path: str) -> list[str]:
    """Return all language names that have tables in the DB."""
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path, timeout=60)
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'files_%'"
    ).fetchall()
    conn.close()
    return [r[0][len("files_"):] for r in rows]


# ────────────────────────── SIGNAL PROCESSING ─────────────────────────────────

def compute_peaks(
    y: np.ndarray,
    sr: int,
    n_fft:      int = N_FFT,
    hop_length: int = HOP_LENGTH,
    amp_min_db: float = AMP_MIN_DB,
    neighborhood: tuple[int, int] = (PEAK_NEIGHBORHOOD_FREQ, PEAK_NEIGHBORHOOD_TIME),
) -> tuple[list, tuple]:
    """
    Return constellation-map peaks as [(freq_bin, time_frame), …] sorted by time.
    Uses the same algorithm as S3 so hashes are byte-for-byte identical.
    """
    S    = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    footprint = generate_binary_structure(2, 1)
    local_max = maximum_filter(S_db, size=neighborhood)
    peaks_mask = (S_db == local_max) & (S_db > amp_min_db)

    background = S_db <= amp_min_db
    eroded_bg  = binary_erosion(background, structure=footprint, iterations=1)
    peaks_mask = peaks_mask & ~eroded_bg

    freq_idx, time_idx = np.where(peaks_mask)
    peaks = sorted(zip(freq_idx.tolist(), time_idx.tolist()), key=lambda x: x[1])
    return peaks, S_db.shape


def generate_hashes(
    peaks:      list,
    hop_length: int   = HOP_LENGTH,
    sr:         int   = SR_EXPECTED,
    fan_value:  int   = FAN_VALUE,
    max_delta:  int   = MAX_TIME_DELTA,
) -> list[tuple[str, float]]:
    """Return [(hash_hex, time_offset_seconds), …]."""
    hashes = []
    n = len(peaks)
    for i in range(n):
        f1, t1 = peaks[i]
        for j in range(1, fan_value + 1):
            if i + j >= n:
                break
            f2, t2 = peaks[i + j]
            dt = t2 - t1
            if dt <= 0 or dt > max_delta:
                continue
            key = f"{f1}|{f2}|{dt}"
            h   = hashlib.sha1(key.encode()).hexdigest()[:HASH_TRUNCATE]
            t_s = (t1 * hop_length) / sr
            hashes.append((h, t_s))
    return hashes


def compute_content_hash(y: np.ndarray) -> str:
    """MD5 of the int16 PCM bytes — used to detect duplicate ad files."""
    int16 = (np.clip(y, -1.0, 1.0) * 32_767).astype(np.int16)
    return hashlib.md5(int16.tobytes()).hexdigest()


def process_file(file_path: str) -> tuple[list, float, np.ndarray]:
    y, sr = librosa.load(file_path, sr=SR_EXPECTED, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    peaks, _  = compute_peaks(y, sr)
    hashes    = generate_hashes(peaks)
    return hashes, duration, y


# ────────────────────────────── DB WRITES ─────────────────────────────────────

def upsert_ad(
    db_path:      str,
    ad_id:        str,
    filename:     str,
    duration:     float,
    hashes:       list[tuple[str, float]],
    content_hash: str,
    language:     str,
) -> None:
    """Insert or replace one ad's record into the language-specific tables."""
    files_tbl, fp_tbl = _table_names(language)

    conn = sqlite3.connect(db_path, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    cur  = conn.cursor()

    # Upsert file record (no language column — it's encoded in the table name)
    cur.execute(
        f"INSERT OR REPLACE INTO {files_tbl} (ad_id, filename, duration, content_hash) "
        f"VALUES (?, ?, ?, ?)",
        (ad_id, filename, duration, content_hash),
    )

    # Replace fingerprints for this ad
    cur.execute(f"DELETE FROM {fp_tbl} WHERE ad_id=?", (ad_id,))

    rows = [(h, ad_id, t) for h, t in hashes]
    for i in range(0, len(rows), BATCH_INSERT_SIZE):
        cur.executemany(
            f"INSERT INTO {fp_tbl} (hash, ad_id, time_offset) VALUES (?,?,?)",
            rows[i : i + BATCH_INSERT_SIZE],
        )
    conn.commit()
    conn.close()


# ──────────────────────────────── MAIN FLOW ───────────────────────────────────

def run_flow(
    ads_language:  str  | None = None,
    specific_ads:  list | None = None,
    force_reindex: bool        = False,
) -> None:
    """
    Fingerprint all ad WAV files and store results in per-language DB tables.

    Args:
        ads_language:  Sub-folder name (e.g. 'Hindi').  None → all languages.
        specific_ads:  List of filenames to (re)index (e.g. ['kent.mp4']).
                       None → all files in the language folder.
        force_reindex: If True, re-compute fingerprints even for known files.
    """
    search_path = (
        os.path.join(ADS_AUDIO_FOLDER, ads_language)
        if ads_language else ADS_AUDIO_FOLDER
    )

    if not os.path.exists(search_path):
        print(f"[ERROR] Fingerprint audio path does not exist: {search_path}")
        return

    # Collect all .wav files recursively
    all_files: list[str] = []
    for root, _, filenames in os.walk(search_path):
        for f in filenames:
            if f.lower().endswith(".wav"):
                all_files.append(os.path.join(root, f))

    # Filter to specific_ads if provided
    if specific_ads:
        if isinstance(specific_ads, str):
            specific_ads = [specific_ads]
        target_stems = {os.path.splitext(os.path.basename(a))[0] for a in specific_ads}
        all_files = [
            f for f in all_files
            if os.path.splitext(os.path.basename(f))[0] in target_stems
        ]

    if not all_files:
        print(f"[WARN] No .wav files found in {search_path} matching filter.")
        return

    # Group files by language (= their immediate parent folder name)
    from collections import defaultdict as _dd
    by_language: dict[str, list[str]] = _dd(list)
    for path in all_files:
        lang = os.path.basename(os.path.dirname(path))
        by_language[lang].append(path)

    total_hashes = 0
    t0 = time.time()

    for lang, lang_files in by_language.items():
        # Ensure the language-specific tables exist
        init_db(DB_PATH, lang)
        files_tbl, _ = _table_names(lang)

        # Pre-load known content hashes for THIS language table only
        conn = sqlite3.connect(DB_PATH, timeout=60)
        known_hashes: set[str] = {
            row[0] for row in conn.execute(
                f"SELECT content_hash FROM {files_tbl} WHERE content_hash IS NOT NULL"
            )
        }
        conn.close()

        print(f"\n[S2] Fingerprinting {len(lang_files)} file(s) for language '{lang}' …")

        for path in tqdm(lang_files, desc=f"Fingerprinting [{lang}]"):
            fname = os.path.basename(path)
            ad_id = os.path.splitext(fname)[0]

            try:
                hashes, duration, y = process_file(path)
                chash = compute_content_hash(y)
            except Exception as exc:
                print(f"\n[ERROR] Could not process {path}: {exc}")
                continue

            if not force_reindex and chash in known_hashes:
                print(f"[SKIP] {fname} — already in '{lang}' table.")
                continue

            upsert_ad(DB_PATH, ad_id, fname, duration, hashes, chash, lang)
            known_hashes.add(chash)
            total_hashes += len(hashes)
            print(f"  ✓  [{lang}] {fname}  dur={duration:.1f}s  → {len(hashes)} hashes")

    elapsed = time.time() - t0
    print(f"\n[S2] Done. {total_hashes:,} hashes stored in {elapsed:.1f}s")