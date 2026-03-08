"""
S3_scan_test_improved_latest.py  —  Ad detection in mixed-broadcast audio
                                    (parallel-optimised)

Parallelism improvements over the previous version
────────────────────────────────────────────────────
1. FILE-LEVEL PARALLELISM  (unchanged from previous, but now correctly sized)
   detect_ads() scans multiple WAV files concurrently with a ThreadPoolExecutor.
   librosa / NumPy / SciPy release the GIL for most heavy operations, so threads
   are effective and the large fp_index dict is shared in memory without copying.

2. HASH-LOOKUP PARALLELISM  (new in detect_for_file)
   The mixed-audio hash list is partitioned into HASH_LOOKUP_CHUNKS equal chunks.
   Each chunk's delta-accumulation runs in its own thread.  The partial dicts are
   merged after all threads finish.  Benefit: large fp_index lookups saturate
   multiple cores on long broadcast files.

3. BOUNDARY REFINEMENT PARALLELISM  (new)
   After merging raw detections, all refine_boundaries() calls run concurrently
   in a small ThreadPoolExecutor (capped at REFINE_WORKERS).  The cross-correlation
   in fftconvolve() releases the GIL, so real speedup is observed even on CPython.

4. LOAD-FP-INDEX PARALLELISM  (new)
   When multiple languages are requested, each language table is loaded in its
   own thread, then results are merged.

All tunable constants live in config.py.  New constants added to config:
    HASH_LOOKUP_CHUNKS  = 4     # partitions for parallel hash lookup
    REFINE_WORKERS      = 4     # threads for parallel boundary refinement
(if absent, safe defaults are used here)
"""

import os
import sqlite3
import hashlib
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import multiprocessing
from typing import Optional

from tqdm import tqdm
import numpy as np
import librosa
import pandas as pd
from scipy.ndimage import maximum_filter, binary_erosion, generate_binary_structure
from scipy.signal import fftconvolve

from config import (
    SR_EXPECTED,
    N_FFT, HOP_LENGTH,
    AMP_MIN_DB,
    PEAK_NEIGH_FREQ, PEAK_NEIGH_TIME,
    FAN_VALUE, MAX_TIME_DELTA, HASH_TRUNCATE,
    DELTA_BIN_SIZE, MIN_DETECTION_SEC, MIN_CONFIDENCE, MERGE_GAP,
    LOCAL_MARGIN, CORR_THRESHOLD,
    FULL_DUR_TOLERANCE, FULL_SCORE_FLOOR,
    DB_PATH, ADS_AUDIO_FOLDER, MIXED_AUDIO_ROOT,
    BULK_SQL_IN_CHUNK,
    HOP_TIME_SEC,
)

# ── Optional new config knobs (fall back gracefully if not yet in config.py) ──
try:
    from config import HASH_LOOKUP_CHUNKS
except ImportError:
    HASH_LOOKUP_CHUNKS = 4    # number of partitions for parallel hash lookup

try:
    from config import REFINE_WORKERS
except ImportError:
    REFINE_WORKERS = 4        # threads for parallel boundary refinement


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def hhmmss_12hr(total_sec: float, custom_hour) -> str:
    total_sec = int(round(total_sec))
    m = (total_sec % 3600) // 60
    s = total_sec % 60
    h = int(custom_hour)
    suffix = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12:02d}:{m:02d}:{s:02d} {suffix}"


# ═══════════════════════════════════════════════════════════════════════════════
#  DATABASE LOADING  (parallel per language)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_language(
    db_path:       str,
    lang:          str,
    ad_id_filter:  Optional[list],
    cur:           sqlite3.Cursor,
) -> tuple[dict, dict]:
    """
    Load fingerprints and durations for a single language from an *already-open*
    SQLite cursor.  Returns (partial_idx, partial_durs).
    """
    from S2_fingerprint_db import _table_names

    files_tbl, fp_tbl = _table_names(lang)

    existing = {
        r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if fp_tbl not in existing or files_tbl not in existing:
        print(f"[WARN] Tables for language '{lang}' not found in DB — skipping.")
        return {}, {}

    if ad_id_filter:
        ph = ",".join(["?"] * len(ad_id_filter))
        fp_query   = f"SELECT hash, ad_id, time_offset FROM {fp_tbl}   WHERE ad_id IN ({ph})"
        file_query = f"SELECT ad_id, duration           FROM {files_tbl} WHERE ad_id IN ({ph})"
        params = ad_id_filter
    else:
        fp_query   = f"SELECT hash, ad_id, time_offset FROM {fp_tbl}"
        file_query = f"SELECT ad_id, duration           FROM {files_tbl}"
        params = []

    partial_idx: dict  = defaultdict(list)
    partial_durs: dict = {}

    cur.execute(fp_query, params)
    for h, ad_id, t_ad in cur:
        partial_idx[h].append((ad_id, t_ad))

    cur.execute(file_query, params)
    for ad_id, dur in cur.fetchall():
        partial_durs[ad_id] = dur

    return dict(partial_idx), partial_durs


def load_fp_index(
    db_path:         str,
    language_filter: Optional[str]  = None,
    specific_ads:    Optional[list] = None,
) -> tuple[dict, dict]:
    """
    Load fingerprint index and ad durations.

    When multiple languages are requested they are loaded concurrently, each in
    its own thread against its own SQLite connection (WAL mode allows concurrent
    readers).  Results are merged before returning.
    """
    from S2_fingerprint_db import _table_names, list_db_languages

    languages = [language_filter] if language_filter else list_db_languages(db_path)
    if not languages:
        print("[S3] No language tables found in DB.")
        return {}, {}

    print(f"[S3] Loading FP index  languages={languages}  specific_ads={specific_ads} …")

    ad_id_filter: Optional[list] = None
    if specific_ads:
        ad_id_filter = [os.path.splitext(a)[0] for a in specific_ads]

    merged_idx:  dict = defaultdict(list)
    merged_durs: dict = {}
    merge_lock        = threading.Lock()

    def _load_lang(lang: str) -> None:
        # Each thread gets its own connection — safe with SQLite WAL
        conn = sqlite3.connect(db_path, timeout=60)
        conn.execute("PRAGMA journal_mode=WAL")
        cur = conn.cursor()
        try:
            pidx, pdurs = _load_language(db_path, lang, ad_id_filter, cur)
        finally:
            conn.close()
        with merge_lock:
            for h, entries in pidx.items():
                merged_idx[h].extend(entries)
            merged_durs.update(pdurs)

    if len(languages) == 1:
        _load_lang(languages[0])
    else:
        with ThreadPoolExecutor(max_workers=len(languages)) as pool:
            futures = [pool.submit(_load_lang, lang) for lang in languages]
            for fut in as_completed(futures):
                fut.result()   # re-raise any exception

    final_idx = dict(merged_idx)
    print(f"[S3] Loaded {len(final_idx):,} unique hashes, {len(merged_durs)} ad(s).")
    return final_idx, merged_durs


def compute_ad_min_hash_time(fp_index: dict) -> dict:
    ad_min: dict = {}
    for entries in fp_index.values():
        for ad_id, t_ad in entries:
            if ad_id not in ad_min or t_ad < ad_min[ad_id]:
                ad_min[ad_id] = t_ad
    return ad_min


# ═══════════════════════════════════════════════════════════════════════════════
#  SIGNAL PROCESSING  (mirrors S2 exactly)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_peaks(y: np.ndarray, sr: int) -> list:
    S    = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    footprint = generate_binary_structure(2, 1)
    local_max = maximum_filter(S_db, size=(PEAK_NEIGH_FREQ, PEAK_NEIGH_TIME))
    peaks_mask = (S_db == local_max) & (S_db > AMP_MIN_DB)
    background = S_db <= AMP_MIN_DB
    eroded_bg  = binary_erosion(background, structure=footprint, iterations=1)
    peaks_mask = peaks_mask & ~eroded_bg
    freq_idx, time_idx = np.where(peaks_mask)
    peaks = sorted(zip(freq_idx.tolist(), time_idx.tolist()), key=lambda x: x[1])
    return peaks


def generate_hashes(peaks: list, hop_length: int = HOP_LENGTH, sr: int = SR_EXPECTED) -> list:
    hashes = []
    n = len(peaks)
    for i in range(n):
        f1, t1 = peaks[i]
        for j in range(1, FAN_VALUE + 1):
            if i + j >= n:
                break
            f2, t2 = peaks[i + j]
            dt = t2 - t1
            if dt <= 0 or dt > MAX_TIME_DELTA:
                continue
            key = f"{f1}|{f2}|{dt}"
            h   = hashlib.sha1(key.encode()).hexdigest()[:HASH_TRUNCATE]
            hashes.append((h, (t1 * hop_length) / sr))
    return hashes


# ═══════════════════════════════════════════════════════════════════════════════
#  PARALLEL HASH-LOOKUP HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _accumulate_chunk(
    chunk: list[tuple[str, float]],
    fp_index: dict,
) -> dict:
    """
    Build a partial ad_delta_acc dict for a slice of (hash, t_mix) pairs.
    Each thread processes a non-overlapping chunk — no locking needed.
    """
    local_acc: dict = defaultdict(lambda: defaultdict(list))
    for h, t_mix in chunk:
        if h not in fp_index:
            continue
        for ad_id, t_ad in fp_index[h]:
            delta   = t_mix - t_ad
            bin_key = round(delta / DELTA_BIN_SIZE) * DELTA_BIN_SIZE
            local_acc[ad_id][bin_key].append((t_mix, t_ad))
    return local_acc


def _merge_accumulators(partials: list[dict]) -> dict:
    """Merge a list of partial ad_delta_acc dicts into one."""
    merged: dict = defaultdict(lambda: defaultdict(list))
    for part in partials:
        for ad_id, dmap in part.items():
            for bin_key, matches in dmap.items():
                merged[ad_id][bin_key].extend(matches)
    return merged


# ═══════════════════════════════════════════════════════════════════════════════
#  BOUNDARY REFINEMENT  (cross-correlation)
# ═══════════════════════════════════════════════════════════════════════════════

def _xcorr_normalised(seg: np.ndarray, template: np.ndarray) -> tuple[np.ndarray, float]:
    if len(seg) < len(template) or len(template) < 1:
        return np.array([0.0]), 0.0
    corr = fftconvolve(seg, template[::-1], mode="valid")
    norm = (np.linalg.norm(seg) + 1e-10) * (np.linalg.norm(template) + 1e-10)
    corr /= norm
    return corr, float(np.max(corr))


def local_refine_start(
    mixed_y: np.ndarray, ad_y: np.ndarray, sr: int, coarse_s: float,
    margin: float = LOCAL_MARGIN,
) -> float:
    i0  = max(0, int((coarse_s - margin) * sr))
    i1  = min(len(mixed_y), int((coarse_s + margin + min(len(ad_y) / sr, margin)) * sr))
    seg = mixed_y[i0:i1]
    prefix_len = min(len(ad_y), int(margin * sr * 1.5))
    ad_pref    = ad_y[:prefix_len]
    corr, peak_val = _xcorr_normalised(seg, ad_pref)
    if peak_val < CORR_THRESHOLD:
        return coarse_s
    shift = np.argmax(corr) / float(sr)
    return (i0 / sr) + shift


def local_refine_end(
    mixed_y: np.ndarray, ad_y: np.ndarray, sr: int,
    refined_s: float, coarse_e: float,
    canonical_dur: Optional[float] = None,
    margin: float = LOCAL_MARGIN,
) -> float:
    expected_end = (refined_s + canonical_dur) if canonical_dur else coarse_e
    i0  = max(0, int((coarse_e - margin) * sr))
    i1  = min(len(mixed_y), int((expected_end + margin) * sr))
    seg = mixed_y[i0:i1]
    suffix_len = min(len(ad_y), int(margin * sr * 1.5))
    ad_suf     = ad_y[-suffix_len:]
    corr, peak_val = _xcorr_normalised(seg, ad_suf)
    if peak_val < CORR_THRESHOLD:
        return coarse_e
    shift     = np.argmax(corr) / float(sr)
    refined_e = (i0 / sr) + shift + (suffix_len / sr)
    return refined_e


def refine_boundaries(
    mixed_y: np.ndarray, ad_y: np.ndarray, sr: int,
    coarse_s: float, coarse_e: float,
    canonical_dur: Optional[float] = None,
) -> tuple[float, float]:
    s2 = local_refine_start(mixed_y, ad_y, sr, coarse_s)
    e2 = local_refine_end(mixed_y, ad_y, sr, s2, coarse_e, canonical_dur)
    if e2 <= s2:
        return s2, coarse_e
    return s2, e2


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE DETECTION  (per file — with internal parallelism)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_for_file(
    mixed_path:       str,
    fp_index:         dict,
    ad_durations:     dict,
    ad_min_hash_time: dict,
    ads_audio_folder: str,
    num_hash_chunks:  int = HASH_LOOKUP_CHUNKS,
    num_refine_workers: int = REFINE_WORKERS,
) -> list[dict]:
    """
    Detect all ad occurrences in a single mixed-audio WAV file.

    Internal parallelism
    ────────────────────
    Phase A — hash lookup & delta accumulation
        The (hash, t_mix) list is split into *num_hash_chunks* equal partitions.
        Each partition is processed by one thread (_accumulate_chunk).
        The partial dicts are merged synchronously after all threads finish.
        → Scales well when fp_index is large (many ads / long index).

    Phase B — boundary refinement
        All refine_boundaries() calls (one per merged detection) are submitted
        to a ThreadPoolExecutor of size *num_refine_workers*.
        fftconvolve() releases the GIL so threads run truly in parallel.
        → Reduces refinement time proportionally to the number of detections.
    """
    print(f"  → Scanning: {os.path.basename(mixed_path)}")

    # ── Load & resample ────────────────────────────────────────────────────────
    y, sr0 = librosa.load(mixed_path, sr=None, mono=True)
    if sr0 != SR_EXPECTED:
        y   = librosa.resample(y, orig_sr=sr0, target_sr=SR_EXPECTED)
        sr0 = SR_EXPECTED
    total_dur = librosa.get_duration(y=y, sr=sr0)

    # ── Peaks & hashes ────────────────────────────────────────────────────────
    peaks  = compute_peaks(y, sr0)
    hashes = generate_hashes(peaks)
    if not hashes:
        print(f"  [WARN] No hashes generated for {os.path.basename(mixed_path)}")
        return []

    # ── Phase A: parallel hash lookup ────────────────────────────────────────
    # Build flat list of (hash, t_mix) and partition into chunks
    chunk_size = max(1, len(hashes) // num_hash_chunks)
    chunks = [hashes[i : i + chunk_size] for i in range(0, len(hashes), chunk_size)]

    if len(chunks) == 1:
        # Single chunk — skip thread overhead
        ad_delta_acc = _accumulate_chunk(chunks[0], fp_index)
    else:
        with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
            futs = [pool.submit(_accumulate_chunk, ch, fp_index) for ch in chunks]
            partials = [fut.result() for fut in as_completed(futs)]
        ad_delta_acc = _merge_accumulators(partials)

    hop_time   = HOP_LENGTH / SR_EXPECTED
    detections = []

    for ad_id, dmap in ad_delta_acc.items():
        for bin_key, matches in dmap.items():
            if not matches:
                continue
            t_mix_vals = [m[0] for m in matches]
            t_ad_vals  = [m[1] for m in matches]
            t_min, t_max = min(t_mix_vals), max(t_mix_vals)
            dur_est  = max(t_max - t_min, hop_time)
            score    = min(1.0, len(t_mix_vals) / max(1.0, dur_est / hop_time))

            if dur_est < MIN_DETECTION_SEC or score < MIN_CONFIDENCE:
                continue

            idx0     = t_mix_vals.index(t_min)
            t_ad_min = ad_min_hash_time.get(ad_id, t_ad_vals[idx0])
            coarse_s = t_min - t_ad_min
            coarse_e = t_max + hop_time

            detections.append({
                "ad_id":    ad_id,
                "coarse_s": max(0.0, coarse_s),
                "coarse_e": min(total_dur, coarse_e),
                "dur_est":  dur_est,
                "score":    score,
            })

    # ── Merge nearby detections of the same ad ────────────────────────────────
    detections.sort(key=lambda x: (x["ad_id"], x["coarse_s"]))
    merged = []
    for det in detections:
        if (
            not merged
            or det["ad_id"] != merged[-1]["ad_id"]
            or det["coarse_s"] > merged[-1]["coarse_e"] + MERGE_GAP
        ):
            merged.append(dict(det))
        else:
            last = merged[-1]
            last["coarse_e"] = max(last["coarse_e"], det["coarse_e"])
            last["score"]    = max(last["score"],    det["score"])
            last["dur_est"]  = last["coarse_e"] - last["coarse_s"]

    if not merged:
        return []

    # ── Phase B: parallel boundary refinement ────────────────────────────────
    # Pre-load all ad WAVs referenced by the detections (deduplicated)
    needed_ads = {d["ad_id"] for d in merged}
    ad_audio: dict[str, Optional[np.ndarray]] = {}
    for ad_id in needed_ads:
        ad_path = os.path.join(ads_audio_folder, f"{ad_id}.wav")
        ad_audio[ad_id] = (
            librosa.load(ad_path, sr=SR_EXPECTED, mono=True)[0]
            if os.path.exists(ad_path) else None
        )

    # def _refine_one(det: dict) -> dict:
    #     ad_id  = det["ad_id"]
    #     ad_y   = ad_audio[ad_id]
    #     ad_dur = ad_durations.get(ad_id)
    #     if ad_y is not None:
    #         refined_s, refined_e = refine_boundaries(
    #             y, ad_y, sr0, det["coarse_s"], det["coarse_e"], canonical_dur=ad_dur
    #         )
    #     else:
    #         refined_s, refined_e = det["coarse_s"], det["coarse_e"]

    #     actual_dur = refined_e - refined_s
    #     if (
    #         ad_dur is not None
    #         and abs(actual_dur - ad_dur) < FULL_DUR_TOLERANCE
    #         and det["score"] >= FULL_SCORE_FLOOR
    #     ):
    #         typ       = "Full"
    #         final_dur = ad_dur
    #         final_e   = refined_s + ad_dur
    #     else:
    #         typ       = "Partial"
    #         final_dur = actual_dur
    #         final_e   = refined_e

    #     return {
    #         "ad_id":    ad_id,
    #         "start":    max(0.0, refined_s),
    #         "end":      min(total_dur, final_e),
    #         "duration": round(final_dur, 3),
    #         "score":    round(det["score"], 3),
    #         "type":     typ,
    #     }

    def _refine_one(det: dict) -> dict:
        ad_id  = det["ad_id"]
        ad_y   = ad_audio[ad_id]
        ad_dur = ad_durations.get(ad_id)
        
        # 1. Perform cross-correlation to find the actual signal match
        if ad_y is not None:
            refined_s, refined_e = refine_boundaries(
                y, ad_y, sr0, det["coarse_s"], det["coarse_e"], canonical_dur=ad_dur
            )
        else:
            refined_s, refined_e = det["coarse_s"], det["coarse_e"]

        # 2. Use the ACTUAL detected duration from the refinement
        actual_dur = refined_e - refined_s
        
        # 3. Determine if it's "Full" or "Partial" for labeling ONLY
        # We no longer force final_dur = ad_dur
        is_full_length = (
            ad_dur is not None 
            and abs(actual_dur - ad_dur) < FULL_DUR_TOLERANCE
        )
        is_high_score = det["score"] >= FULL_SCORE_FLOOR

        typ = "Full" if (is_full_length and is_high_score) else "Partial"

        return {
            "ad_id":    ad_id,
            "start":    max(0.0, refined_s),
            "end":      min(total_dur, refined_e), # Exact detected end time
            "duration": round(actual_dur, 3),      # Exact detected duration
            "score":    round(det["score"], 3),
            "type":     typ,
        }

    refine_workers = min(num_refine_workers, len(merged))

    if refine_workers <= 1 or len(merged) == 1:
        final = [_refine_one(d) for d in merged]
    else:
        with ThreadPoolExecutor(max_workers=refine_workers) as pool:
            futs = {pool.submit(_refine_one, d): i for i, d in enumerate(merged)}
            final = [None] * len(merged)
            for fut in as_completed(futs):
                final[futs[fut]] = fut.result()

    # ── Remove overlapping detections (keep higher score; prefer longer) ──────
    final.sort(key=lambda x: x["start"])
    out = []
    for d in final:
        if not out or d["start"] > out[-1]["end"]:
            out.append(d)
        else:
            prev = out[-1]
            if d["score"] > prev["score"] or (
                d["score"] == prev["score"] and d["duration"] > prev["duration"]
            ):
                out[-1] = d

    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  TOP-LEVEL ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def detect_ads(
    language:     Optional[str]  = None,
    channel:      Optional[str]  = None,
    date:         Optional[str]  = None,
    time_sel:     Optional[list] = None,
    specific_ads: Optional[list] = None,
    num_workers:  Optional[int]  = None,
) -> list[dict]:
    """
    Detect ads in all mixed audio files for *channel/date*.

    Outer parallelism (file-level)
    ───────────────────────────────
    Each WAV file is processed by detect_for_file() in its own thread.
    The fp_index dict is shared read-only across threads — no copying needed.

    Inner parallelism (within detect_for_file)
    ───────────────────────────────────────────
    • Hash lookup is parallelised across HASH_LOOKUP_CHUNKS thread partitions.
    • Boundary refinement is parallelised across REFINE_WORKERS threads.

    Returns a list of row-dicts ready for DataFrame construction.
    """
    mixed_audio_folder = f"Outputs/video_to_audio/{channel}/{date}"
    ads_audio_folder   = f"Outputs/ads_fingerprints/{language}"

    if not os.path.isdir(mixed_audio_folder):
        print(f"[ERROR] Mixed audio folder not found: {mixed_audio_folder}")
        return []

    fp_index, ad_durations = load_fp_index(
        DB_PATH, language_filter=language, specific_ads=specific_ads
    )
    if not fp_index:
        print("[WARN] Fingerprint index is empty — no detections possible.")
        return []

    ad_min_hash_time = compute_ad_min_hash_time(fp_index)

    # ── Select files ──────────────────────────────────────────────────────────
    all_wav = [f for f in os.listdir(mixed_audio_folder) if f.lower().endswith(".wav")]
    if time_sel:
        targets = {
            os.path.splitext(f)[0].replace(".mpd", "")
            for f in (time_sel if isinstance(time_sel, list) else [time_sel])
        }
        files = [f for f in all_wav if os.path.splitext(f)[0].replace(".mpd", "") in targets]
    else:
        files = all_wav

    if not files:
        print(f"[WARN] No WAV files found in {mixed_audio_folder} matching filter.")
        return []

    print(
        f"[S3] Scanning {len(files)} file(s) with {len(ad_durations)} ad fingerprint(s) "
        f"[hash_chunks={HASH_LOOKUP_CHUNKS}, refine_workers={REFINE_WORKERS}] …"
    )

    # ── Outer parallelism: one thread per file ────────────────────────────────
    # Each thread also runs inner parallelism (hash chunks + refinement).
    # Cap outer workers so total thread count stays reasonable:
    #   outer * (HASH_LOOKUP_CHUNKS + REFINE_WORKERS) ≤ ~2× CPU count
    cpu        = multiprocessing.cpu_count()
    inner_per  = HASH_LOOKUP_CHUNKS + REFINE_WORKERS
    max_outer  = max(1, (cpu * 2) // inner_per)
    workers    = num_workers or min(max_outer, len(files))

    def _scan(fname: str) -> list[dict]:
        return detect_for_file(
            os.path.join(mixed_audio_folder, fname),
            fp_index,
            ad_durations,
            ad_min_hash_time,
            ads_audio_folder,
        )

    file_results: list = [None] * len(files)

    if workers == 1 or len(files) == 1:
        for i, fname in enumerate(tqdm(files, desc="Scanning")):
            file_results[i] = _scan(fname)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(_scan, fname): i for i, fname in enumerate(files)}
            for future in tqdm(as_completed(future_map), total=len(files), desc="Scanning"):
                idx = future_map[future]
                try:
                    file_results[idx] = future.result()
                except Exception as exc:
                    print(f"[ERROR] {files[idx]}: {exc}")
                    file_results[idx] = []

    # ── Format output rows ────────────────────────────────────────────────────
    all_rows = []
    seg_no   = 1
    for i, dets in enumerate(file_results):
        if not dets:
            continue
        hour_prefix = os.path.splitext(files[i])[0][:2]
        for d in dets:
            all_rows.append({
                "Segment No":  seg_no,
                "Channel":     channel,
                "Date":        date,
                "Time":        hour_prefix,
                "Label":       "Ad",
                "Ad Name":     d["ad_id"],
                "Language":    language,
                "Start Time":  hhmmss_12hr(d["start"], hour_prefix),
                "End Time":    hhmmss_12hr(d["end"],   hour_prefix),
                "Duration(s)": round(d["duration"], 2),
                "Type":        d["type"],
                "Score":       d["score"],
                "SourceFile":  files[i],
            })
            seg_no += 1

    print(f"[S3] Detected {len(all_rows)} segment(s) in {channel}/{date}.")
    return all_rows

