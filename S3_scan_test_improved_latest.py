"""
scan_and_detect_offset_end_clamp.py (Enhanced Version)
-----------------------------------------------------
Adds:
 - Channel, Date, Time (from filename)
 - Ad Name
 - Language (from ADS_AUDIO_FOLDER)
in CSV output.
"""

import os
import sqlite3
import hashlib
import time
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import librosa
import pandas as pd
from scipy.ndimage import maximum_filter, binary_erosion, generate_binary_structure
from scipy.signal import fftconvolve
import multiprocessing

# === CONFIG ===
# MIXED_AUDIO_FOLDER = "Outputs/video_to_audio/SonyTen1SD/20250918"
DB_PATH = "Outputs/DB/ads_fingerprints.db"
ADS_AUDIO_FOLDER = "Outputs/ads_fingerprints"
# CSV_OUTPUT = "Outputs/detected_ads_endclamp.csv"

SR_EXPECTED = 16000

N_FFT = 2048
HOP_LENGTH = 512
AMP_MIN_DB = -65
PEAK_NEIGH_FREQ = 30
PEAK_NEIGH_TIME = 15

FAN_VALUE = 20
MAX_TIME_DELTA = 300
HASH_TRUNCATE = 20
DELTA_BIN_SIZE = 0.2
MIN_DETECTION_SEC = 2.0
MIN_CONFIDENCE = 0.2
MERGE_GAP = 3.0

LOCAL_MARGIN = 2.0
CORR_THRESHOLD = 0.4

FULL_MATCH_RATIO = 0.93

BULK_SQL_IN_CHUNK = 4000


# === HELPERS ===
# def hhmmss(sec_f):
#     s = int(round(sec_f))
#     h = s // 3600
#     m = (s % 3600) // 60
#     s = s % 60
#     return f"{h:02d}:{m:02d}:{s:02d}"

def hhmmss_12hr(sec_f, custom_hour):
    s = int(round(sec_f))
    m = (s % 3600) // 60
    s = s % 60

    # Convert custom_hour (string like "00", "01") to int
    h = int(custom_hour)
    
    # 12-hour format
    suffix = "AM" if h < 12 else "PM"
    h_12 = h % 12
    if h_12 == 0:
        h_12 = 12  # 12 AM or 12 PM

    return f"{h_12:02d}:{m:02d}:{s:02d} {suffix}"



def extract_channel_date(mixed_folder):
    """
    Extracts channel and date from path like:
    Outputs/video_to_audio/SonyTen1SD/20250918
    """
    parts = mixed_folder.strip("/").split("/")
    if len(parts) >= 3:
        channel = parts[-2]
        date = parts[-1]
    else:
        channel = "Unknown"
        date = "Unknown"
    return channel, date


def load_fp_index(db_path, language_filter=None):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    print("Loading fingerprint index...")

    if language_filter:
        cur.execute("SELECT hash, ad_id, time_offset, language FROM fingerprints WHERE language=?", (language_filter,))
    else:
        cur.execute("SELECT hash, ad_id, time_offset, language FROM fingerprints")

    idx = defaultdict(list)
    for h, ad_id, t_ad, lang in cur:
        idx[h].append((ad_id, t_ad))

    if language_filter:
        cur.execute("SELECT ad_id, duration FROM files WHERE language=?", (language_filter,))
    else:
        cur.execute("SELECT ad_id, duration FROM files")

    ad_durs = {ad_id: dur for ad_id, dur in cur.fetchall()}
    conn.close()

    print(f"Loaded {len(idx)} unique hashes, {len(ad_durs)} ads for language: {language_filter or 'ALL'}")
    return idx, ad_durs


def compute_ad_min_hash_time(fp_index):
    ad_min = {}
    for h, entries in fp_index.items():
        for ad_id, t_ad in entries:
            if (ad_id not in ad_min) or (t_ad < ad_min[ad_id]):
                ad_min[ad_id] = t_ad
    return ad_min


def compute_peaks(y, sr):
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    footprint = generate_binary_structure(2, 1)
    local_max = maximum_filter(S_db, size=(PEAK_NEIGH_FREQ, PEAK_NEIGH_TIME))
    peaks_mask = (S_db == local_max) & (S_db > AMP_MIN_DB)
    background = (S_db <= AMP_MIN_DB)
    eroded_bg = binary_erosion(background, structure=footprint, iterations=1)
    peaks_mask = peaks_mask & (~eroded_bg)
    freq_idxs, time_idxs = np.where(peaks_mask)
    peaks = list(zip(freq_idxs.tolist(), time_idxs.tolist()))
    peaks.sort(key=lambda x: x[1])
    return peaks


def generate_hashes(peaks, hop_length=HOP_LENGTH, sr=SR_EXPECTED):
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
            h = hashlib.sha1(key.encode("utf8")).hexdigest()[:HASH_TRUNCATE]
            t_off = (t1 * hop_length) / sr
            hashes.append((h, t_off))
    return hashes


def local_refine_start(mixed_y, ad_y, sr, coarse_s, coarse_e, margin=LOCAL_MARGIN):
    start0 = max(0.0, coarse_s - margin)
    end0 = min(len(mixed_y)/sr, coarse_s + margin + min(len(ad_y)/sr, margin))
    i0 = int(start0 * sr)
    i1 = int(end0 * sr)
    seg = mixed_y[i0:i1]
    if len(seg) < 1 or len(ad_y) < 1:
        return coarse_s
    prefix_len = min(len(ad_y), int(margin * sr * 1.5))
    ad_pref = ad_y[:prefix_len]
    corr = fftconvolve(seg, ad_pref[::-1], mode="valid")
    norm_seg = np.linalg.norm(seg) + 1e-10
    norm_pref = np.linalg.norm(ad_pref) + 1e-10
    corr = corr / (norm_seg * norm_pref)
    peak_idx = np.argmax(corr)
    peak_val = corr[peak_idx]
    if peak_val < CORR_THRESHOLD:
        return coarse_s
    shift = peak_idx / float(sr)
    return start0 + shift


def local_refine_end(mixed_y, ad_y, sr, coarse_s, coarse_e, margin=LOCAL_MARGIN):
    start0 = max(0.0, coarse_e - margin - min(len(ad_y)/sr, margin))
    end0 = min(len(mixed_y)/sr, coarse_e + margin)
    i0 = int(start0 * sr)
    i1 = int(end0 * sr)
    seg = mixed_y[i0:i1]
    if len(seg) < 1 or len(ad_y) < 1:
        return coarse_e
    suffix_len = min(len(ad_y), int(margin * sr * 1.5))
    ad_suf = ad_y[-suffix_len:]
    corr = fftconvolve(seg, ad_suf[::-1], mode="valid")
    norm_seg = np.linalg.norm(seg) + 1e-10
    norm_suf = np.linalg.norm(ad_suf) + 1e-10
    corr = corr / (norm_seg * norm_suf)
    peak_idx = np.argmax(corr)
    peak_val = corr[peak_idx]
    if peak_val < CORR_THRESHOLD:
        return coarse_e
    shift = peak_idx / float(sr)
    return start0 + shift


def refine_boundaries(mixed_y, ad_y, sr, coarse_s, coarse_e, t_max_mix, canonical_dur=None):
    s2 = local_refine_start(mixed_y, ad_y, sr, coarse_s, coarse_e)
    expected_span = coarse_e - coarse_s
    tentative_e = s2 + expected_span
    e2 = local_refine_end(mixed_y, ad_y, sr, s2, tentative_e)
    if canonical_dur is not None:
        if abs((e2 - s2) - canonical_dur) < (canonical_dur * 0.25):
            e2 = s2 + canonical_dur
    hop_time = HOP_LENGTH / SR_EXPECTED
    max_bound = t_max_mix + hop_time + LOCAL_MARGIN
    if e2 > max_bound and e2 > coarse_e:
        e2 = max(max_bound, coarse_e)
    if e2 <= s2:
        return coarse_s, coarse_e
    return s2, e2


def detect_for_file(mixed_path, fp_index, ad_durations, ad_min_hash_time):
    print(f"\nProcessing: {os.path.basename(mixed_path)}")
    y, sr0 = librosa.load(mixed_path, sr=None, mono=True)
    if sr0 != SR_EXPECTED:
        y = librosa.resample(y, orig_sr=sr0, target_sr=SR_EXPECTED)
        sr0 = SR_EXPECTED
    total_dur = librosa.get_duration(y=y, sr=sr0)

    peaks = compute_peaks(y, sr0)
    hashes = generate_hashes(peaks, hop_length=HOP_LENGTH, sr=sr0)
    if not hashes:
        print("No hashes generated.")
        return []

    hash_to_t = defaultdict(list)
    for h, t in hashes:
        hash_to_t[h].append(t)

    ad_delta_acc = defaultdict(lambda: defaultdict(list))
    for h, t_list in hash_to_t.items():
        if h not in fp_index:
            continue
        for (ad_id, t_ad) in fp_index[h]:
            for t_mix in t_list:
                delta = t_mix - t_ad
                bin_key = round(delta / DELTA_BIN_SIZE) * DELTA_BIN_SIZE
                ad_delta_acc[ad_id][bin_key].append((t_mix, t_ad))

    detections = []
    hop_time = HOP_LENGTH / SR_EXPECTED
    for ad_id, dmap in ad_delta_acc.items():
        for bin_key, matches in dmap.items():
            if not matches:
                continue
            t_mix_vals = [m[0] for m in matches]
            t_ad_vals = [m[1] for m in matches]
            t_min = min(t_mix_vals)
            t_max = max(t_mix_vals)
            dur_est = max(t_max - t_min, hop_time)
            matched = len(t_mix_vals)
            expected_slots = max(1.0, dur_est / hop_time)
            score = min(1.0, matched / expected_slots)
            if dur_est >= MIN_DETECTION_SEC and score >= MIN_CONFIDENCE:
                idx0 = t_mix_vals.index(t_min)
                t_mix_ear = t_mix_vals[idx0]
                t_ad_min = ad_min_hash_time.get(ad_id, t_ad_vals[idx0])
                coarse_s = t_mix_ear - t_ad_min
                coarse_e = t_max + hop_time
                canon = ad_durations.get(ad_id)
                if canon is not None:
                    if (t_max - t_min) >= (0.9 * canon):
                        coarse_e = min(coarse_s + canon, coarse_e)
                detections.append({
                    "ad_id": ad_id,
                    "coarse_s": coarse_s,
                    "coarse_e": coarse_e,
                    "t_max_mix": t_max,
                    "dur_est": dur_est,
                    "score": score
                })

    detections.sort(key=lambda x: (x["ad_id"], x["coarse_s"]))
    merged = []
    for det in detections:
        if not merged:
            merged.append(det)
        else:
            last = merged[-1]
            if det["ad_id"] == last["ad_id"] and det["coarse_s"] <= last["coarse_e"] + MERGE_GAP:
                new_s = min(last["coarse_s"], det["coarse_s"])
                new_e = max(last["coarse_e"], det["coarse_e"])
                new_score = max(last["score"], det["score"])
                new_tmax = max(last.get("t_max_mix", 0.0), det.get("t_max_mix", 0.0))
                merged[-1] = {
                    "ad_id": last["ad_id"],
                    "coarse_s": new_s,
                    "coarse_e": new_e,
                    "t_max_mix": new_tmax,
                    "dur_est": new_e - new_s,
                    "score": new_score
                }
            else:
                merged.append(det)

    final = []
    y_full = y
    for det in merged:
        ad_id = det["ad_id"]
        cs, ce = det["coarse_s"], det["coarse_e"]
        tmax_mix = det.get("t_max_mix", ce)
        ad_path = os.path.join(ADS_AUDIO_FOLDER, f"{ad_id}.wav")
        if os.path.exists(ad_path):
            ad_y, _ = librosa.load(ad_path, sr=SR_EXPECTED, mono=True)
        else:
            ad_y = None

        if ad_y is not None:
            refined_s, refined_e = refine_boundaries(y_full, ad_y, sr0, cs, ce, tmax_mix, canonical_dur=ad_durations.get(ad_id))
        else:
            refined_s, refined_e = cs, ce

        dur = max(0.0, refined_e - refined_s)
        typ = "Partial"
        ad_dur = ad_durations.get(ad_id)
        if ad_dur is not None and dur >= FULL_MATCH_RATIO * ad_dur:
            typ = "Full"
            refined_e = refined_s + ad_dur
            dur = ad_dur

        final.append({
            "ad_id": ad_id,
            "start": refined_s,
            "end": refined_e,
            "duration": dur,
            "score": round(det["score"], 3),
            "type": typ
        })

    final = sorted(final, key=lambda x: x["start"])
    out = []
    for d in final:
        if not out:
            out.append(d)
        else:
            last = out[-1]
            if d["start"] <= last["end"]:
                if d["score"] > last["score"]:
                    out[-1] = d
                elif d["score"] == last["score"] and d["duration"] > last["duration"]:
                    out[-1] = d
            else:
                out.append(d)
    return out
        

def _detect_worker(args):
    """Worker function for multiprocessing pool."""
    mixed_path, fp_index, ad_durations, ad_min_hash_time = args
    # detect_for_file handles the core logic
    return detect_for_file(mixed_path, fp_index, ad_durations, ad_min_hash_time)


def detect_ads(language=None, channel=None, date=None, time_sel=None, num_workers=None): # ADD num_workers
    """
    Detect ads in converted mixed audio using the fingerprint DB.
    Uses multiprocessing to speed up file scanning.

    Args:
        language (str): Language tag for ad samples.
        channel (str): Channel name.
        date (str): Date string.
        time_sel (str or list or None): A single filename string (e.g., '1200.mpd.mp4'),
                                        a list of filename strings, or None to process all files.
        num_workers (int or None): The number of worker processes to use.
    
    Returns:
        list: A list of dictionaries representing detected ad segments.
    """
    global ADS_AUDIO_FOLDER

    MIXED_AUDIO_FOLDER = f"Outputs/video_to_audio/{channel}/{date}"
    ADS_AUDIO_FOLDER   = f"Outputs/ads_fingerprints/{language}"

    print(f"\n[INFO] Detecting ads for channel={channel}, date={date}, time_sel={time_sel}, language={language}")
    print(f"[INFO] Using mixed audio folder: {MIXED_AUDIO_FOLDER}")

    # Assuming load_fp_index, DB_PATH, compute_ad_min_hash_time, _detect_worker, 
    # and hhmmss_12hr are defined elsewhere and accessible.
    fp_index, ad_durations = load_fp_index(DB_PATH, language_filter=language)
    ad_min_hash_time = compute_ad_min_hash_time(fp_index)

    all_rows = []
    
    # --- File Discovery and Filtering ---
    
    # Get all .wav files in the mixed audio folder
    all_files = [
        f for f in os.listdir(MIXED_AUDIO_FOLDER)
        if f.lower().endswith(".wav")
    ]

    files_to_process = all_files

    if time_sel is not None:
        # Normalize time_sel to a list if it's a single string
        if isinstance(time_sel, str):
            target_files = [time_sel]
        elif isinstance(time_sel, list):
            target_files = time_sel
        else:
            print(f"[WARNING] Invalid type for time_sel: {type(time_sel)}. Processing all files.")
            target_files = [] # This will effectively stop the filtering below

        if target_files:
            # Create a set of base names (e.g., '1200.mpd.mp4' -> '1200') for efficient lookup
            # We strip the extension and any ".mpd" tag to match the format used in 'files'
            target_basenames = {
                os.path.splitext(f)[0].replace(".mpd", "") for f in target_files
            }
            
            # Filter the list of all .wav files
            files_to_process = [
                f for f in all_files 
                if os.path.splitext(f)[0].replace(".mpd", "") in target_basenames
            ]

    if not files_to_process:
        print(f"[INFO] No matching audio files found for {channel} {date} {time_sel}")
        return []

    # Rename variable for clarity
    files = files_to_process 

    # --- Setup Parallel Processing ---
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
        
    print(f"[INFO] Using {num_workers} worker processes for detection on {len(files)} files.")
    
    # Prepare arguments for the worker function
    all_args = []
    for fname in files:
        mixed_path = os.path.join(MIXED_AUDIO_FOLDER, fname)
        # Note: fp_index, ad_durations, ad_min_hash_time are shared across processes
        all_args.append((mixed_path, fp_index, ad_durations, ad_min_hash_time))

    
    # Execute detection in parallel
    results = []
    with multiprocessing.Pool(processes=num_workers) as pool:
        # pool.imap is better for long-running tasks as it returns results as soon as they are ready
        # tqdm is used to wrap the pool results for progress tracking
        for dets in tqdm(pool.imap(_detect_worker, all_args), total=len(all_args), desc="Mixed files (Parallel)"):
            results.append(dets)
            
    # --- Aggregate Results ---
    seg_no = 1
    for i, dets in enumerate(results):
        fname = files[i]  # Get the original filename
        
        # Ensure the filename is used to extract the hour (e.g., '12' from '1200.mpd.wav')
        # This handles cases like '1200.mpd.wav' or just '1200.wav'
        # Split by '.', take the first part, and take the first two characters.
        initial_digits = os.path.splitext(fname)[0][:2] 
        
        for d in dets:
            all_rows.append({
                "Segment No": seg_no,
                "Channel": channel,
                "Date": date,
                "Time": initial_digits,
                "Label": "Ad",
                "Ad Name": d["ad_id"],
                "Language": language,
                "Start Time": hhmmss_12hr(d["start"], initial_digits),
                "End Time": hhmmss_12hr(d["end"], initial_digits),
                "Duration(s)": round(d["duration"], 3),
                "Avg_Score": d["score"],
                "Type": d["type"],
                "SourceFile": fname
            })
            seg_no += 1

    if not all_rows:
        print(f"No ads detected for {channel} {date} ({language}).")

    return all_rows