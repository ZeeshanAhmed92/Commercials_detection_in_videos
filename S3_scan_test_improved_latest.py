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
DB_PATH = "Outputs/DB/ads_fingerprints.db"
ADS_AUDIO_FOLDER = "Outputs/ads_fingerprints"

SR_EXPECTED = 16000

N_FFT = 2048
HOP_LENGTH = 512
# AMP_MIN_DB = -65
# PEAK_NEIGH_FREQ = 30
# PEAK_NEIGH_TIME = 15
AMP_MIN_DB = -65
PEAK_NEIGH_FREQ = 20
PEAK_NEIGH_TIME = 10

FAN_VALUE = 20
MAX_TIME_DELTA = 200
HASH_TRUNCATE = 20
DELTA_BIN_SIZE = 0.02
MIN_DETECTION_SEC = 0.6
MIN_CONFIDENCE = 0.7
MERGE_GAP = 3.0

# LOCAL_MARGIN = 2.0
LOCAL_MARGIN = 2.0
CORR_THRESHOLD = 0.5
# CORR_THRESHOLD = 0.3

# FULL_MATCH_RATIO = 0.85
FULL_MATCH_RATIO = 0.90

BULK_SQL_IN_CHUNK = 5000



# === HELPERS ===

def hhmmss_12hr(total_sec, custom_hour):
    """Simple HH:MM:SS AM/PM converter"""
    total_sec = int(total_sec)
    m = (total_sec % 3600) // 60
    s = total_sec % 60
    
    h = int(custom_hour)
    suffix = "AM" if h < 12 else "PM"
    h_12 = h % 12
    if h_12 == 0: h_12 = 12
    
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


def load_fp_index(db_path, language_filter=None, specific_ads=None):
    """
    Modified to support filtering by a specific list of ad IDs.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    print(f"Loading fingerprint index (lang={language_filter}, specific_ads={specific_ads})...")

    # Base Queries
    fp_query = "SELECT hash, ad_id, time_offset, language FROM fingerprints WHERE 1=1"
    file_query = "SELECT ad_id, duration FROM files WHERE 1=1"
    params = []

    if language_filter:
        fp_query += " AND language=?"
        file_query += " AND language=?"
        params.append(language_filter)

    if specific_ads:
        # specific_ads might contain filenames like 'ad1.mp4', we need ad_id 'ad1'
        ad_ids = [os.path.splitext(a)[0] for a in specific_ads]
        placeholders = ",".join(["?"] * len(ad_ids))
        fp_query += f" AND ad_id IN ({placeholders})"
        file_query += f" AND ad_id IN ({placeholders})"
        params.extend(ad_ids)

    # Load Fingerprints
    idx = defaultdict(list)
    cur.execute(fp_query, params)
    for h, ad_id, t_ad, lang in cur:
        idx[h].append((ad_id, t_ad))

    # Load Durations
    cur.execute(file_query, params)
    ad_durs = {ad_id: dur for ad_id, dur in cur.fetchall()}
    
    conn.close()
    print(f"Loaded {len(idx)} unique hashes, {len(ad_durs)} ads matched.")
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

def local_refine_end(mixed_y, ad_y, sr, coarse_s, coarse_e, margin=LOCAL_MARGIN, canonical_dur=None):
    """
    Refines the end boundary of a detection using cross-correlation.
    If canonical_dur is provided, it extends the search window to verify 
    if the ad played to its full length.
    """
    # 1. Determine the search window. 
    # If we know the ad's expected length, look around that expected end point.
    if canonical_dur is not None:
        # The expected end is 'start + full length'
        expected_end = coarse_s + canonical_dur
        # Search from slightly before the coarse end to slightly after the expected end
        start0 = max(0.0, coarse_e - margin)
        end0 = min(len(mixed_y)/sr, expected_end + margin)
    else:
        # Standard search window around the coarse (fingerprint-based) end
        start0 = max(0.0, coarse_e - margin)
        end0 = min(len(mixed_y)/sr, coarse_e + margin)

    i0 = int(start0 * sr)
    i1 = int(end0 * sr)
    seg = mixed_y[i0:i1]

    if len(seg) < 1 or len(ad_y) < 1:
        return coarse_e

    # 2. Use the "tail" of the ad sample for correlation (e.g., last 3 seconds or margin * 1.5)
    suffix_len = min(len(ad_y), int(margin * sr * 1.5))
    ad_suf = ad_y[-suffix_len:]

    # 3. Perform cross-correlation to find the best alignment
    corr = fftconvolve(seg, ad_suf[::-1], mode="valid")
    
    # Normalize the correlation to get a value between 0 and 1
    norm_seg = np.linalg.norm(seg) + 1e-10
    norm_suf = np.linalg.norm(ad_suf) + 1e-10
    corr = corr / (norm_seg * norm_suf)

    peak_idx = np.argmax(corr)
    peak_val = corr[peak_idx]

    # 4. Evidence-based return: Only update if we find a strong match
    # If peak_val is low, it means the ad likely didn't play through that section.
    if peak_val < CORR_THRESHOLD:
        return coarse_e

    # Calculate the timestamp of the best match
    shift = peak_idx / float(sr)
    refined_e = start0 + shift + (suffix_len / float(sr))
    
    return refined_e


def refine_boundaries(mixed_y, ad_y, sr, coarse_s, coarse_e, t_max_mix, canonical_dur=None):
    # First, find the best start point
    s2 = local_refine_start(mixed_y, ad_y, sr, coarse_s, coarse_e)
    
    # Then, find the best end point using the dynamic search range
    e2 = local_refine_end(mixed_y, ad_y, sr, s2, coarse_e, canonical_dur=canonical_dur)
    
    # Safety check: ensure end is after start
    if e2 <= s2:
        return s2, coarse_e
        
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
        return []

    hash_to_t = defaultdict(list)
    for h, t in hashes:
        hash_to_t[h].append(t)

    ad_delta_acc = defaultdict(lambda: defaultdict(list))
    for h, t_list in hash_to_t.items():
        if h not in fp_index: continue
        for (ad_id, t_ad) in fp_index[h]:
            for t_mix in t_list:
                delta = t_mix - t_ad
                bin_key = round(delta / DELTA_BIN_SIZE) * DELTA_BIN_SIZE
                ad_delta_acc[ad_id][bin_key].append((t_mix, t_ad))

    detections = []
    hop_time = HOP_LENGTH / SR_EXPECTED
    for ad_id, dmap in ad_delta_acc.items():
        for bin_key, matches in dmap.items():
            if not matches: continue
            t_mix_vals = [m[0] for m in matches]
            t_ad_vals = [m[1] for m in matches]
            t_min, t_max = min(t_mix_vals), max(t_mix_vals)
            dur_est = max(t_max - t_min, hop_time)
            score = min(1.0, len(t_mix_vals) / max(1.0, dur_est / hop_time))
            
            if dur_est >= MIN_DETECTION_SEC and score >= MIN_CONFIDENCE:
                idx0 = t_mix_vals.index(t_min)
                t_ad_min = ad_min_hash_time.get(ad_id, t_ad_vals[idx0])
                coarse_s = t_min - t_ad_min
                coarse_e = t_max + hop_time
                
                detections.append({
                    "ad_id": ad_id, "coarse_s": coarse_s, "coarse_e": coarse_e,
                    "t_max_mix": t_max, "dur_est": dur_est, "score": score
                })

    detections.sort(key=lambda x: (x["ad_id"], x["coarse_s"]))
    merged = []
    for det in detections:
        if not merged: merged.append(det)
        else:
            last = merged[-1]
            if det["ad_id"] == last["ad_id"] and det["coarse_s"] <= last["coarse_e"] + MERGE_GAP:
                last["coarse_e"] = max(last["coarse_e"], det["coarse_e"])
                last["t_max_mix"] = max(last["t_max_mix"], det["t_max_mix"])
                last["score"] = max(last["score"], det["score"])
                last["dur_est"] = last["coarse_e"] - last["coarse_s"]
            else: merged.append(det)

    final = []
    for det in merged:
        ad_id = det["ad_id"]
        ad_path = os.path.join(ADS_AUDIO_FOLDER, f"{ad_id}.wav")
        ad_y = librosa.load(ad_path, sr=SR_EXPECTED, mono=True)[0] if os.path.exists(ad_path) else None
        ad_dur = ad_durations.get(ad_id)

        # Refine boundaries with Cross-Correlation
        if ad_y is not None:
            refined_s, refined_e = refine_boundaries(y, ad_y, sr0, det["coarse_s"], det["coarse_e"], det["t_max_mix"], canonical_dur=ad_dur)
        else:
            refined_s, refined_e = det["coarse_s"], det["coarse_e"]

        # STRICT DURATION & TYPE CHECK
        actual_dur = refined_e - refined_s
        
        # Logic: If it's a 20s ad but we only detected 6s, it CANNOT be "Full"
        # We require it to be within 0.8s of the reference length to be "Full"
        if ad_dur is not None and abs(actual_dur - ad_dur) < 0.8 and det["score"] > 0.8:
            typ = "Full"
            final_dur = ad_dur
            final_e = refined_s + ad_dur
        else:
            typ = "Partial"
            final_dur = actual_dur
            final_e = refined_e

        final.append({
            "ad_id": ad_id, "start": refined_s, "end": final_e,
            "duration": final_dur, "score": round(det["score"], 3), "type": typ
        })

    # Overlap removal
    final = sorted(final, key=lambda x: x["start"])
    out = []
    for d in final:
        if not out: out.append(d)
        else:
            if d["start"] <= out[-1]["end"]:
                if d["score"] > out[-1]["score"]: out[-1] = d
            else: out.append(d)
    return out
        

def _detect_worker(args):
    return detect_for_file(*args)

def detect_ads(language=None, channel=None, date=None, time_sel=None, specific_ads=None, num_workers=None):
    global ADS_AUDIO_FOLDER
    MIXED_AUDIO_FOLDER = f"Outputs/video_to_audio/{channel}/{date}"
    ADS_AUDIO_FOLDER = f"Outputs/ads_fingerprints/{language}"

    fp_index, ad_durations = load_fp_index(DB_PATH, language_filter=language, specific_ads=specific_ads)
    ad_min_hash_time = compute_ad_min_hash_time(fp_index)

    all_files = [f for f in os.listdir(MIXED_AUDIO_FOLDER) if f.lower().endswith(".wav")]
    if time_sel:
        target_basenames = {os.path.splitext(f)[0].replace(".mpd", "") for f in (time_sel if isinstance(time_sel, list) else [time_sel])}
        files = [f for f in all_files if os.path.splitext(f)[0].replace(".mpd", "") in target_basenames]
    else:
        files = all_files

    if not files or not fp_index: return []

    num_workers = num_workers or multiprocessing.cpu_count()
    all_args = [(os.path.join(MIXED_AUDIO_FOLDER, f), fp_index, ad_durations, ad_min_hash_time) for f in files]
    
    results = []
    with multiprocessing.Pool(processes=num_workers) as pool:
        for dets in tqdm(pool.imap(_detect_worker, all_args), total=len(all_args), desc="Scanning"):
            results.append(dets)

    all_rows = []
    seg_no = 1
    for i, dets in enumerate(results):
        initial_digits = os.path.splitext(files[i])[0][:2]
        
        for d in dets:
            # 1. Use high-precision floats for the math
            start_val = d["start"]
            duration_val = d["duration"]
            end_val = start_val + duration_val
            
            # 2. Format to HH:MM:SS
            # We round to the nearest second for the display string
            start_str = hhmmss_12hr(round(start_val), initial_digits)
            end_str = hhmmss_12hr(round(end_val), initial_digits)
            
            all_rows.append({
                "Segment No": seg_no,
                "Channel": channel,
                "Date": date,
                "Time": initial_digits,
                "Label": "Ad",
                "Ad Name": d["ad_id"],
                "Language": language,
                "Start Time": start_str,
                "End Time": end_str,
                "Duration(s)": round(duration_val, 2),
                "Type": d["type"],
                "SourceFile": files[i]
            })
            seg_no += 1

    return all_rows