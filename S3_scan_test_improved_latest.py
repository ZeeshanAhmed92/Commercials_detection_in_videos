#!/usr/bin/env python3
"""
scan_and_detect_offset_end_clamp.py

Uses:
 - Offset correction via earliest hash offset (ad_min_hash_time)
 - End bounding via matched span + optional canonical extension
 - Local start/end refinement, with end clamped not to exceed matched span
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

# === CONFIG — adjust thresholds as needed ===
MIXED_AUDIO_FOLDER = "Outputs/video_to_audio"
DB_PATH = "Outputs/DB/ads_fingerprints.db"
ADS_AUDIO_FOLDER = "Outputs/ads_fingerprints"
CSV_OUTPUT = "Outputs/detected_ads_endclamp.csv"

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

FULL_MATCH_RATIO = 0.9

BULK_SQL_IN_CHUNK = 4000

def hhmmss(sec_f):
    s = int(round(sec_f))
    h = s // 3600
    m = (s % 3600) // 60
    s = s % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def load_fp_index(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    print("Loading fingerprint index...")
    cur.execute("SELECT hash, ad_id, time_offset FROM fingerprints")
    idx = defaultdict(list)
    for h, ad_id, t_ad in cur:
        idx[h].append((ad_id, t_ad))
    cur.execute("SELECT ad_id, duration FROM files")
    ad_durs = {}
    for ad_id, dur in cur:
        ad_durs[ad_id] = dur
    conn.close()
    print(f"Loaded {len(idx)} unique hashes, {len(ad_durs)} ads")
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
    """
    Refine start & end, with emphasis on local correlation.
    The end clamping based on t_max_mix is REMOVED to prioritize correlation,
    unless the canonical duration is used.
    """
    s2 = local_refine_start(mixed_y, ad_y, sr, coarse_s, coarse_e)
    expected_span = coarse_e - coarse_s
    tentative_e = s2 + expected_span
    e2 = local_refine_end(mixed_y, ad_y, sr, s2, tentative_e)

    # Optionally clamp to canonical if detection seems full
    if canonical_dur is not None:
        if abs((e2 - s2) - canonical_dur) < (canonical_dur * 0.25):
            e2 = s2 + canonical_dur
            
    # Add a safety net based on t_max_mix, but only if e2 is far off.
    # This acts as a fallback if local_refine_end gave a bad result.
    hop_time = HOP_LENGTH / SR_EXPECTED
    max_bound = t_max_mix + hop_time + LOCAL_MARGIN # Allow a little margin
    if e2 > max_bound and e2 > coarse_e:
        e2 = max(max_bound, coarse_e) # Clamp, but not shorter than coarse_e
    
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
                # Use offset correction: t_ad_min
                idx0 = t_mix_vals.index(t_min)
                t_mix_ear = t_mix_vals[idx0]
                t_ad_min = ad_min_hash_time.get(ad_id, t_ad_vals[idx0])
                coarse_s = t_mix_ear - t_ad_min
                # print(f"Ad {ad_id}, bin_key {bin_key:.3f}, t_mix_ear={t_mix_ear:.3f}, t_ad_min={t_ad_min:.3f}, coarse_s={coarse_s:.3f}")

                # For end, do not always extend full canonical
                coarse_e = t_max + hop_time
                # If span covers a large fraction of canonical, allow extension
                canon = ad_durations.get(ad_id)
                if canon is not None:
                    if (t_max - t_min) >= (0.8 * canon):
                        coarse_e = min(coarse_s + canon, coarse_e)

                detections.append({
                    "ad_id": ad_id,
                    "coarse_s": coarse_s,
                    "coarse_e": coarse_e,
                    "t_max_mix": t_max,
                    "dur_est": dur_est,
                    "score": score
                })

    # Merge same-ad detections
    detections.sort(key=lambda x: (x["ad_id"], x["coarse_s"]))
    merged = []
    for det in detections:
        if not merged:
            merged.append(det)
        else:
            last = merged[-1]
            if det["ad_id"] == last["ad_id"] and det["coarse_s"] <= last["coarse_e"] + MERGE_GAP:
                # Merge spans
                new_s = min(last["coarse_s"], det["coarse_s"])
                new_e = max(last["coarse_e"], det["coarse_e"])
                new_score = max(last["score"], det["score"])
                # For merged, t_max_mix = max of underlying
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

        final.append({
            "ad_id": ad_id,
            "start": refined_s,
            "end": refined_e,
            "duration": dur,
            "score": round(det["score"], 3),
            "type": typ
        })

    # Resolve overlapping detections
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

def detect_ads():
    fp_index, ad_durations = load_fp_index(DB_PATH)
    ad_min_hash_time = compute_ad_min_hash_time(fp_index)

    all_rows = []
    seg_no = 1
    files = [f for f in os.listdir(MIXED_AUDIO_FOLDER) if f.lower().endswith(".wav")]
    for fname in tqdm(files, desc="Mixed files"):
        dets = detect_for_file(os.path.join(MIXED_AUDIO_FOLDER, fname), fp_index, ad_durations, ad_min_hash_time)
        for d in dets:
            all_rows.append({
                "Segment No": seg_no,
                "Label": "Ad",
                "Start Time": hhmmss(d["start"]),
                "End Time": hhmmss(d["end"]),
                "Duration(s)": round(d["duration"], 3),
                "Ad_ID": d["ad_id"],
                "Avg_Score": d["score"],
                "Type": d["type"],
                "SourceFile": fname
            })
            seg_no += 1

    if not all_rows:
        print("No ads detected.")
        return
    df = pd.DataFrame(all_rows)
    df.sort_values(by=["SourceFile", "Start Time"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["Segment No"] = df.index + 1
    df.to_csv(CSV_OUTPUT, index=False)
    print("\nDetected Ads:\n", df.to_string(index=False))
    print(f"Saved to {CSV_OUTPUT}")


