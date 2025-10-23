#!/usr/bin/env python3
"""
build_landmarks_db.py

- Reads WAV files from ADS_AUDIO_FOLDER (assumes standardized audio from Step 1:
  mono, 16000 Hz, PCM s16, loudnorm)
- Extracts spectrogram peaks (local maxima)
- Forms landmark pairs (f1,t1) -> (f2,t2) within a target zone (fan)
- Hashes each landmark pair and stores (hash, ad_id, time_offset) in SQLite DB

Duplication removal:
- Compute a content-hash (MD5) of the audio samples (resampled to SR_EXPECTED)
  and store it in the files table. If a file with same content_hash already
  exists in DB, the file is skipped (no duplicate fingerprints).

Dependencies:
    pip install numpy scipy librosa soundfile tqdm

Usage:
    python build_landmarks_db.py
"""
import os
import sqlite3
import hashlib
from tqdm import tqdm
import numpy as np
import librosa
from scipy.ndimage import maximum_filter, binary_erosion, generate_binary_structure
import time

# ----------------- CONFIG -----------------
ADS_AUDIO_FOLDER = "Outputs/ads_fingerprints"  # input 
DB_PATH = "Outputs/DB/ads_fingerprints.db"

# Spectrogram / peak params
SR_EXPECTED = 16000      # expected sample rate from Step 1
N_FFT = 2048
HOP_LENGTH = 512
AMP_MIN_DB_ADV = -65         # peaks below this (dB rel to max) are ignored
PEAK_NEIGHBORHOOD_FREQ = 30   # frequency neighborhood size (bins)
PEAK_NEIGHBORHOOD_TIME = 15   # time neighborhood size (frames)

# Landmark pairing params
FAN_VALUE = 20           # how many neighbor peaks to pair with (Shazam uses small fan)
MAX_TIME_DELTA = 300     # max frames between peak pairs (in frames)
HASH_TRUNCATE = 20       # characters of sha1 hex to store (reduces DB size)
BATCH_INSERT_SIZE = 5000
# ------------------------------------------

def init_db(db_path):
    """
    Create tables and ensure content_hash column exists in files table.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # create tables if not exist
    cur.execute("""
        CREATE TABLE IF NOT EXISTS files (
            ad_id TEXT PRIMARY KEY,
            filename TEXT,
            duration REAL,
            content_hash TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS fingerprints (
            hash TEXT,
            ad_id TEXT,
            time_offset REAL
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hash ON fingerprints(hash)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ad ON fingerprints(ad_id)")
    conn.commit()
    conn.close()

def compute_adv_peaks(y, sr,
                  n_fft=N_FFT, hop_length=HOP_LENGTH,
                  amp_min_db=AMP_MIN_DB_ADV,
                  neighborhood_size=(PEAK_NEIGHBORHOOD_FREQ, PEAK_NEIGHBORHOOD_TIME)):
    """
    Return peaks as list of (freq_bin, time_frame)
    """
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    # local maximum filter
    footprint = generate_binary_structure(2, 1)
    neighborhood = (neighborhood_size[0], neighborhood_size[1])
    local_max = maximum_filter(S_db, size=neighborhood)
    peaks_mask = (S_db == local_max) & (S_db > amp_min_db)

    # remove background small islands
    background_mask = (S_db <= amp_min_db)
    eroded_bg = binary_erosion(background_mask, structure=footprint, iterations=1)
    peaks_mask = peaks_mask & (~eroded_bg)

    freq_idxs, time_idxs = np.where(peaks_mask)
    peaks = list(zip(freq_idxs.tolist(), time_idxs.tolist()))

    # sort peaks by time (important for pairing)
    peaks.sort(key=lambda x: x[1])
    return peaks, S_db.shape

def generate_ads_hashes_from_peaks(peaks, hop_length, sr, fan_value=FAN_VALUE, max_delta=MAX_TIME_DELTA):
    hashes = []
    n = len(peaks)
    for i in range(n):
        f1, t1 = peaks[i]
        # pair with up to fan_value following peaks within max_delta frames
        for j in range(1, fan_value + 1):
            if (i + j) >= n:
                break
            f2, t2 = peaks[i + j]
            dt = t2 - t1
            if dt <= 0 or dt > max_delta:
                continue
            key = f"{f1}|{f2}|{dt}"
            h = hashlib.sha1(key.encode("utf8")).hexdigest()[:HASH_TRUNCATE]
            time_offset_seconds = (t1 * hop_length) / sr
            hashes.append((h, time_offset_seconds))
    return hashes

def process_file(file_path):
    
    # load with target sample rate to get consistent samples for content hash
    y, sr = librosa.load(file_path, sr=SR_EXPECTED, mono=True)  # force SR_EXPECTED
    duration = librosa.get_duration(y=y, sr=sr)
    peaks, spec_shape = compute_adv_peaks(y, sr)
    hashes = generate_ads_hashes_from_peaks(peaks, hop_length=HOP_LENGTH, sr=sr)
    return hashes, duration, y

def add_hashes_to_db(db_path, ad_id, filename, duration, hashes, content_hash=None):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # upsert file record with content_hash (if provided)
    cur.execute("""
        INSERT OR REPLACE INTO files (ad_id, filename, duration, content_hash)
        VALUES (?, ?, ?, ?)
    """, (ad_id, filename, duration, content_hash))
    # insert fingerprints in batches
    to_insert = [(h, ad_id, t) for (h, t) in hashes]
    for i in range(0, len(to_insert), BATCH_INSERT_SIZE):
        batch = to_insert[i:i+BATCH_INSERT_SIZE]
        cur.executemany("INSERT INTO fingerprints (hash, ad_id, time_offset) VALUES (?, ?, ?)", batch)
        conn.commit()
    conn.close()

def compute_content_hash_from_samples(y):
    """
    Computing MD5 of 16-bit PCM representation of samples after resampling/loading.
    """
    # clip to [-1,1] then scale to int16
    y_clipped = np.clip(y, -1.0, 1.0)
    int16 = (y_clipped * 32767).astype(np.int16)
    b = int16.tobytes()
    return hashlib.md5(b).hexdigest()

def run_flow():
    init_db(DB_PATH)
    files = [f for f in os.listdir(ADS_AUDIO_FOLDER) if f.lower().endswith(".wav")]
    print(f"Found {len(files)} audio files in ads folder.")

    total_hashes = 0
    start_time = time.time()

    # open DB connection for existence checks
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    for fname in tqdm(files, desc="Processing ads"):
        path = os.path.join(ADS_AUDIO_FOLDER, fname)
        ad_id = os.path.splitext(fname)[0]  

        try:
           
            hashes, duration, y_resampled = process_file(path)
            content_hash = compute_content_hash_from_samples(y_resampled)
        except Exception as e:
            print(f"\nError processing {fname}: {e}")
            continue

        # check if content_hash already exists in DB
        cur.execute("SELECT ad_id, filename FROM files WHERE content_hash=?", (content_hash,))
        found = cur.fetchone()
        if found:
            existing_ad_id, existing_fn = found
            print(f"Skipping {fname} — already in DB as '{existing_fn}' (ad_id={existing_ad_id})")
            continue

        # not found then insert file and fingerprints
        add_hashes_to_db(DB_PATH, ad_id, fname, duration, hashes, content_hash=content_hash)
        total_hashes += len(hashes)

    conn.close()
    elapsed = time.time() - start_time
    print(f"\nDone. Total hashes stored: ~{total_hashes}. Time elapsed: {elapsed:.1f}s")
    print(f"DB saved to: {DB_PATH}")

