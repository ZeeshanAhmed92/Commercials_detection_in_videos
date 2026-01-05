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
AMP_MIN_DB_ADV = -75        # peaks below this (dB rel to max) are ignored
PEAK_NEIGHBORHOOD_FREQ = 20   # frequency neighborhood size (bins)
PEAK_NEIGHBORHOOD_TIME = 10   # time neighborhood size (frames)

# Landmark pairing params
FAN_VALUE = 20          # how many neighbor peaks to pair with (Shazam uses small fan)
MAX_TIME_DELTA = 300     # max frames between peak pairs (in frames)
HASH_TRUNCATE = 20       # characters of sha1 hex to store (reduces DB size)
BATCH_INSERT_SIZE = 4000
# ------------------------------------------

def init_db(db_path):
    """
    Create tables and ensure required columns exist (including 'language').
    """
    conn = sqlite3.connect(db_path, timeout=30)
    cur = conn.cursor()

    # create tables if not exist
    cur.execute("""
        CREATE TABLE IF NOT EXISTS files (
            ad_id TEXT PRIMARY KEY,
            filename TEXT,
            duration REAL,
            content_hash TEXT,
            language TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS fingerprints (
            hash TEXT,
            ad_id TEXT,
            time_offset REAL,
            language TEXT
        )
    """)

    # ensure columns exist for backward compatibility
    for table, column in [("files", "language"), ("fingerprints", "language")]:
        try:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} TEXT")
        except sqlite3.OperationalError:
            # column already exists
            pass

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

def add_hashes_to_db(db_path, ad_id, filename, duration, hashes, content_hash=None, language=None):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # upsert file record with content_hash (if provided)
    cur.execute("""
        INSERT OR REPLACE INTO files (ad_id, filename, duration, content_hash, language)
        VALUES (?, ?, ?, ?, ?)
    """, (ad_id, filename, duration, content_hash, language))

    # insert fingerprints in batches (with language)
    to_insert = [(h, ad_id, t, language) for (h, t) in hashes]
    for i in range(0, len(to_insert), BATCH_INSERT_SIZE):
        batch = to_insert[i:i+BATCH_INSERT_SIZE]
        cur.executemany(
            "INSERT INTO fingerprints (hash, ad_id, time_offset, language) VALUES (?, ?, ?, ?)",
            batch
        )
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

def run_flow(ads_language=None, specific_ads=None):
    """
    ads_language: folder name inside ADS_AUDIO_FOLDER (e.g. "Hindi")
    target_ads: list of specific filenames (e.g. ["ad1.mp4", "ad2.mp4"]) or None
    """
    init_db(DB_PATH)

    # 1. Determine which specific language folder to look into
    search_path = ADS_AUDIO_FOLDER
    if ads_language:
        search_path = os.path.join(ADS_AUDIO_FOLDER, ads_language)

    if not os.path.exists(search_path):
        print(f"[ERROR] Path does not exist: {search_path}")
        return

    # 2. Recursively collect .wav files
    all_files = []
    for root, _, filenames in os.walk(search_path):
        for f in filenames:
            if f.lower().endswith(".wav"):
                all_files.append(os.path.join(root, f))

    # 3. Filter for target_ads if provided
    if specific_ads:
        # Normalize target_ads (handle strings vs lists and remove extensions)
        if isinstance(specific_ads, str):
            specific_ads = [specific_ads]
            
        target_names = [os.path.splitext(os.path.basename(a))[0] for a in specific_ads]
        
        # Only keep files whose basename matches a name in target_names
        files_to_process = [
            f for f in all_files 
            if os.path.splitext(os.path.basename(f))[0] in target_names
        ]
        
        if not files_to_process:
            print(f"[WARN] No matching .wav files found for targets: {specific_ads}")
            print(f"Looked in: {search_path}")
            return
    else:
        files_to_process = all_files

    print(f"Processing {len(files_to_process)} audio files.")

    total_hashes = 0
    start_time = time.time()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    for path in tqdm(files_to_process, desc="Fingerprinting Ads"):
        fname = os.path.basename(path)
        ad_id = os.path.splitext(fname)[0]
        # Get language from the immediate parent folder name
        language = os.path.basename(os.path.dirname(path))

        try:
            # Note: process_file uses the PEAK_NEIGH_TIME defined above
            hashes, duration, y_resampled = process_file(path)
            content_hash = compute_content_hash_from_samples(y_resampled)
        except Exception as e:
            print(f"\nError processing {path}: {e}")
            continue

        # Check for duplicates using content hash
        cur.execute("SELECT ad_id, filename FROM files WHERE content_hash=?", (content_hash,))
        found = cur.fetchone()
        if found:
            existing_ad_id, existing_fn = found
            # If specifically targeting, we might want to re-process, 
            # but usually skipping is safer for DB integrity.
            print(f"Skipping {fname} — already in DB as '{existing_fn}'")
            continue

        add_hashes_to_db(DB_PATH, ad_id, fname, duration, hashes, content_hash=content_hash, language=language)
        total_hashes += len(hashes)

    conn.close()
    elapsed = time.time() - start_time
    print(f"\nDone. Total hashes stored: ~{total_hashes}. Time elapsed: {elapsed:.1f}s")

