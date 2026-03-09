"""
config.py  —  Single source of truth for all pipeline parameters.

Import this in S2, S3, utils, and app.py instead of defining constants locally.
Changing a value here automatically affects every stage.
"""

import os

# ═══════════════════════════════════════════════════════════════════════════════
#  PATHS
# ═══════════════════════════════════════════════════════════════════════════════

BASE_PATH        = "//192.168.39.7/mps/disk1/disk1-recordings"
# BASE_PATH = "./"
ADS_SAMPLES_PATH = "Inputs/ads_samples"
ADS_AUDIO_FOLDER = "Outputs/ads_fingerprints"
MIXED_AUDIO_ROOT = "Outputs/video_to_audio"
DB_PATH          = "Outputs/DB/ads_fingerprints.db"
OUTPUT_DIR       = "Outputs/API_Results"


# ═══════════════════════════════════════════════════════════════════════════════
#  AUDIO CONVERSION  (S1)
# ═══════════════════════════════════════════════════════════════════════════════

SR_EXPECTED      = 16_000        # target sample rate (Hz)
LOUDNORM_I       = -16           # integrated loudness target (LUFS)
LOUDNORM_TP      = -1.5          # true-peak ceiling (dBFS)
LOUDNORM_LRA     = 11            # loudness range (LU)
MIN_WAV_BYTES    = 16_000        # minimum valid output file size
CONVERT_WORKERS_ADS   = 6        # ffmpeg threads for ad samples
CONVERT_WORKERS_MIXED = 8        # ffmpeg threads for mixed recordings
SUPPORTED_VIDEO_FORMATS = (
    ".mp4", ".mkv", ".avi", ".mov", ".flv", ".ts", ".mpeg"
)


# ═══════════════════════════════════════════════════════════════════════════════
#  SPECTROGRAM / PEAK EXTRACTION  (S2 & S3 — MUST BE IDENTICAL)
# ═══════════════════════════════════════════════════════════════════════════════

N_FFT                  = 2048
HOP_LENGTH             = 256
AMP_MIN_DB             = -80     # peaks below this dB (rel to max) are ignored
PEAK_NEIGHBORHOOD_FREQ = 20      # frequency neighborhood size (bins)
PEAK_NEIGHBORHOOD_TIME = 10      # time neighborhood size (frames)
# Short aliases (used in S3)
PEAK_NEIGH_FREQ = PEAK_NEIGHBORHOOD_FREQ
PEAK_NEIGH_TIME = PEAK_NEIGHBORHOOD_TIME


# ═══════════════════════════════════════════════════════════════════════════════
#  FINGERPRINT GENERATION  (S2 & S3 — MUST BE IDENTICAL)
# ═══════════════════════════════════════════════════════════════════════════════

FAN_VALUE        = 40            # neighbor peaks paired per anchor point
MAX_TIME_DELTA   = 400           # max frame gap between paired peaks
HASH_TRUNCATE    = 20            # SHA-1 hex chars stored per hash


# ═══════════════════════════════════════════════════════════════════════════════
#  DATABASE  (S2)
# ═══════════════════════════════════════════════════════════════════════════════

BATCH_INSERT_SIZE = 5_000        # fingerprint rows per SQLite commit


# ═══════════════════════════════════════════════════════════════════════════════
#  DETECTION TUNING  (S3)
# ═══════════════════════════════════════════════════════════════════════════════

DELTA_BIN_SIZE     = HOP_LENGTH / SR_EXPECTED   # ~0.032 s — 1 hop frame
MIN_DETECTION_SEC  = 1          # discard detections shorter than this
MIN_CONFIDENCE     = 0.70        # minimum hash-match score to keep a candidate
MERGE_GAP          = 3.0        # merge detections of same ad within this gap (s)

# Boundary refinement (cross-correlation)
LOCAL_MARGIN       = 3.0         # search window around coarse boundary (s)
CORR_THRESHOLD     = 0.5        # minimum normalised correlation to accept refinement

# Full / Partial classification
FULL_DUR_TOLERANCE = 0.5         # ± seconds from reference duration → "Full"
FULL_SCORE_FLOOR   = 0.95        # minimum score to classify as "Full"

BULK_SQL_IN_CHUNK  = 5_000


# ═══════════════════════════════════════════════════════════════════════════════
#  PARALLELISM  (utils)
# ═══════════════════════════════════════════════════════════════════════════════

MAX_OUTER_TASKS    = 4           # concurrent pipeline tasks (each uses inner workers)
# Inner worker count is auto-detected from RAM at runtime (see utils._default_workers)


# ═══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE: derived constants (do not edit)
# ═══════════════════════════════════════════════════════════════════════════════

HOP_TIME_SEC = HOP_LENGTH / SR_EXPECTED   # seconds per spectrogram frame ≈ 0.032 s


# ═══════════════════════════════════════════════════════════════════════════════
#  S3 INTERNAL PARALLELISM  (new)
# ═══════════════════════════════════════════════════════════════════════════════

HASH_LOOKUP_CHUNKS = 4    # partitions for parallel hash-lookup inside detect_for_file()
REFINE_WORKERS     = 4    # threads for parallel boundary refinement inside detect_for_file()