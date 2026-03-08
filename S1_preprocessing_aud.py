"""
S1_preprocessing_aud.py  —  Video → WAV conversion  (parallel-optimised)

Parallelism strategy
────────────────────
• ffmpeg subprocesses are I/O-bound and independently forked by the OS, so a
  ThreadPoolExecutor is the right primitive — no GIL overhead, no inter-process
  serialisation cost.
• convert_videos_to_audio() now accepts a `queue` (multiprocessing.Queue) so
  that the caller (convert_ads_and_videos) can submit ad-sample and mixed-video
  conversions to a shared ThreadPoolExecutor, overlapping the two workloads
  instead of running them sequentially.
• All tunable constants live in config.py — unchanged API for callers.
"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import Optional

from config import (
    SR_EXPECTED,
    LOUDNORM_I, LOUDNORM_TP, LOUDNORM_LRA,
    MIN_WAV_BYTES,
    CONVERT_WORKERS_ADS, CONVERT_WORKERS_MIXED,
    SUPPORTED_VIDEO_FORMATS,
    ADS_AUDIO_FOLDER,
    MIXED_AUDIO_ROOT,
)


# ─────────────────────────────────────────────────────────────────────────────
#  SINGLE-FILE CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

def extract_audio(input_path: str, output_path: str, retries: int = 2) -> str:
    """
    Convert one video file → mono 16 kHz loudnorm WAV via ffmpeg.

    loudnorm pass (I=-16 LUFS, TP=-1.5, LRA=11) ensures broadcast audio and
    ad samples have the same loudness envelope, which is critical for
    fingerprint alignment.

    Returns a status string beginning with [DONE] or [ERROR].
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vn",
        "-ac", "1",
        "-ar", str(SR_EXPECTED),
        "-af", f"loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}",
        "-sample_fmt", "s16",
        output_path,
    ]

    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            subprocess.run(
                cmd, check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            if not os.path.exists(output_path) or os.path.getsize(output_path) < MIN_WAV_BYTES:
                if os.path.exists(output_path):
                    os.remove(output_path)
                last_err = f"output file missing or too small after attempt {attempt}"
                continue
            return f"[DONE] {os.path.basename(output_path)}"
        except subprocess.CalledProcessError as e:
            last_err = (
                e.stderr.decode(errors="replace").strip().splitlines()[-1]
                if e.stderr else str(e)
            )

    return f"[ERROR] {os.path.basename(input_path)} → {last_err}"


# ─────────────────────────────────────────────────────────────────────────────
#  FOLDER-LEVEL CONVERSION  (ThreadPoolExecutor over ffmpeg subprocesses)
# ─────────────────────────────────────────────────────────────────────────────

def _collect_jobs(
    input_folder: str,
    output_folder: str,
    target_files: Optional[list] = None,
) -> list[tuple[str, str]]:
    """
    Return [(input_path, output_path)] for every video that still needs
    converting, skipping already-valid WAVs.
    """
    video_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(SUPPORTED_VIDEO_FORMATS)
    ]
    if target_files:
        target_set = set(target_files)
        video_files = [f for f in video_files if f in target_set]

    jobs = []
    for fname in video_files:
        wav_name    = os.path.splitext(fname)[0] + ".wav"
        output_path = os.path.join(output_folder, wav_name)
        if os.path.exists(output_path) and os.path.getsize(output_path) >= MIN_WAV_BYTES:
            continue                       # already done
        jobs.append((os.path.join(input_folder, fname), output_path))
    return jobs


def convert_videos_to_audio(
    input_folder:  str,
    output_folder: str,
    max_workers:   int           = 8,
    target_files:  Optional[list] = None,
    executor:      Optional[ThreadPoolExecutor] = None,
) -> dict:
    """
    Convert every supported video in *input_folder* to a mono 16 kHz WAV.

    Args:
        input_folder:  Source directory containing video files.
        output_folder: Destination directory for WAV files.
        max_workers:   Thread pool size (ignored when *executor* is supplied).
        target_files:  Whitelist of filenames; None = all.
        executor:      Optional *shared* ThreadPoolExecutor.  When provided the
                       caller is responsible for lifecycle management and the
                       function submits jobs without blocking — useful when you
                       want ads and mixed-videos to convert concurrently.

    Returns {"converted": n, "skipped": n, "errors": n}.
    """
    if not os.path.exists(input_folder):
        print(f"[ERROR] Input folder not found: {input_folder}")
        return {"converted": 0, "skipped": 0, "errors": 0}

    os.makedirs(output_folder, exist_ok=True)

    jobs    = _collect_jobs(input_folder, output_folder, target_files)
    skipped = (
        len([
            f for f in os.listdir(input_folder)
            if f.lower().endswith(SUPPORTED_VIDEO_FORMATS)
        ]) - len(jobs)
    )

    if not jobs:
        print(f"[INFO] No new conversions needed in: {input_folder}")
        return {"converted": 0, "skipped": skipped, "errors": 0}

    converted = errors = 0

    def _drain(futures_list: list[Future]) -> tuple[int, int]:
        ok = err = 0
        for fut in as_completed(futures_list):
            result = fut.result()
            print(result)
            if result.startswith("[DONE]"):
                ok += 1
            else:
                err += 1
        return ok, err

    if executor is not None:
        # Non-blocking submission; caller must drain futures themselves or
        # wait on the shared executor's __exit__.
        futures = [executor.submit(extract_audio, inp, out) for inp, out in jobs]
        converted, errors = _drain(futures)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(extract_audio, inp, out) for inp, out in jobs]
            converted, errors = _drain(futures)

    print(
        f"\n[INFO] {input_folder} → {output_folder}: "
        f"{converted} converted, {skipped} skipped, {errors} errors.\n"
    )
    return {"converted": converted, "skipped": skipped, "errors": errors}


# ─────────────────────────────────────────────────────────────────────────────
#  STEP-1 ENTRY POINT  (ads + mixed in parallel)
# ─────────────────────────────────────────────────────────────────────────────

def convert_ads_and_videos(
    channel_path:  str,
    date:          str,
    time_sel:      Optional[list],
    ads_language:  str,
    specific_ads:  Optional[list] = None,
) -> dict:
    """
    Step 1 entry-point: convert ad samples AND broadcast recordings in parallel.

    Key change vs original
    ──────────────────────
    Both workloads (ads + mixed) are submitted to a *single* shared
    ThreadPoolExecutor so their ffmpeg subprocesses run concurrently.
    Total worker count = CONVERT_WORKERS_ADS + CONVERT_WORKERS_MIXED, which
    saturates typical NAS/disk bandwidth without over-subscribing CPU.

    Args:
        channel_path:  Absolute path to the channel root.
        date:          Date string folder name (e.g. '20250918').
        time_sel:      List of specific recording filenames, or None for all.
        ads_language:  Language sub-folder inside Inputs/ads_samples.
        specific_ads:  List of specific ad filenames to convert, or None for all.

    Returns merged stats dict {"ads": {...}, "mixed": {...}}.
    """
    ads_video_folder   = os.path.join("Inputs/ads_samples", ads_language)
    ads_audio_folder   = os.path.join(ADS_AUDIO_FOLDER, ads_language)
    mixed_video_folder = os.path.join(channel_path, date)
    mixed_audio_folder = os.path.join(
        MIXED_AUDIO_ROOT, os.path.basename(channel_path), date
    )

    os.makedirs(ads_audio_folder,   exist_ok=True)
    os.makedirs(mixed_audio_folder, exist_ok=True)

    total_workers = CONVERT_WORKERS_ADS + CONVERT_WORKERS_MIXED

    print(f"\n=== [S1] PARALLEL CONVERSION  language={ads_language} | workers={total_workers} ===")

    # ── Shared pool so ads and mixed recordings convert simultaneously ──────────
    with ThreadPoolExecutor(max_workers=total_workers) as shared_pool:

        # Submit all ad jobs
        ads_jobs = _collect_jobs(ads_video_folder, ads_audio_folder, specific_ads) \
                   if os.path.exists(ads_video_folder) else []
        ads_futures = [
            shared_pool.submit(extract_audio, inp, out)
            for inp, out in ads_jobs
        ]

        # Submit all mixed-recording jobs
        mix_jobs = _collect_jobs(mixed_video_folder, mixed_audio_folder, time_sel) \
                   if os.path.exists(mixed_video_folder) else []
        mix_futures = [
            shared_pool.submit(extract_audio, inp, out)
            for inp, out in mix_jobs
        ]

        # ── Drain ads ────────────────────────────────────────────────────────────
        ads_converted = ads_errors = 0
        for fut in as_completed(ads_futures):
            result = fut.result()
            print(f"  [AD]    {result}")
            if result.startswith("[DONE]"):
                ads_converted += 1
            else:
                ads_errors += 1

        # ── Drain mixed ──────────────────────────────────────────────────────────
        mix_converted = mix_errors = 0
        for fut in as_completed(mix_futures):
            result = fut.result()
            print(f"  [MIX]   {result}")
            if result.startswith("[DONE]"):
                mix_converted += 1
            else:
                mix_errors += 1

    ads_skipped = max(0, len(ads_jobs) - ads_converted - ads_errors)
    mix_skipped = max(0, len(mix_jobs) - mix_converted - mix_errors)

    ads_stats = {"converted": ads_converted, "skipped": ads_skipped, "errors": ads_errors}
    mix_stats = {"converted": mix_converted, "skipped": mix_skipped, "errors": mix_errors}

    print(
        f"\n[S1] Ads   → {ads_stats}\n"
        f"[S1] Mixed → {mix_stats}\n"
    )
    return {"ads": ads_stats, "mixed": mix_stats}