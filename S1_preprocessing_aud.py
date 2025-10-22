# import os
# import subprocess

# #  INPUT FOLDERS
# ADS_VIDEO_FOLDER = "Inputs/ad_samples"
# MIXED_VIDEO_FOLDER = "Inputs/videos"

# # OUTPUT FOLDERS
# ADS_AUDIO_FOLDER = "Outputs/ads_fingerprints"
# MIXED_AUDIO_FOLDER = "Outputs/video_to_audio"

# SUPPORTED_VIDEO_FORMATS = (".mp4", ".mkv", ".avi", ".mov", ".flv")

# # CREATE OUTPUT DIRS
# os.makedirs(ADS_AUDIO_FOLDER, exist_ok=True)
# os.makedirs(MIXED_AUDIO_FOLDER, exist_ok=True)

# # FFmpeg extraction function 
# def extract_audio(input_path, output_path):
#     """
#     Extract audio using same params:
#     - mono
#     - 16kHz
#     - PCM 16-bit WAV
#     - loudness normalized
#     """
#     cmd = [
#         "ffmpeg",
#         "-y",  # overwrite
#         "-i", input_path,
#         "-vn",  # no video
#         "-ac", "1",  # mono
#         "-ar", "16000",  # 16kHz
#         "-af", "loudnorm",  # normalize loudness
#         "-sample_fmt", "s16",  # 16-bit PCM
#         output_path
#     ]
#     try:
#         subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         print(f" Extracted: {output_path}")
#     except subprocess.CalledProcessError as e:
#         print(f" Error extracting {input_path}\n{e.stderr.decode()}")


# converted, skipped = 0, 0

# for file_name in os.listdir(ADS_VIDEO_FOLDER):
#     if file_name.lower().endswith(SUPPORTED_VIDEO_FORMATS):
#         input_path = os.path.join(ADS_VIDEO_FOLDER, file_name)
#         base_name = file_name.split('.')[0] + ".wav"
#         output_path = os.path.join(ADS_AUDIO_FOLDER, base_name)

#         if os.path.exists(output_path):
#             print(f"[SKIP] Already converted: {output_path}")
#             skipped += 1
#             continue

#         extract_audio(input_path, output_path)
#         converted += 1

# print(f"[INFO] Finished {ADS_AUDIO_FOLDER}: {converted} new, {skipped} skipped")

# converted, skipped = 0, 0

# for file_name in os.listdir(MIXED_VIDEO_FOLDER):
#     if file_name.lower().endswith(SUPPORTED_VIDEO_FORMATS):
#         input_path = os.path.join(MIXED_VIDEO_FOLDER, file_name)
#         base_name = file_name.split('.')[0] + ".wav"
#         output_path = os.path.join(MIXED_AUDIO_FOLDER, base_name)

#         if os.path.exists(output_path):
#             print(f"[SKIP] Already converted: {output_path}")
#             skipped += 1
#             continue

#         extract_audio(input_path, output_path)
#         converted += 1

# print(f"[INFO] Finished {ADS_AUDIO_FOLDER}: {converted} new, {skipped} skipped")

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# INPUT FOLDERS
ADS_VIDEO_FOLDER = "Inputs/ad_samples"
MIXED_VIDEO_FOLDER = "Inputs/videos"

# OUTPUT FOLDERS
ADS_AUDIO_FOLDER = "Outputs/ads_fingerprints"
MIXED_AUDIO_FOLDER = "Outputs/video_to_audio"

SUPPORTED_VIDEO_FORMATS = (".mp4", ".mkv", ".avi", ".mov", ".flv")

# CREATE OUTPUT DIRS
os.makedirs(ADS_AUDIO_FOLDER, exist_ok=True)
os.makedirs(MIXED_AUDIO_FOLDER, exist_ok=True)


def extract_audio(input_path, output_path):
    """
    Extract audio using FFmpeg:
    - mono
    - 16kHz
    - PCM 16-bit WAV
    - loudness normalized
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vn", "-ac", "1",
        "-ar", "16000",
        "-af", "loudnorm",
        "-sample_fmt", "s16",
        output_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"[DONE] {os.path.basename(output_path)}"
    except subprocess.CalledProcessError as e:
        return f"[ERROR] {os.path.basename(input_path)} → {e}"


def convert_videos_to_audio(input_folder, output_folder, max_workers=6):
    """
    Convert all videos in input_folder to WAV audio in output_folder using threads.
    """
    files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(SUPPORTED_VIDEO_FORMATS)
    ]

    if not files:
        print(f"[INFO] No videos found in {input_folder}")
        return

    converted, skipped = 0, 0
    futures = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for file_name in files:
            input_path = os.path.join(input_folder, file_name)
            base_name = os.path.splitext(file_name)[0] + ".wav"
            output_path = os.path.join(output_folder, base_name)

            if os.path.exists(output_path):
                print(f"[SKIP] {base_name}")
                skipped += 1
                continue

            futures.append(executor.submit(extract_audio, input_path, output_path))

        for future in as_completed(futures):
            print(future.result())
            converted += 1

    print(f"\n[INFO] Finished converting {input_folder} → {output_folder}: {converted} new, {skipped} skipped.\n")


def convert_ads_and_videos():
    """
    Convert both ad samples and mixed videos in parallel.
    """
    print("\n=== Converting AD SAMPLES ===")
    convert_videos_to_audio(ADS_VIDEO_FOLDER, ADS_AUDIO_FOLDER, max_workers=4)

    print("\n=== Converting MIXED VIDEOS ===")
    convert_videos_to_audio(MIXED_VIDEO_FOLDER, MIXED_AUDIO_FOLDER, max_workers=6)



