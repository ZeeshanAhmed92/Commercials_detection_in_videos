# import os
# import subprocess
# from concurrent.futures import ThreadPoolExecutor, as_completed

# # ========= USER SELECTION =========
# CHANNEL_PATH = "/media/mediaplanet/Data/SonyTen1SD"  # <-- change this
# DATE = "20250918"                                  # <-- change this
# ADS_LANGUAGE = "Hindi"                             # <-- change this

# # ========= AUTO PATHS =========
# ADS_VIDEO_FOLDER = os.path.join("Inputs/ads_samples", ADS_LANGUAGE)
# MIXED_VIDEO_FOLDER = os.path.join(CHANNEL_PATH, DATE)

# ADS_AUDIO_FOLDER   = os.path.join("Outputs/ads_fingerprints", ADS_LANGUAGE)
# MIXED_AUDIO_FOLDER = os.path.join("Outputs/video_to_audio", os.path.basename(CHANNEL_PATH), DATE)

# SUPPORTED_VIDEO_FORMATS = (".mp4", ".mkv", ".avi", ".mov", ".flv", ".ts")

# # CREATE OUTPUT DIRS
# os.makedirs(ADS_AUDIO_FOLDER, exist_ok=True)
# os.makedirs(MIXED_AUDIO_FOLDER, exist_ok=True)


# def extract_audio(input_path, output_path):
#     """
#     Extract audio using FFmpeg with a robust fallback mechanism.
#     First tries: mono, 16kHz, PCM 16-bit WAV, loudness normalized.
#     If that fails, falls back to: mono, 16kHz, PCM 16-bit WAV (no loudnorm).
#     """
#     input_file_name = os.path.basename(input_path)
    
#     # --- 1. Normalized Extraction (Preferred) ---
#     cmd_normalized = [
#         "ffmpeg", "-y",
#         "-i", input_path,
#         "-vn", "-ac", "1",
#         "-ar", "16000",
#         # "-af", "loudnorm",  # <-- The often problematic filter
#         "-sample_fmt", "s16",
#         output_path
#     ]
    
#     try:
#         # Capture stderr to get FFmpeg's error message if check=True fails
#         result = subprocess.run(cmd_normalized, check=True, capture_output=True, text=True)
#         return f"[DONE] {input_file_name} (Normalized)"
        
#     except subprocess.CalledProcessError as e:
#         # Store the error message from the failed normalized attempt
#         normalized_error = e.stderr.strip().split('\n')[-2:]  # Last few lines often have the core error
#         normalized_error_msg = " | ".join(normalized_error)
        
#         # --- 2. Fallback: Simple Extraction (No loudnorm) ---
#         print(f"[WARN] Failed Normalized extraction for {input_file_name}. Falling back to simple copy.")
        
#         cmd_fallback = [
#             "ffmpeg", "-y",
#             "-hwaccel", "cuda",
#             "-i", input_path,
#             "-vn", "-ac", "1",
#             "-ar", "16000",
#             "-sample_fmt", "s16",
#             output_path
#         ]
        
#         try:
#             # Re-run without loudnorm
#             subprocess.run(cmd_fallback, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#             return f"[DONE] {input_file_name} (Fallback-Simple)"
        
#         except subprocess.CalledProcessError as e_fallback:
#             # Final failure: report both errors
#             return f"[ERROR] {input_file_name} → Normalized Fail: '{normalized_error_msg}' | Fallback Fail: '{e_fallback}'"

# # --- Main conversion functions remain the same ---

# def convert_videos_to_audio(input_folder, output_folder, max_workers=14):
#     """
#     Convert all videos in input_folder to WAV audio in output_folder using threads.
#     """
#     if not os.path.exists(input_folder):
#         print(f"[ERROR] Input folder not found: {input_folder}")
#         return

#     files = [
#         f for f in os.listdir(input_folder)
#         if f.lower().endswith(SUPPORTED_VIDEO_FORMATS)
#     ]

#     if not files:
#         print(f"[INFO] No videos found in {input_folder}")
#         return

#     converted, skipped = 0, 0
#     futures = []

#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         for file_name in files:
#             input_path = os.path.join(input_folder, file_name)
#             base_name = os.path.splitext(file_name)[0] + ".wav"
#             output_path = os.path.join(output_folder, base_name)

#             if os.path.exists(output_path):
#                 print(f"[SKIP] {base_name}")
#                 skipped += 1
#                 continue

#             futures.append(executor.submit(extract_audio, input_path, output_path))

#         for future in as_completed(futures):
#             result = future.result()
#             print(result)
#             if not result.startswith("[SKIP]"):
#                  converted += 1

#     print(f"\n[INFO] Finished converting {input_folder} → {output_folder}: {converted} processed, {skipped} skipped.\n")


# def convert_ads_and_videos():
#     """
#     Convert both ad samples and mixed videos in parallel.
#     """
#     print("\n=== Converting AD SAMPLES ===")
#     convert_videos_to_audio(ADS_VIDEO_FOLDER, ADS_AUDIO_FOLDER, max_workers=4)

#     print("\n=== Converting MIXED VIDEOS ===")
#     convert_videos_to_audio(MIXED_VIDEO_FOLDER, MIXED_AUDIO_FOLDER, max_workers=6)

# if __name__ == "__main__":
#     convert_ads_and_videos()

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

SUPPORTED_VIDEO_FORMATS = (".mp4", ".mkv", ".avi", ".mov", ".flv", ".ts")

def extract_audio(input_path, output_path):
    """
    Extract audio using FFmpeg with mono 16kHz PCM WAV.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vn", "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        output_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"[DONE] {os.path.basename(output_path)}"
    except subprocess.CalledProcessError as e:
        return f"[ERROR] {os.path.basename(input_path)} → {e}"

def convert_videos_to_audio(input_folder, output_folder, max_workers=10):
    """
    Convert all videos in input_folder to WAV audio in output_folder using threads.
    """
    if not os.path.exists(input_folder):
        print(f"[ERROR] Input folder not found: {input_folder}")
        return

    files = [f for f in os.listdir(input_folder) if f.lower().endswith(SUPPORTED_VIDEO_FORMATS)]
    if not files:
        print(f"[INFO] No videos found in {input_folder}")
        return

    os.makedirs(output_folder, exist_ok=True)
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
            result = future.result()
            print(result)
            if result.startswith("[DONE]"):
                converted += 1

    print(f"\n[INFO] Finished converting {input_folder} → {output_folder}: {converted} processed, {skipped} skipped.\n")


def convert_ads_and_videos(channel_path, date, time_sel, ads_language="Hindi"):
    """
    Convert both ad samples and mixed videos for the selected channel/date.
    """
    ads_video_folder = os.path.join("Inputs/ads_samples", ads_language)
    ads_audio_folder = os.path.join("Outputs/ads_fingerprints", ads_language)
    mixed_video_folder = os.path.join(channel_path, date)
    mixed_audio_folder = os.path.join("Outputs/video_to_audio", os.path.basename(channel_path), date)

    os.makedirs(ads_audio_folder, exist_ok=True)
    os.makedirs(mixed_audio_folder, exist_ok=True)

    print("\n=== Converting AD SAMPLES ===")
    convert_videos_to_audio(ads_video_folder, ads_audio_folder, max_workers=4)

    print("\n=== Converting MIXED VIDEOS ===")
    convert_videos_to_audio(mixed_video_folder, mixed_audio_folder, max_workers=6)
