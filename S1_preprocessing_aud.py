import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

SUPPORTED_VIDEO_FORMATS = (".mp4", ".mkv", ".avi", ".mov", ".flv", ".ts")

def extract_audio(input_path, output_path):
    """
    Extract audio using FFmpeg with mono 16kHz PCM WAV.
    """

    # cmd = [
    #     "ffmpeg", "-y",
    #     "-i", input_path,
    #     "-vn", "-ac", "1",
    #     "-ar", "16000",
    #     "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", # Targeted for broadcast consistency
    #     "-sample_fmt", "s16",
    #     output_path
    # ]

    # cmd = [
    #     "ffmpeg",
    #     "-y",  # overwrite
    #     "-i", input_path,
    #     "-vn",  # no video
    #     "-ac", "1",  # mono
    #     "-ar", "16000",  # 16kHz
    #     "-af", "loudnorm",  # normalize loudness
    #     "-sample_fmt", "s16",  # 16-bit PCM
    #     output_path
    # ]

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

def convert_videos_to_audio(input_folder, output_folder, max_workers=10, target_files=None): 
    """
    Convert all videos in input_folder to WAV audio in output_folder using threads.
    If target_files is provided, only convert those.
    """
    if not os.path.exists(input_folder):
        print(f"[ERROR] Input folder not found: {input_folder}")
        return

    all_files = os.listdir(input_folder)
    
    # Filter files by supported formats
    video_files = [f for f in all_files if f.lower().endswith(SUPPORTED_VIDEO_FORMATS)]
    
    if target_files:
        # If specific ads/files are requested, filter the full list
        files_to_process = [f for f in video_files if f in target_files]
    else:
        # Otherwise, process all video files in the language folder
        files_to_process = video_files

    if not files_to_process:
        print(f"[INFO] No videos found in {input_folder} for target selection.")
        return

    os.makedirs(output_folder, exist_ok=True)
    converted, skipped = 0, 0
    futures = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for file_name in files_to_process:
            input_path = os.path.join(input_folder, file_name)
            base_name = os.path.splitext(file_name)[0] + ".wav"
            output_path = os.path.join(output_folder, base_name)

            if os.path.exists(output_path):
                # Only skip if the file already exists to save time
                skipped += 1
                continue

            futures.append(executor.submit(extract_audio, input_path, output_path))

        for future in as_completed(futures):
            result = future.result()
            if result.startswith("[DONE]"):
                converted += 1
            print(result)

    print(f"\n[INFO] Finished converting {input_folder} → {output_folder}: {converted} processed, {skipped} skipped.\n")


def convert_ads_and_videos(channel_path, date, time_sel, ads_language, specific_ads=None):
    """
    Convert both ad samples and mixed videos for the selected channel/date.
    specific_ads: List of filenames (e.g. ['kent.mp4']) to filter ad processing.
    """
    # Dynamic path based on the 'language' sent in request (Tamil, Kannada, etc.)
    ads_video_folder = os.path.join("Inputs/ads_samples", ads_language)
    ads_audio_folder = os.path.join("Outputs/ads_fingerprints", ads_language)
    
    mixed_video_folder = os.path.join(channel_path, date)
    mixed_audio_folder = os.path.join("Outputs/video_to_audio", os.path.basename(channel_path), date)

    os.makedirs(ads_audio_folder, exist_ok=True)
    os.makedirs(mixed_audio_folder, exist_ok=True)

    print(f"\n=== Converting AD SAMPLES ({ads_language}) ===")
    # Apply specific_ads filtering logic to the ads folder
    convert_videos_to_audio(
        ads_video_folder, 
        ads_audio_folder, 
        max_workers=4, 
        target_files=specific_ads
    )

    print("\n=== Converting MIXED VIDEOS ===")
    # Apply time_sel filtering logic to the channel recordings
    convert_videos_to_audio(
        mixed_video_folder, 
        mixed_audio_folder, 
        max_workers=6, 
        target_files=time_sel
    )