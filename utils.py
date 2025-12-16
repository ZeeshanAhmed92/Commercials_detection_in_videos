# =============================================================
#   IMPORTS
# =============================================================
import os
import time
import json
import psutil
import pandas as pd
import multiprocessing
import concurrent.futures
from S1_preprocessing_aud import convert_ads_and_videos
from S2_fingerprint_db import run_flow
from S3_scan_test_improved_latest import detect_ads


def run_full_pipeline(channel, date, time_sel, base_path, ads_language="Hindi", num_workers=None):
    """
    Full pipeline to convert videos, create fingerprints, and detect ads.

    Args:
        channel (str): The channel name (e.g., 'NDTVIndia').
        date (str): The date string (e.g., '20251205').
        time_sel (list or None): A list of specific file names to process 
                                 (e.g., ["1200.mpd.mp4"]), or None to process all files 
                                 in the date folder.
        base_path (str): The root path to the video recordings.
        ads_language (str): The language tag for ad samples (e.g., 'Hindi').
        num_workers (int or None): The number of workers for parallel processing 
                                   during detection (overrides auto-detection).

    Returns:
        list: A list of dictionaries, where each dictionary represents a detected ad segment.
    """
    if num_workers is None:
        total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        if total_ram_gb < 8:
            num_workers = 1
        elif total_ram_gb < 16:
            num_workers = 2
        else:
            # Set a reasonable default limit based on available resources
            num_workers = min(5, multiprocessing.cpu_count())

    print(f"\n=== Starting full pipeline for {channel}/{date} ===")
    final_path = os.path.join(base_path, channel)
    
    # --- Step 1: Conversion (Now respects the time_sel list) ---

    t0 = time.time()
    # time_sel is passed to convert_ads_and_videos to filter which files get converted.
    convert_ads_and_videos(final_path, date, time_sel, ads_language=ads_language)
    print(f"[{channel}/{date}][TIMER] Convert time: {time.time() - t0:.2f} sec")

    # --- Step 2: Fingerprint DB (Build/Update DB for AD samples) ---

    t1 = time.time()
    run_flow()
    print(f"[{channel}/{date}][TIMER] Fingerprint time: {time.time() - t1:.2f} sec")

    # --- Step 3: Detection (Scan the converted audio files) ---
    t2 = time.time()
    detected_segments = detect_ads(
        language=ads_language, 
        channel=channel, 
        date=date, 
        time_sel=time_sel, # Pass the list of files to filter the audio folder
        num_workers=num_workers # Pass the worker count for parallel detection
    )
    print(f"[{channel}/{date}][TIMER] Detect time: {time.time() - t2:.2f} sec")

    print(f"[{channel}/{date}][TOTAL] Pipeline completed in {time.time() - t0:.2f} sec\n")
    
    return detected_segments


def run_task_wrapper(task, num_workers):
    """Wrapper function to execute run_full_pipeline for concurrent execution."""
    
    print(f"==================================================")
    print(f"   STARTING TASK: {task['channel']} / {task['date']}")
    print(f"==================================================")

    # Call the pipeline and collect the results
    task_results = run_full_pipeline(
        channel=task['channel'],
        date=task['date'],
        time_sel=task['time_sel'],
        base_path=task['base_path'],
        ads_language=task['ads_language'],
        num_workers=num_workers 
    )
    return task_results


def run_all_tasks_and_save_json(tasks_to_run, output_json_path="Outputs/ALL_DETECTED_ADS.json", num_workers=None):
    """
    Runs all defined tasks in parallel using ThreadPoolExecutor and saves 
    the aggregated results to a single JSON file.
    """
    
    # Logic to determine workers (You can keep your hardcoded 5 if preferred)
    max_workers = 5
    if num_workers is not None:
        max_workers = num_workers
    
    print("\n" + "#"*70)
    print(f"🚀 Running {len(tasks_to_run)} tasks in parallel with {max_workers} threads.")
    print("#"*70)
    
    ALL_RESULTS = []

    # Use ThreadPoolExecutor for concurrent execution of all tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor
        future_to_task = {
            executor.submit(run_task_wrapper, task, num_workers): task 
            for task in tasks_to_run
        }
        
        # Iterate over completed futures to collect results
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                task_results = future.result()
                ALL_RESULTS.extend(task_results) # Add results to the master list
                print(f"✅ COMPLETED: {task['channel']} / {task['date']} - {len(task_results)} segments detected.")
            except Exception as exc:
                print(f"❌ FAILED TASK: {task['channel']} / {task['date']} generated an exception: {exc}")

    # --- Final Aggregation and Saving ---
    
    if ALL_RESULTS:
        df = pd.DataFrame(ALL_RESULTS)
        
        # Re-sort and re-index for the final Segment No
        # (Ensures the JSON list is ordered chronologically)
        df.sort_values(by=["Channel", "Date", "Time", "Start Time"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["Segment No"] = df.index + 1
        
        # Ensure the Outputs directory exists
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        
        # === SAVE AS JSON ===
        # orient='records' creates: [{"col": "val"}, {"col": "val"}]
        # indent=4 makes it human-readable
        df.to_json(output_json_path, orient='records', indent=4)
        
        print("\n" + "="*70)
        print(f"🏆 SUCCESS: Aggregated results for {len(ALL_RESULTS)} segments saved to:")
        print(f" {output_json_path}")
        print("="*70)
        
        # Return the data as a list of dicts for the API
        return df.to_dict(orient='records')

    else:
        print("\n[INFO] No ad segments were detected across all tasks.")
        
        # Create an empty JSON file so the path exists
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump([], f)
            
        return []
