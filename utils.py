# In utils.py

from S1_preprocessing_aud import convert_ads_and_videos
from S2_fingerprint_db import run_flow
# Make sure this import is correct:
from S3_scan_test_improved_latest import detect_ads # Assuming this is your modified file
import time
import pandas as pd
import os
import psutil
import multiprocessing # Import for setting default workers

def run_full_pipeline(channel, date, time_sel, base_path, ads_language="Hindi", num_workers=None): # ADD num_workers
    """
    Full pipeline with an option for parallel workers in detection.
    """
    # if num_workers is None:
    #     num_workers = multiprocessing.cpu_count()


    if num_workers is None:
        total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        if total_ram_gb < 8:
            num_workers = 1
        elif total_ram_gb < 16:
            num_workers = 2
        else:
            num_workers = min(4, multiprocessing.cpu_count())


    print("\n=== Starting full pipeline ===")
    final_path = os.path.join(base_path, channel)
    
    # Step 1: Conversion (Keep sequential)
    t0 = time.time()
    convert_ads_and_videos(final_path, date, time_sel, ads_language="Hindi")
    print(f"[TIMER] Convert time: {time.time() - t0:.2f} sec")

    # Step 2: Fingerprint DB (Keep sequential)
    t1 = time.time()
    run_flow()
    print(f"[TIMER] Fingerprint time: {time.time() - t1:.2f} sec")

    # Step 3: Detection (Parallelized)
    t2 = time.time()
    detected_segments = detect_ads(
        language=ads_language, 
        channel=channel, 
        date=date, 
        time_sel=time_sel,
        # num_workers=num_workers # Pass the worker count
    )
    print(f"[TIMER] Detect time: {time.time() - t2:.2f} sec")

    print(f"[TOTAL] Pipeline completed in {time.time() - t0:.2f} sec\n")
    
    return detected_segments 


def run_all_tasks_and_save_csv(tasks_to_run, output_csv_path="Outputs/ALL_DETECTED_ADS.csv", num_workers=None): # ADD num_workers
    """
    Runs all defined tasks and saves the aggregated results to a single CSV.
    """
    # ... (unchanged code) ...
    
    ALL_RESULTS = []
    
    for i, task in enumerate(tasks_to_run):
        print(f"==================================================")
        print(f"    STARTING TASK {i+1}/{len(tasks_to_run)}: {task['channel']} / {task['date']}")
        print(f"==================================================")

        # Call the pipeline and collect the results
        task_results = run_full_pipeline(
            channel=task['channel'],
            date=task['date'],
            time_sel=task['time_sel'],
            base_path=task['base_path'],
            ads_language=task['ads_language'],
            num_workers=num_workers # Pass the worker count
        )
        ALL_RESULTS.extend(task_results) # Add results to the master list

    # ... (unchanged code for final aggregation and saving) ...
    # ... (Make sure to handle the `df` DataFrame creation and saving logic) ...
    
    if ALL_RESULTS:
        df = pd.DataFrame(ALL_RESULTS)
        # Re-sort and re-index for the final Segment No
        df.sort_values(by=["Channel", "Date", "Time", "Start Time"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["Segment No"] = df.index + 1
        
        # Ensure the Outputs directory exists
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        
        print("\n" + "="*70)
        print(f"🏆 SUCCESS: Aggregated results for {len(ALL_RESULTS)} segments saved to:")
        print(f"    {output_csv_path}")
        print("="*70)
        
        # Optional: Print the first few results
        print("\nHead of Final Results:\n", df.head().to_string(index=False))

    else:
        print("\n[INFO] No ad segments were detected across all tasks.")


if __name__ == "__main__":
    BASE_PATH = "/run/user/1000/gvfs/smb-share:server=media-system-product-name.local,share=mps/disk1/disk1-recordings"
    ADS_LANGUAGE = "Hindi"
    
    # Define the number of workers to use (e.g., 4, or None for max cores)
    N_WORKERS = 6
    
    # 1. Define channels and base path
    CHANNELS = ["NdtvIndiaBackup"]
    
    tasks_to_run = []
    
    # --- START MODIFICATION ---
    TARGET_DATE = "20251117"  # <--- New variable for the specific date
    
    for channel in CHANNELS:
        channel_full_path = os.path.join(BASE_PATH, channel)
        
        if not os.path.exists(channel_full_path):
            print(f"[WARNING] Channel folder not found: {channel_full_path}. Skipping.")
            continue
            
        # Check if the TARGET_DATE folder exists within the channel
        target_date_path = os.path.join(channel_full_path, TARGET_DATE)
        # TARGET_TIME = "2200"
        
        if os.path.exists(target_date_path) and TARGET_DATE.isdigit() and len(TARGET_DATE) == 8:

            tasks_to_run.append({
                "channel": channel,
                "date": TARGET_DATE,
                "time_sel": None,
                "base_path": BASE_PATH,
                "ads_language": ADS_LANGUAGE
            })
        else:
             print(f"[WARNING] Date folder not found for {channel}/{TARGET_DATE}. Skipping.")

    # --- END MODIFICATION ---
    
    # 3. Run all tasks and save the final aggregated CSV
    run_all_tasks_and_save_csv(
        tasks_to_run,
        output_csv_path="Outputs/ALL_DETECTED_ADS_AGGREGATED.csv",
        num_workers=N_WORKERS # Pass the worker count to the runner
    )


# if __name__ == "__main__":
#     BASE_PATH = "/run/user/1000/gvfs/smb-share:server=192.168.39.4,share=mps/disk1/disk1-recordings"
#     ADS_LANGUAGE = "Hindi"
    
#     # Define the number of workers to use (e.g., 4, or None for max cores)
#     N_WORKERS = 8
    
#     # 1. Define channels and base path
#     CHANNELS = ["SonyTen1SD"]

#     tasks_to_run = []

#     # ... (unchanged code for discovering channels and dates) ...
#     for channel in CHANNELS:
#         channel_full_path = os.path.join(BASE_PATH, channel)
        
#         if not os.path.exists(channel_full_path):
#             print(f"[WARNING] Channel folder not found: {channel_full_path}. Skipping.")
#             continue
            
#         all_items = os.listdir(channel_full_path)
#         date_folders = sorted([d for d in all_items if d.isdigit() and len(d) == 8])
        
#         for date in date_folders:
#             tasks_to_run.append({
#                 "channel": channel,
#                 "date": date,
#                 "time_sel": None,
#                 "base_path": BASE_PATH,
#                 "ads_language": ADS_LANGUAGE
#             })

#     # 3. Run all tasks and save the final aggregated CSV
#     run_all_tasks_and_save_csv(
#         tasks_to_run,
#         output_csv_path="Outputs/ALL_DETECTED_ADS_AGGREGATED.csv",
#         num_workers=N_WORKERS # Pass the worker count to the runner
#     )


# if __name__ == "__main__":
#     BASE_PATH = "/run/user/1000/gvfs/smb-share:server=192.168.39.4,share=mps/disk1/disk1-recordings"
#     ADS_LANGUAGE = "Hindi"
    
#     # Define channels and target date
#     CHANNELS = ["SonyTen1SD"]
#     TARGET_DATE = "20250911"  # 👈 Only process this date

#     tasks_to_run = []

#     for channel in CHANNELS:
#         channel_full_path = os.path.join(BASE_PATH, channel)
        
#         if not os.path.exists(channel_full_path):
#             print(f"[WARNING] Channel folder not found: {channel_full_path}. Skipping.")
#             continue
            
#         all_items = os.listdir(channel_full_path)
#         date_folders = sorted([d for d in all_items if d.isdigit() and len(d) == 8])

#         # ✅ Only include the target date if it exists
#         if TARGET_DATE in date_folders:
#             tasks_to_run.append({
#                 "channel": channel,
#                 "date": TARGET_DATE,
#                 "time_sel": None,  # Process all videos in the date folder
#                 "base_path": BASE_PATH,
#                 "ads_language": ADS_LANGUAGE
#             })
#         else:
#             print(f"[INFO] Date folder {TARGET_DATE} not found under {channel_full_path}")

#     # 3. Run all tasks and save the final aggregated CSV
#     if tasks_to_run:
#         run_all_tasks_and_save_csv(
#             tasks_to_run,
#             output_csv_path=f"Outputs/ALL_DETECTED_ADS_{TARGET_DATE}.csv"
#         )
#     else:
#         print("[INFO] No tasks to run.")
