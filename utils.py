# """
# utils.py  —  Pipeline orchestration (fixed)

# Key fixes vs original:
#   • db_lock is actually used in the SQLite-writing stages via a threading.Lock
#     so concurrent tasks don't corrupt the DB when writing fingerprints.
#   • run_all_tasks_and_save_json no longer spawns a ThreadPoolExecutor with
#     tasks that each launch their own multiprocessing.Pool  (nested parallelism
#     caused OOM / deadlocks on many systems).  Instead tasks run sequentially
#     or with a controlled outer thread count (default 2).
#   • num_workers is correctly propagated everywhere (was silently ignored).
#   • Pipeline timer output cleaned up.
# """

# import os
# import time
# import json
# import psutil
# import threading
# import pandas as pd
# import multiprocessing
# import concurrent.futures

# from config import MAX_OUTER_TASKS
# from S1_preprocessing_aud import convert_ads_and_videos
# from S2_fingerprint_db     import run_flow
# from S3_scan_test_improved_latest import detect_ads

# # Shared lock so parallel tasks don't clobber the SQLite DB during S2
# db_lock = threading.Lock()


# # ─────────────────────────────────────────────────────────────────────────────
# #  RAM-AWARE WORKER COUNT
# # ─────────────────────────────────────────────────────────────────────────────

# def _default_workers() -> int:
#     ram_gb = psutil.virtual_memory().total / (1024 ** 3)
#     cpu    = multiprocessing.cpu_count()
#     if ram_gb < 8:
#         return 1
#     elif ram_gb < 16:
#         return 2
#     else:
#         return min(4, cpu)


# # ─────────────────────────────────────────────────────────────────────────────
# #  SINGLE-TASK PIPELINE
# # ─────────────────────────────────────────────────────────────────────────────

# def run_full_pipeline(
#     channel:      str,
#     date:         str,
#     time_sel:     list | None,
#     base_path:    str,
#     ads_language: str  | None = None,
#     specific_ads: list | None = None,
#     num_workers:  int  | None = None,
# ) -> list[dict]:
#     """
#     Run S1 → S2 → S3 for one (channel, date) task.

#     Returns a list of detection-row dicts.
#     """
#     if num_workers is None:
#         num_workers = _default_workers()

#     channel_path = os.path.join(base_path, channel)
#     t_pipeline   = time.time()

#     print(f"\n{'='*60}")
#     print(f"  PIPELINE  {channel} / {date}  (workers={num_workers})")
#     print(f"{'='*60}")

#     # ── S1: Convert videos → WAV ──────────────────────────────────────────────
#     t0 = time.time()
#     convert_ads_and_videos(
#         channel_path,
#         date,
#         time_sel,
#         ads_language  = ads_language,
#         specific_ads  = specific_ads,
#     )
#     print(f"  [S1] Conversion   : {time.time() - t0:.1f}s")

#     # ── S2: Build / update fingerprint DB ────────────────────────────────────
#     t1 = time.time()
#     with db_lock:   # FIX: serialise DB writes across concurrent tasks
#         run_flow(ads_language, specific_ads)
#     print(f"  [S2] Fingerprint  : {time.time() - t1:.1f}s")

#     # ── S3: Detect ads ────────────────────────────────────────────────────────
#     t2 = time.time()
#     detected = detect_ads(
#         language     = ads_language,
#         channel      = channel,
#         date         = date,
#         time_sel     = time_sel,
#         specific_ads = specific_ads,
#         num_workers  = num_workers,
#     )
#     print(f"  [S3] Detection    : {time.time() - t2:.1f}s")
#     print(f"  [TOTAL] Pipeline  : {time.time() - t_pipeline:.1f}s  |  {len(detected)} segment(s) found")

#     return detected


# # ─────────────────────────────────────────────────────────────────────────────
# #  TASK WRAPPER  (used by ThreadPoolExecutor)
# # ─────────────────────────────────────────────────────────────────────────────

# def run_task_wrapper(task: dict, num_workers: int | None) -> list[dict]:
#     return run_full_pipeline(
#         channel      = task["channel"],
#         date         = task["date"],
#         time_sel     = task.get("time_sel"),
#         base_path    = task["base_path"],
#         ads_language = task.get("ads_language"),
#         specific_ads = task.get("specific_ads"),
#         num_workers  = num_workers,
#     )


# # ─────────────────────────────────────────────────────────────────────────────
# #  MULTI-TASK RUNNER
# # ─────────────────────────────────────────────────────────────────────────────

# def run_all_tasks_and_save_json(
#     tasks_to_run:     list[dict],
#     output_json_path: str       = "Outputs/ALL_DETECTED_ADS.json",
#     num_workers:      int | None = None,
# ) -> list[dict]:
#     """
#     Run all tasks and aggregate results into a JSON file.

#     FIX: max outer threads set to 2 (was 5) to prevent each thread from
#     launching its own multiprocessing pool and exhausting RAM/CPU.
#     If tasks share the same channel/date, S2 runs under db_lock so the
#     SQLite DB stays consistent.
#     """
#     # Outer parallelism: run at most 2 tasks in parallel so S3's inner
#     # thread pools don't cause resource exhaustion
#     outer_workers = min(MAX_OUTER_TASKS, len(tasks_to_run))
#     inner_workers = num_workers or _default_workers()

#     print(f"\n{'#'*60}")
#     print(f"  {len(tasks_to_run)} task(s)  |  outer_threads={outer_workers}  inner_workers={inner_workers}")
#     print(f"{'#'*60}\n")

#     ALL_RESULTS: list[dict] = []

#     if outer_workers <= 1:
#         # Sequential — simplest, safest
#         for task in tasks_to_run:
#             try:
#                 rows = run_task_wrapper(task, inner_workers)
#                 ALL_RESULTS.extend(rows)
#                 print(f"  ✅  {task['channel']}/{task['date']}  → {len(rows)} segment(s)")
#             except Exception as exc:
#                 print(f"  ❌  {task['channel']}/{task['date']}  → {exc}")
#     else:
#         with concurrent.futures.ThreadPoolExecutor(max_workers=outer_workers) as pool:
#             futures = {
#                 pool.submit(run_task_wrapper, task, inner_workers): task
#                 for task in tasks_to_run
#             }
#             for future in concurrent.futures.as_completed(futures):
#                 task = futures[future]
#                 try:
#                     rows = future.result()
#                     ALL_RESULTS.extend(rows)
#                     print(f"  ✅  {task['channel']}/{task['date']}  → {len(rows)} segment(s)")
#                 except Exception as exc:
#                     print(f"  ❌  {task['channel']}/{task['date']}  → {exc}")

#     # ── Save to JSON ──────────────────────────────────────────────────────────
#     os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

#     if ALL_RESULTS:
#         df = pd.DataFrame(ALL_RESULTS)
#         sort_cols = [c for c in ["Channel", "Date", "Time", "Start Time"] if c in df.columns]
#         if sort_cols:
#             df.sort_values(by=sort_cols, inplace=True)
#         df.reset_index(drop=True, inplace=True)
#         df["Segment No"] = df.index + 1
#         df.to_json(output_json_path, orient="records", indent=4)
#         print(f"\n  🏆 {len(ALL_RESULTS)} segment(s) saved → {output_json_path}")
#         return df.to_dict(orient="records")
#     else:
#         with open(output_json_path, "w") as fh:
#             json.dump([], fh)
#         print("\n  [INFO] No ad segments detected.")
#         return []

"""
utils.py  —  Pipeline orchestration

Cancellation behaviour:
  • Each task checks the job's cancel_requested flag BEFORE acquiring the
    semaphore. If set, the task is marked 'cancelled' immediately — no pipeline
    work starts and the slot is never consumed.
  • A task that has already acquired a slot and started the pipeline runs to
    completion (hard-killing ffmpeg mid-conversion is unsafe). It is then
    marked 'cancelled' instead of 'done' if cancel was requested.
  • Job-level status: queued → running → done | cancelled | cancelling | error
  • Channel-level status: queued → running → done | cancelled | error
"""

import os
import time
import json
import psutil
import threading
import pandas as pd
import multiprocessing
import concurrent.futures

from config import MAX_OUTER_TASKS
from S1_preprocessing_aud import convert_ads_and_videos
from S2_fingerprint_db     import run_flow
from S3_scan_test_improved_latest import detect_ads

db_lock             = threading.Lock()
_pipeline_semaphore = threading.Semaphore(MAX_OUTER_TASKS)
_active_count       = 0
_active_lock        = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
#  JOB-STORE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _update_channel_state(job_id: str | None, task_index: int, **kwargs):
    if job_id is None:
        return
    try:
        import app as _app
        with _app._jobs_lock:
            _app._jobs[job_id]["channels"][task_index].update(kwargs)
    except Exception:
        pass


def _is_cancel_requested(job_id: str | None) -> bool:
    if job_id is None:
        return False
    try:
        import app as _app
        with _app._jobs_lock:
            return _app._jobs[job_id].get("cancel_requested", False)
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  RAM-AWARE WORKER COUNT
# ─────────────────────────────────────────────────────────────────────────────

def _default_workers() -> int:
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    cpu    = multiprocessing.cpu_count()
    if ram_gb < 8:    return 1
    elif ram_gb < 16: return 2
    else:             return min(4, cpu)


# ─────────────────────────────────────────────────────────────────────────────
#  SINGLE-TASK PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(
    channel:      str,
    date:         str,
    time_sel:     list | None,
    base_path:    str,
    ads_language: str  | None = None,
    specific_ads: list | None = None,
    num_workers:  int  | None = None,
) -> list[dict]:
    if num_workers is None:
        num_workers = _default_workers()

    channel_path = os.path.join(base_path, channel)
    t_pipeline   = time.time()

    print(f"\n{'='*60}")
    print(f"  PIPELINE  {channel} / {date}  (workers={num_workers})")
    print(f"{'='*60}")

    t0 = time.time()
    convert_ads_and_videos(
        channel_path, date, time_sel,
        ads_language=ads_language, specific_ads=specific_ads,
    )
    print(f"  [S1] Conversion   : {time.time() - t0:.1f}s")

    t1 = time.time()
    with db_lock:
        run_flow(ads_language, specific_ads)
    print(f"  [S2] Fingerprint  : {time.time() - t1:.1f}s")

    t2 = time.time()
    detected = detect_ads(
        language=ads_language, channel=channel, date=date,
        time_sel=time_sel, specific_ads=specific_ads, num_workers=num_workers,
    )
    print(f"  [S3] Detection    : {time.time() - t2:.1f}s")
    print(f"  [TOTAL]           : {time.time() - t_pipeline:.1f}s  |  {len(detected)} segment(s)")

    return detected


# ─────────────────────────────────────────────────────────────────────────────
#  TASK WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def run_task_wrapper(
    task:        dict,
    num_workers: int | None,
    queue_pos:   int,
    total:       int,
    job_id:      str | None = None,
    task_index:  int        = 0,
) -> list[dict]:
    global _active_count
    label = f"{task['channel']}/{task['date']}"

    # ── Check cancel BEFORE waiting for a slot ────────────────────────────────
    if _is_cancel_requested(job_id):
        print(f"  [CANCEL] #{queue_pos}/{total}  {label}  → skipped (queued)")
        _update_channel_state(job_id, task_index,
                              status="cancelled",
                              finished_at=time.time())
        return []

    # ── Wait for a free pipeline slot ─────────────────────────────────────────
    print(f"  [QUEUE  #{queue_pos}/{total}]  {label}  → waiting ...")
    _pipeline_semaphore.acquire()

    # ── Re-check cancel after acquiring slot (may have been set while waiting) ─
    if _is_cancel_requested(job_id):
        _pipeline_semaphore.release()
        print(f"  [CANCEL] #{queue_pos}/{total}  {label}  → skipped (was waiting)")
        _update_channel_state(job_id, task_index,
                              status="cancelled",
                              finished_at=time.time())
        return []

    with _active_lock:
        _active_count += 1
        active_now = _active_count

    print(f"  [START  #{queue_pos}/{total}]  {label}  (active={active_now}/{MAX_OUTER_TASKS})")
    _update_channel_state(job_id, task_index,
                          status="running", started_at=time.time())

    try:
        rows = run_full_pipeline(
            channel      = task["channel"],
            date         = task["date"],
            time_sel     = task.get("time_sel"),
            base_path    = task["base_path"],
            ads_language = task.get("ads_language"),
            specific_ads = task.get("specific_ads"),
            num_workers  = num_workers,
        )

        # If cancel was requested while this task ran, mark it cancelled
        if _is_cancel_requested(job_id):
            _update_channel_state(job_id, task_index,
                                  status="cancelled",
                                  finished_at=time.time(),
                                  segments=len(rows))
            print(f"  [CANCEL] {label}  → finished but marked cancelled")
            return []   # don't contribute results to cancelled job
        else:
            _update_channel_state(job_id, task_index,
                                  status="done",
                                  finished_at=time.time(),
                                  segments=len(rows))
            print(f"  [DONE]   {label}  → {len(rows)} segment(s)")
            return rows

    except Exception as exc:
        _update_channel_state(job_id, task_index,
                              status="error",
                              finished_at=time.time(),
                              error=str(exc))
        print(f"  [FAIL]   {label}  → {exc}")
        return []

    finally:
        _pipeline_semaphore.release()
        with _active_lock:
            _active_count -= 1


# ─────────────────────────────────────────────────────────────────────────────
#  MULTI-TASK RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_all_tasks_and_save_json(
    tasks_to_run:     list[dict],
    output_json_path: str        = "Outputs/Detection_results.json",
    num_workers:      int | None = None,
    job_id:           str | None = None,
) -> list[dict]:
    total         = len(tasks_to_run)
    inner_workers = num_workers or _default_workers()

    print(f"\n{'#'*60}")
    print(f"  {total} task(s)  |  slots={MAX_OUTER_TASKS}  workers={inner_workers}")
    print(f"{'#'*60}\n")

    if total == 0:
        return []

    ALL_RESULTS:  list[dict] = []
    results_lock = threading.Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=total) as pool:
        futures = {
            pool.submit(
                run_task_wrapper,
                task, inner_workers, idx + 1, total, job_id, idx
            ): task
            for idx, task in enumerate(tasks_to_run)
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                rows = future.result()
                with results_lock:
                    ALL_RESULTS.extend(rows)
            except Exception as exc:
                task  = futures[future]
                label = f"{task['channel']}/{task['date']}"
                print(f"  [FAIL]  {label}  → unexpected: {exc}")

    # ── Save to shared Detection_results.json ─────────────────────────────────
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    if ALL_RESULTS:
        df = pd.DataFrame(ALL_RESULTS)
        sort_cols = [c for c in ["Channel", "Date", "Time", "Start Time"] if c in df.columns]
        if sort_cols:
            df.sort_values(by=sort_cols, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["Segment No"] = df.index + 1
        df.to_json(output_json_path, orient="records", indent=4)
        print(f"\n  {len(ALL_RESULTS)} segment(s) saved → {output_json_path}")
        return df.to_dict(orient="records")
    else:
        with open(output_json_path, "w") as fh:
            json.dump([], fh)
        print("\n  [INFO] No segments detected.")
        return []