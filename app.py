# """
# app.py  —  Flask REST API for ad detection (fixed)

# Changes vs original:
#   • Startup validation: warns if ADS_SAMPLES_PATH or BASE_PATH don't exist.
#   • Cleaner error messages with HTTP 422 for invalid inputs.
#   • Minor: consistent use of Path-style joins.
# """

# import os
# from flask import Flask, request, jsonify
# from utils import run_all_tasks_and_save_json
# from config import BASE_PATH, ADS_SAMPLES_PATH, OUTPUT_DIR

# app = Flask(__name__)

# os.makedirs(OUTPUT_DIR, exist_ok=True)


# # ── HELPERS ───────────────────────────────────────────────────────────────────

# def normalize_to_list(value):
#     if value is None:        return []
#     if isinstance(value, str):  return [value]
#     if isinstance(value, list): return value
#     return []


# def get_all_languages() -> list[str]:
#     if not os.path.exists(ADS_SAMPLES_PATH):
#         return ["Hindi"]
#     langs = [
#         d for d in os.listdir(ADS_SAMPLES_PATH)
#         if os.path.isdir(os.path.join(ADS_SAMPLES_PATH, d))
#     ]
#     return langs or ["Hindi"]


# def get_all_dates_for_channel(channel_path: str) -> list[str]:
#     if not os.path.exists(channel_path):
#         return []
#     return sorted([
#         d for d in os.listdir(channel_path)
#         if os.path.isdir(os.path.join(channel_path, d))
#         and d.isdigit()
#         and len(d) == 8
#     ])


# # ── ENDPOINT ──────────────────────────────────────────────────────────────────

# @app.route("/detect", methods=["POST"])
# def detect_ads_endpoint():
#     try:
#         data = request.get_json()
#         if not data:
#             return jsonify({"error": "No JSON body provided"}), 400

#         channels     = normalize_to_list(data.get("channel"))
#         input_dates  = normalize_to_list(data.get("date"))
#         input_langs  = normalize_to_list(data.get("language"))
#         target_times = normalize_to_list(data.get("time")) or None
#         target_ads   = normalize_to_list(data.get("specific_ads")) or None

#         if not channels:
#             return jsonify({"error": "'channel' field is required"}), 422

#         languages    = input_langs or get_all_languages()
#         tasks_to_run = []

#         for ch in channels:
#             channel_path = os.path.join(BASE_PATH, ch)
#             if not os.path.exists(channel_path):
#                 print(f"[WARN] Channel path not found: {channel_path}")
#                 continue

#             dates = input_dates or get_all_dates_for_channel(channel_path)

#             for dt in dates:
#                 date_path = os.path.join(channel_path, dt)
#                 if not os.path.exists(date_path):
#                     print(f"[WARN] Date path not found: {date_path}")
#                     continue

#                 for lang in languages:
#                     base_params = {
#                         "channel":      ch,
#                         "date":         dt,
#                         "base_path":    BASE_PATH,
#                         "ads_language": lang,
#                         "specific_ads": target_ads,
#                     }

#                     if not target_times:
#                         tasks_to_run.append({**base_params, "time_sel": None})
#                     else:
#                         video_exts = (
#                             ".mp4", ".mkv", ".avi", ".mov",
#                             ".mpeg", ".mpd.mp4", ".ts",
#                         )
#                         all_files = [
#                             f for f in os.listdir(date_path)
#                             if f.lower().endswith(video_exts)
#                         ]
#                         matched = [
#                             f for f in all_files
#                             if os.path.splitext(f)[0].split(".")[0] in target_times
#                         ]
#                         if matched:
#                             tasks_to_run.append({**base_params, "time_sel": matched})
#                         else:
#                             print(f"[WARN] No files matched times {target_times} in {ch}/{dt}")

#         if not tasks_to_run:
#             return jsonify({
#                 "status":  "failed",
#                 "message": "No valid tasks generated — check channel / date / time inputs.",
#             }), 404

#         output_json = os.path.join(OUTPUT_DIR, "Detection_results.json")
#         results     = run_all_tasks_and_save_json(
#             tasks_to_run,
#             output_json_path = output_json,
#             num_workers      = None,     # auto-detect
#         )

#         return jsonify({
#             "status":            "success",
#             "tasks_created":     len(tasks_to_run),
#             "segments_detected": len(results),
#             "output_file":       output_json,
#             "data":              results,
#         })

#     except Exception as exc:
#         print(f"[SERVER ERROR] {exc}")
#         return jsonify({"status": "error", "message": str(exc)}), 500


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5002, debug=True)


"""
app.py  —  Flask REST API for ad detection

Endpoints
─────────
POST   /detect              Submit a new detection job (async).
GET    /job/<job_id>        Live status — per-channel queue/run/done state.
GET    /jobs                Summary list of all jobs (all statuses).
POST   /job/<job_id>/cancel Request cancellation of a queued or running job.
"""

import os
import uuid
import time
import threading
from flask import Flask, request, jsonify

from utils import run_all_tasks_and_save_json
from config import BASE_PATH, ADS_SAMPLES_PATH, OUTPUT_DIR, MAX_OUTER_TASKS

app = Flask(__name__)
os.makedirs(OUTPUT_DIR, exist_ok=True)

SHARED_OUTPUT_JSON = os.path.join(OUTPUT_DIR, "Detection_results.json")


# ═══════════════════════════════════════════════════════════════════════════════
#  IN-MEMORY JOB STORE
# ═══════════════════════════════════════════════════════════════════════════════

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _new_job(job_id: str, tasks: list[dict]) -> dict:
    return {
        "job_id":            job_id,
        # queued | running | done | cancelling | cancelled | error
        "status":            "queued",
        "submitted_at":      time.time(),
        "started_at":        None,
        "finished_at":       None,
        "cancel_requested":  False,   # flag polled by utils workers
        "total_tasks":       len(tasks),
        "channels": [
            {
                "channel":     t["channel"],
                "date":        t["date"],
                "language":    t.get("ads_language"),
                # queued | running | done | cancelled | error
                "status":      "queued",
                "started_at":  None,
                "finished_at": None,
                "segments":    None,
                "error":       None,
            }
            for t in tasks
        ],
        "segments_detected": None,
        "output_file":       None,
        "error":             None,
    }


def _job_summary(job: dict) -> dict:
    ch      = job["channels"]
    queued  = sum(1 for c in ch if c["status"] == "queued")
    running = sum(1 for c in ch if c["status"] == "running")
    done    = sum(1 for c in ch if c["status"] == "done")
    cancelled = sum(1 for c in ch if c["status"] == "cancelled")
    errors  = sum(1 for c in ch if c["status"] == "error")

    elapsed = None
    if job["started_at"]:
        end     = job["finished_at"] or time.time()
        elapsed = round(end - job["started_at"], 1)

    return {
        "job_id":            job["job_id"],
        "status":            job["status"],
        "submitted_at":      job["submitted_at"],
        "started_at":        job["started_at"],
        "finished_at":       job["finished_at"],
        "elapsed_sec":       elapsed,
        "cancel_requested":  job["cancel_requested"],
        "total_tasks":       job["total_tasks"],
        "active_slots":      MAX_OUTER_TASKS,
        "tasks_queued":      queued,
        "tasks_running":     running,
        "tasks_done":        done,
        "tasks_cancelled":   cancelled,
        "tasks_error":       errors,
        "segments_detected": job["segments_detected"],
        "output_file":       job["output_file"],
        "error":             job["error"],
        "channels": [
            {
                "channel":    c["channel"],
                "date":       c["date"],
                "language":   c["language"],
                "status":     c["status"],
                "segments":   c["segments"],
                "elapsed_sec": (
                    round(c["finished_at"] - c["started_at"], 1)
                    if c["started_at"] and c["finished_at"] else
                    round(time.time() - c["started_at"], 1)
                    if c["started_at"] else None
                ),
                "error":      c["error"],
            }
            for c in job["channels"]
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  BACKGROUND RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def _run_job_background(job_id: str, tasks_to_run: list[dict]):
    with _jobs_lock:
        _jobs[job_id]["status"]     = "running"
        _jobs[job_id]["started_at"] = time.time()

    try:
        results = run_all_tasks_and_save_json(
            tasks_to_run,
            output_json_path = SHARED_OUTPUT_JSON,
            num_workers      = None,
            job_id           = job_id,
        )

        with _jobs_lock:
            job = _jobs[job_id]
            ch  = job["channels"]

            # Determine final job-level status from channel outcomes
            any_done      = any(c["status"] == "done"      for c in ch)
            all_cancelled = all(c["status"] == "cancelled" for c in ch)

            if all_cancelled:
                final_status = "cancelled"
            elif job["cancel_requested"] and not any_done:
                final_status = "cancelled"
            else:
                final_status = "done"

            job["status"]            = final_status
            job["finished_at"]       = time.time()
            job["segments_detected"] = len(results)
            job["output_file"]       = SHARED_OUTPUT_JSON if results else None

    except Exception as exc:
        with _jobs_lock:
            _jobs[job_id]["status"]      = "error"
            _jobs[job_id]["finished_at"] = time.time()
            _jobs[job_id]["error"]       = str(exc)
        print(f"[JOB {job_id}] ERROR: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_to_list(value):
    if value is None:           return []
    if isinstance(value, str):  return [value]
    if isinstance(value, list): return value
    return []


def get_all_languages() -> list[str]:
    if not os.path.exists(ADS_SAMPLES_PATH):
        return ["Hindi"]
    langs = [
        d for d in os.listdir(ADS_SAMPLES_PATH)
        if os.path.isdir(os.path.join(ADS_SAMPLES_PATH, d))
    ]
    return langs or ["Hindi"]


def get_all_dates_for_channel(channel_path: str) -> list[str]:
    if not os.path.exists(channel_path):
        return []
    return sorted([
        d for d in os.listdir(channel_path)
        if os.path.isdir(os.path.join(channel_path, d))
        and d.isdigit()
        and len(d) == 8
    ])


# ═══════════════════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

# ── POST /detect ──────────────────────────────────────────────────────────────

@app.route("/detect", methods=["POST"])
def detect_ads_endpoint():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        channels     = normalize_to_list(data.get("channel"))
        input_dates  = normalize_to_list(data.get("date"))
        input_langs  = normalize_to_list(data.get("language"))
        target_times = normalize_to_list(data.get("time")) or None
        target_ads   = normalize_to_list(data.get("specific_ads")) or None

        if not channels:
            return jsonify({"error": "'channel' field is required"}), 422

        languages    = input_langs or get_all_languages()
        tasks_to_run = []

        for ch in channels:
            channel_path = os.path.join(BASE_PATH, ch)
            if not os.path.exists(channel_path):
                print(f"[WARN] Channel path not found: {channel_path}")
                continue

            dates = input_dates or get_all_dates_for_channel(channel_path)

            for dt in dates:
                date_path = os.path.join(channel_path, dt)
                if not os.path.exists(date_path):
                    print(f"[WARN] Date path not found: {date_path}")
                    continue

                for lang in languages:
                    base_params = {
                        "channel":      ch,
                        "date":         dt,
                        "base_path":    BASE_PATH,
                        "ads_language": lang,
                        "specific_ads": target_ads,
                    }

                    if not target_times:
                        tasks_to_run.append({**base_params, "time_sel": None})
                    else:
                        video_exts = (
                            ".mp4", ".mkv", ".avi", ".mov",
                            ".mpeg", ".mpd.mp4", ".ts",
                        )
                        all_files = [
                            f for f in os.listdir(date_path)
                            if f.lower().endswith(video_exts)
                        ]
                        matched = [
                            f for f in all_files
                            if os.path.splitext(f)[0].split(".")[0] in target_times
                        ]
                        if matched:
                            tasks_to_run.append({**base_params, "time_sel": matched})
                        else:
                            print(f"[WARN] No files matched times {target_times} in {ch}/{dt}")

        if not tasks_to_run:
            return jsonify({
                "status":  "failed",
                "message": "No valid tasks generated — check channel / date / time inputs.",
            }), 404

        job_id = str(uuid.uuid4())
        with _jobs_lock:
            _jobs[job_id] = _new_job(job_id, tasks_to_run)

        threading.Thread(
            target = _run_job_background,
            args   = (job_id, tasks_to_run),
            daemon = True,
            name   = f"job-{job_id[:8]}",
        ).start()

        return jsonify({
            "status":      "accepted",
            "job_id":      job_id,
            "total_tasks": len(tasks_to_run),
            "message":     f"Job queued. Poll GET /job/{job_id} for live status.",
        }), 202

    except Exception as exc:
        print(f"[SERVER ERROR] {exc}")
        return jsonify({"status": "error", "message": str(exc)}), 500


# ── GET /job/<job_id> ─────────────────────────────────────────────────────────

@app.route("/job/<job_id>", methods=["GET"])
def get_job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": f"Job '{job_id}' not found"}), 404
    return jsonify(_job_summary(job)), 200


# ── GET /jobs ─────────────────────────────────────────────────────────────────

@app.route("/jobs", methods=["GET"])
def list_jobs():
    # Optional filter: ?status=running  or  ?status=done,cancelled
    status_filter = request.args.get("status")
    allowed = set(status_filter.split(",")) if status_filter else None

    with _jobs_lock:
        jobs_copy = list(_jobs.values())

    summaries = []
    for job in jobs_copy:
        if allowed and job["status"] not in allowed:
            continue
        s = _job_summary(job)
        s.pop("channels", None)   # keep list lean
        summaries.append(s)

    summaries.sort(key=lambda j: j["submitted_at"], reverse=True)
    return jsonify({"total": len(summaries), "jobs": summaries}), 200


# ── POST /job/<job_id>/cancel ─────────────────────────────────────────────────

@app.route("/job/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return jsonify({"error": f"Job '{job_id}' not found"}), 404

        if job["status"] in ("done", "cancelled", "error"):
            return jsonify({
                "error":  f"Job is already '{job['status']}' — cannot cancel.",
                "job_id": job_id,
                "status": job["status"],
            }), 409

        if job["cancel_requested"]:
            return jsonify({
                "message": "Cancellation already requested.",
                "job_id":  job_id,
                "status":  job["status"],
            }), 200

        # Set the flag — workers in utils.py poll this before starting
        job["cancel_requested"] = True
        job["status"]           = "cancelling"

    print(f"[JOB {job_id}] Cancellation requested.")
    return jsonify({
        "message": (
            "Cancellation requested. Queued channels will be skipped immediately. "
            "Any channel currently running will be marked cancelled once it finishes."
        ),
        "job_id":  job_id,
        "status":  "cancelling",
    }), 202


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)