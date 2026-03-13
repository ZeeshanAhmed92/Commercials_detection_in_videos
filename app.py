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
app.py  —  Flask REST API for ad detection (fixed)

Changes vs original:
  • Startup validation: warns if ADS_SAMPLES_PATH or BASE_PATH don't exist.
  • Cleaner error messages with HTTP 422 for invalid inputs.
  • Minor: consistent use of Path-style joins.
"""

import os
from flask import Flask, request, jsonify
from utils import run_all_tasks_and_save_json
from config import BASE_PATH, ADS_SAMPLES_PATH, OUTPUT_DIR

app = Flask(__name__)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── HELPERS ───────────────────────────────────────────────────────────────────

def normalize_to_list(value):
    if value is None:        return []
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


# ── ENDPOINT ──────────────────────────────────────────────────────────────────

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

        output_json = os.path.join(OUTPUT_DIR, "Detection_results.json")
        results     = run_all_tasks_and_save_json(
            tasks_to_run,
            output_json_path = output_json,
            num_workers      = None,     # auto-detect
        )

        return jsonify({
            "status":            "success",
            "tasks_created":     len(tasks_to_run),
            "segments_detected": len(results),
            "output_file":       output_json,
            "data":              results,
        })

    except Exception as exc:
        print(f"[SERVER ERROR] {exc}")
        return jsonify({"status": "error", "message": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)