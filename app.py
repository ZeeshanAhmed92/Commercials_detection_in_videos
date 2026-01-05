import os
from flask import Flask, request, jsonify
from utils import run_all_tasks_and_save_json

app = Flask(__name__)

# === CONFIG ===
BASE_PATH = "/run/user/1000/gvfs/smb-share:server=192.168.39.4,share=mps/disk1/disk1-recordings"
ADS_SAMPLES_PATH = "Inputs/ads_samples"
OUTPUT_DIR = "Outputs/API_Results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === HELPERS ===

def normalize_to_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return value
    return []

def get_all_languages():
    if not os.path.exists(ADS_SAMPLES_PATH):
        return ["Hindi"]
    langs = [
        d for d in os.listdir(ADS_SAMPLES_PATH)
        if os.path.isdir(os.path.join(ADS_SAMPLES_PATH, d))
    ]
    return langs if langs else ["Hindi"]

def get_all_dates_for_channel(channel_path):
    if not os.path.exists(channel_path):
        return []
    return sorted([
        d for d in os.listdir(channel_path)
        if os.path.isdir(os.path.join(channel_path, d))
        and d.isdigit()
        and len(d) == 8
    ])

# === API ENDPOINT ===

@app.route('/detect', methods=['POST'])
def detect_ads():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # -----------------------------
        # 1. Parse request
        # -----------------------------
        channels      = normalize_to_list(data.get("channel"))
        input_dates   = normalize_to_list(data.get("date"))
        input_langs   = normalize_to_list(data.get("language"))
        target_times  = normalize_to_list(data.get("time")) or None
        target_ads    = normalize_to_list(data.get("specific_ads")) or None

        if not channels:
            return jsonify({"error": "Channel is required"}), 400

        # -----------------------------
        # 2. Languages
        # -----------------------------
        languages = input_langs if input_langs else get_all_languages()

        tasks_to_run = []

        # -----------------------------
        # 3. Task generation
        # -----------------------------
        for ch in channels:
            channel_path = os.path.join(BASE_PATH, ch)
            if not os.path.exists(channel_path):
                print(f"[WARN] Channel not found: {channel_path}")
                continue

            dates_to_process = (
                input_dates if input_dates else get_all_dates_for_channel(channel_path)
            )

            for dt in dates_to_process:
                date_path = os.path.join(channel_path, dt)
                if not os.path.exists(date_path):
                    print(f"[WARN] Date not found: {date_path}")
                    continue

                for lang in languages:
                    # Logic for creating task parameters
                    task_params = {
                        "channel": ch,
                        "date": dt,
                        "base_path": BASE_PATH,
                        "ads_language": lang,
                        "specific_ads": target_ads  # Passing the specific ads list
                    }

                    # ------------------------------------
                    # CASE A: No time provided → all files in folder
                    # ------------------------------------
                    if not target_times:
                        task_params["time_sel"] = None
                        tasks_to_run.append(task_params)

                    # ------------------------------------
                    # CASE B: Time provided → match prefix
                    # ------------------------------------
                    else:
                        all_files = [
                            f for f in os.listdir(date_path)
                            if f.lower().endswith((
                                ".mp4", ".mkv", ".avi",
                                ".mov", ".mpeg", ".mpd.mp4"
                            ))
                        ]

                        files_to_process = [
                            f for f in all_files
                            if os.path.splitext(f)[0].split('.')[0] in target_times
                        ]

                        if files_to_process:
                            task_params["time_sel"] = files_to_process
                            tasks_to_run.append(task_params)
                        else:
                            print(
                                f"[WARN] No matching files for times "
                                f"{target_times} in {ch}/{dt}"
                            )

        # -----------------------------
        # 4. Validation
        # -----------------------------
        if not tasks_to_run:
            return jsonify({
                "status": "failed",
                "message": "No valid tasks generated. Check channel/date/time inputs."
            }), 404

        # -----------------------------
        # 5. Execute + Save JSON
        # -----------------------------
        output_json = os.path.join(OUTPUT_DIR, "Detection_results.json")

        # Note: Ensure your utils.run_all_tasks_and_save_json is updated 
        # to handle the "specific_ads" key in the task dictionaries.
        results = run_all_tasks_and_save_json(
            tasks_to_run,
            output_json_path=output_json,
            num_workers=None
        )

        return jsonify({
            "status": "success",
            "tasks_created": len(tasks_to_run),
            "segments_detected": len(results),
            "output_file": output_json,
            "data": results
        })

    except Exception as e:
        print(f"[SERVER ERROR] {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)