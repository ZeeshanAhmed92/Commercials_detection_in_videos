from S1_preprocessing_aud import convert_ads_and_videos
from S2_fingerprint_db import run_flow
from S3_scan_test_improved_latest import detect_ads
import time


if __name__ == "__main__":
    t0 = time.time()
    convert_ads_and_videos()
    print("Convert time:", time.time() - t0)
    t1 = time.time()
    run_flow()
    print("Fingerprint time:", time.time() - t1)
    t2 = time.time()
    detect_ads()
    print("Detect time:", time.time() - t2)
    print("Total time:", time.time() - t0)