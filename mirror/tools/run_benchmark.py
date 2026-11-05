#!/usr/bin/env python3
"""
run_benchmark.py — one command to benchmark Mirror's score on your own machine.

Mirror's sandbox can't reach the dataset (Google Drive) or the MediaPipe model
(Google CDN), so this step runs here, on your laptop, which can. It:
  1. installs the Python deps
  2. downloads the MediaPipe face_landmarker model
  3. downloads + unzips SCUT-FBP5500 (5,500 human-rated faces) from Google Drive
  4. runs train_score.py, which extracts Mirror's markers, fits the score to the
     human ratings, reports held-out accuracy, and writes score_model.json

Usage:
    python run_benchmark.py

It writes mirror/score_model.json. Commit and push that one file (or send it to
Claude) and Mirror switches from the uncalibrated index to a real percentile.

Notes:
- First run downloads ~172MB and processes 5,500 images — expect a few minutes.
- If the Google Drive link hits its daily quota, open the Drive link in the
  README by hand, download SCUT-FBP5500.zip, drop it next to this script, and
  rerun — it'll skip the download and use the local zip.
"""
import os, sys, subprocess, zipfile, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "SCUT-FBP5500")
ZIP_PATH = os.path.join(HERE, "SCUT-FBP5500.zip")
MODEL = os.path.join(HERE, "face_landmarker.task")
DRIVE_ID = "1w0TorBfTIqbquQVd6k3h_77ypnrvfGwf"          # from the SCUT README
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
SEG = os.path.join(HERE, "selfie_multiclass_256x256.tflite")
SEG_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
           "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite")

def pipi(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

def ensure_deps():
    print("[1/4] installing deps (mediapipe, opencv, sklearn, scipy, gdown)…")
    try:
        pipi("mediapipe", "opencv-python", "scikit-learn", "scipy", "numpy", "gdown")
    except subprocess.CalledProcessError as e:
        sys.exit(f"pip install failed: {e}\nInstall them manually and rerun.")

def ensure_model():
    for path, url, name in ((MODEL, MODEL_URL, "face landmarker"), (SEG, SEG_URL, "hair/skin segmenter")):
        if os.path.exists(path):
            print(f"[2/4] {name} model already present."); continue
        print(f"[2/4] downloading {name} model…")
        urllib.request.urlretrieve(url, path)

def ensure_dataset():
    if os.path.isdir(os.path.join(DATA_DIR, "Images")):
        print("[3/4] dataset already present."); return
    if not os.path.exists(ZIP_PATH):
        print("[3/4] downloading SCUT-FBP5500 from Google Drive (~172MB)…")
        import gdown
        try:
            gdown.download(id=DRIVE_ID, output=ZIP_PATH, quiet=False)
        except Exception as e:
            sys.exit(f"Drive download failed ({e}).\n"
                     "Open the Drive link in the SCUT README, download "
                     "SCUT-FBP5500.zip into this folder, and rerun.")
    print("    unzipping…")
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(HERE)
    # the zip may nest one level; normalise to DATA_DIR
    if not os.path.isdir(os.path.join(DATA_DIR, "Images")):
        for name in os.listdir(HERE):
            cand = os.path.join(HERE, name)
            if os.path.isdir(cand) and os.path.isdir(os.path.join(cand, "Images")):
                if cand != DATA_DIR:
                    os.replace(cand, DATA_DIR)
                break

def run_training():
    print("[4/4] extracting markers + fitting score (this is the slow part)…")
    # train_score.py looks for face_landmarker.task in its CWD
    subprocess.check_call([sys.executable, os.path.join(HERE, "train_score.py"),
                           "--root", DATA_DIR], cwd=HERE)

if __name__ == "__main__":
    ensure_deps()
    ensure_model()
    ensure_dataset()
    run_training()
    print("\nDone. Commit mirror/score_model.json (or send it to Claude).")
