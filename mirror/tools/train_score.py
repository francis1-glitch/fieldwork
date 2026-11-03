#!/usr/bin/env python3
"""
train_score.py  —  Benchmark Mirror's composite score against human ratings.

What it does
------------
Takes the SCUT-FBP5500 dataset (5,500 faces, each rated 1-5 by 60 raters),
extracts the SAME markers Mirror computes (facial thirds, fifths, eye spacing,
lip position, symmetry, skin evenness, skin smoothness) from every image, then
fits linear weights so the composite score best predicts the human mean rating.
It reports how well it actually predicts (Pearson r, R^2) on a held-out test
split, and prints weights you paste into mirror.html (the WEIGHTS object).

This is the step that turns the score from "reasonable guess" into "fit to real
ratings" — Mirror's whole claim is that attractiveness has measurable signal, so
the score has to be shown to track real judgments, not just asserted.

Setup
-----
1. Download SCUT-FBP5500:
     https://github.com/HCIILAB/SCUT-FBP5500-Database-Release
   You need:  Images/            (the 5500 jpgs)
              train_test_files/All_labels.txt   (filename <space> mean_score)
2. pip install mediapipe opencv-python numpy scikit-learn
3. python3 train_score.py --root /path/to/SCUT-FBP5500

Notes
-----
- Landmark extraction on 5,500 images takes a while (minutes, CPU). Features are
  cached to features.npz so reruns are instant.
- If a face isn't detected in an image it's skipped (reported at the end).
"""
import argparse, os, sys, numpy as np

# Same canonical MediaPipe indices Mirror uses
IDX = dict(chin=152, forehead=10, noseBottom=2, lBrow=105, rBrow=334,
           lEyeOuter=33, lEyeInner=133, rEyeOuter=263, rEyeInner=362,
           mouthL=61, mouthR=291, upperLip=0, lowerLip=17,
           lFace=234, rFace=454, lCheek=50, rCheek=280,
           foreheadMid=151, lCheek2=205, rCheek2=425)

FEATURES = ["third_dev", "facew", "eyespace", "lip", "symmetry",
            "skin_even", "skin_smooth"]

def dist(a, b):
    return float(np.hypot(a[0]-b[0], a[1]-b[1]))

def rgb2lab_L_a(pixels):
    # pixels: Nx3 uint8 RGB -> returns (L array, a array)
    p = pixels.astype(np.float64) / 255.0
    def inv(c):
        return np.where(c > 0.04045, ((c+0.055)/1.055)**2.4, c/12.92)
    r, g, b = inv(p[:,0]), inv(p[:,1]), inv(p[:,2])
    x = (r*0.4124+g*0.3576+b*0.1805)/0.95047
    y = (r*0.2126+g*0.7152+b*0.0722)
    z = (r*0.0193+g*0.1192+b*0.9505)/1.08883
    def f(t):
        return np.where(t > 0.008856, np.cbrt(t), 7.787*t + 16/116)
    fx, fy, fz = f(x), f(y), f(z)
    L = 116*fy - 16
    a = 500*(fx-fy)
    return L, a

def extract(img_bgr, landmarker):
    import mediapipe as mp
    import cv2
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = landmarker.detect(mp_img)
    if not res.face_landmarks:
        return None
    lm = res.face_landmarks[0]
    P = {k: (lm[v].x*w, lm[v].y*h) for k, v in IDX.items()}

    browY = (P['lBrow'][1]+P['rBrow'][1])/2
    upper = abs(browY-P['forehead'][1]); mid = abs(P['noseBottom'][1]-browY)
    low = abs(P['chin'][1]-P['noseBottom'][1]); tot = upper+mid+low
    thirds = np.array([upper/tot, mid/tot, low/tot])
    third_dev = float(np.mean(np.abs(thirds - 1/3)))     # 0 = perfect canon

    eyeW = dist(P['lEyeOuter'], P['lEyeInner'])
    facew = dist(P['lFace'], P['rFace'])/eyeW
    eyespace = dist(P['lEyeInner'], P['rEyeInner'])/eyeW
    lip = abs(P['upperLip'][1]-P['noseBottom'][1]) / max(1e-6, abs(P['chin'][1]-P['lowerLip'][1]))

    midX = (P['forehead'][0]+P['chin'][0])/2
    pairs = [('lEyeOuter','rEyeOuter'),('mouthL','mouthR'),('lFace','rFace'),('lBrow','rBrow'),('lCheek','rCheek')]
    asym = 0.0
    for l, r in pairs:
        dl = abs(P[l][0]-midX); dr = abs(P[r][0]-midX)
        asym += abs(dl-dr)/((dl+dr)/2 or 1)
    symmetry = max(0.0, 1 - asym/len(pairs))

    # skin patches
    rad = max(6, int(w*0.03))
    Ls, lums = [], []
    texs = []
    for key in ['foreheadMid','lCheek','rCheek','lCheek2','rCheek2']:
        cx, cy = int(P[key][0]), int(P[key][1])
        x0, y0 = max(0,cx-rad), max(0,cy-rad)
        patch = rgb[y0:cy+rad, x0:cx+rad]
        if patch.size == 0: continue
        px = patch.reshape(-1,3)
        L, _ = rgb2lab_L_a(px)
        Ls.append(L); lums.append(L)
        gray = patch.astype(np.float64) @ np.array([0.299,0.587,0.114])
        texs.append(np.mean(np.abs(np.diff(gray, axis=1))) if gray.shape[1]>1 else 0)
    if not Ls:
        return None
    allL = np.concatenate(lums)
    skin_even = float(max(0, min(1, 1 - np.std(allL)/25)))
    skin_smooth = float(max(0, min(1, 1 - np.mean(texs)/30)))

    return np.array([third_dev, facew, eyespace, lip, symmetry, skin_even, skin_smooth])

def build_landmarker():
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    model = "face_landmarker.task"
    if not os.path.exists(model):
        sys.exit("Download face_landmarker.task first:\n"
                 "  curl -L -o face_landmarker.task "
                 "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                 "face_landmarker/float16/1/face_landmarker.task")
    opts = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model),
        running_mode=vision.RunningMode.IMAGE, num_faces=1)
    return vision.FaceLandmarker.create_from_options(opts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="SCUT-FBP5500 root folder")
    ap.add_argument("--labels", default="train_test_files/All_labels.txt")
    ap.add_argument("--images", default="Images")
    args = ap.parse_args()

    import cv2
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from scipy.stats import pearsonr

    label_path = os.path.join(args.root, args.labels)
    img_dir = os.path.join(args.root, args.images)
    pairs = []
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                pairs.append((parts[0], float(parts[1])))
    print(f"{len(pairs)} labeled faces")

    cache = "features.npz"
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        X, y = d["X"], d["y"]
        print(f"loaded cached features: {X.shape}")
    else:
        lm = build_landmarker()
        X, y, miss = [], [], 0
        for i, (fn, score) in enumerate(pairs):
            img = cv2.imread(os.path.join(img_dir, fn))
            if img is None: miss += 1; continue
            feat = extract(img, lm)
            if feat is None: miss += 1; continue
            X.append(feat); y.append(score)
            if (i+1) % 250 == 0: print(f"  {i+1}/{len(pairs)}")
        X, y = np.array(X), np.array(y)
        np.savez(cache, X=X, y=y)
        print(f"extracted {len(X)} usable, skipped {miss}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = LinearRegression().fit(Xtr, ytr)
    pred = reg.predict(Xte)
    r, _ = pearsonr(pred, yte)
    r2 = reg.score(Xte, ytr if False else yte)  # test R^2
    print("\n=== Benchmark (held-out 20%) ===")
    print(f"Pearson r : {r:.3f}")
    print(f"R^2       : {r2:.3f}")
    print("(SCUT-FBP5500 state-of-the-art deep nets reach ~0.90 r; a linear fit on"
          " a handful of geometric+skin markers will be lower — that honest number"
          " IS the result. If r is decent, the markers carry real signal.)")
    print("\nPer-marker coefficients (standardized sign shows direction):")
    for name, c in zip(FEATURES, reg.coef_):
        print(f"  {name:12s} {c:+.4f}")

    # Convert the 4 Mirror component weights from the regression, normalized to sum 1.
    # Map markers -> components the way mirror.html groups them.
    comp = {
        "proportion": abs(reg.coef_[0])+abs(reg.coef_[1])+abs(reg.coef_[2])+abs(reg.coef_[3]),
        "symmetry":   abs(reg.coef_[4]),
        "skinEven":   abs(reg.coef_[5]),
        "skinSmooth": abs(reg.coef_[6]),
    }
    tot = sum(comp.values()) or 1
    print("\nPaste into mirror.html WEIGHTS (normalized):")
    print("const WEIGHTS={ " +
          ", ".join(f"{k}:{v/tot:.2f}" for k, v in
                    [("symmetry",comp["symmetry"]),("proportion",comp["proportion"]),
                     ("skinEven",comp["skinEven"]),("skinSmooth",comp["skinSmooth"])]) +
          " };")

if __name__ == "__main__":
    main()
