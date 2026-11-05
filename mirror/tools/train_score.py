#!/usr/bin/env python3
"""
train_score.py  —  Benchmark Mirror's composite score against human ratings.

What it does
------------
Takes the SCUT-FBP5500 dataset (5,500 faces, each rated 1-5 by 60 raters),
extracts the SAME markers Mirror computes (facial thirds, fifths, eye spacing,
lip position, symmetry, skin evenness, skin smoothness) from every image, then
fits a standardized ridge model to the human mean rating, reports held-out
Pearson r / R^2, and writes ../score_model.json. Mirror loads that file and shows
a percentile against the whole dataset instead of an uncalibrated blend.

v1.6: features now match Mirror exactly — head roll is corrected, and the facial
thirds use the real hairline (trichion) from the hair/skin segmenter, because the
MediaPipe mesh stops mid-forehead (landmark 10 is not the hairline).

This is the step that turns the score from "reasonable guess" into "fit to real
ratings" — Mirror's whole claim is that attractiveness has measurable signal, so
the score has to be shown to track real judgments, not just asserted.

Setup
-----
1. Download SCUT-FBP5500:
     https://github.com/HCIILAB/SCUT-FBP5500-Database-Release
   You need:  Images/            (the 5500 jpgs)
              train_test_files/All_labels.txt   (filename <space> mean_score)
2. pip install mediapipe opencv-python numpy scikit-learn scipy
   + face_landmarker.task and selfie_multiclass_256x256.tflite next to this script
   (run_benchmark.py fetches both)
3. python3 train_score.py --root /path/to/SCUT-FBP5500

Notes
-----
- Landmark extraction on 5,500 images takes a while (minutes, CPU). Features are
  cached to features.npz so reruns are instant.
- If a face isn't detected in an image it's skipped (reported at the end).
"""
import argparse, json, os, sys, numpy as np

# Same MediaPipe indices Mirror uses
IDX = dict(chin=152, forehead=10, noseBottom=2, lBrow=105, rBrow=334,
           lEyeOuter=33, lEyeInner=133, rEyeOuter=263, rEyeInner=362,
           mouthL=61, mouthR=291, upperLip=0, lowerLip=17,
           lFace=234, rFace=454, lCheek=50, rCheek=280,
           foreheadMid=151, lCheek2=205, rCheek2=425, glabella=9, noseTip=1)

# Must match FEATS in mirror/index.html, same order
FEATURES = ["third_dev", "upper_third", "lower_third", "mid_low", "facew", "eyespace",
            "lip", "symmetry", "skin_even", "skin_smooth"]
SEG_HAIR, SEG_FACE = 1, 3
SEG_MODEL = "selfie_multiclass_256x256.tflite"
LM_MODEL = "face_landmarker.task"

def dist(a, b):
    return float(np.hypot(a[0]-b[0], a[1]-b[1]))

def rgb2lab_L_a(pixels):
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
    return 116*fy - 16, 500*(fx-fy)

def face_frame(L):
    a, b = L[IDX['lEyeOuter']], L[IDX['rEyeOuter']]
    roll = np.arctan2(b[1]-a[1], b[0]-a[0])
    c = ((a[0]+b[0])/2, (a[1]+b[1])/2); cs, sn = np.cos(-roll), np.sin(-roll)
    def rot(p):
        return (c[0]+(p[0]-c[0])*cs-(p[1]-c[1])*sn, c[1]+(p[0]-c[0])*sn+(p[1]-c[1])*cs)
    return np.degrees(roll), rot

def find_trichion(L, mask):
    """Mirror's findTrichion: walk up the face axis from landmark 10 through face skin until hair."""
    if mask is None: return None
    h, w = mask.shape
    def at(x, y):
        X, Y = int(round(x)), int(round(y))
        return -1 if (X < 0 or Y < 0 or X >= w or Y >= h) else int(mask[Y, X])
    top, chin = L[IDX['forehead']], L[IDX['chin']]
    ux, uy = top[0]-chin[0], top[1]-chin[1]; n = np.hypot(ux, uy) or 1; ux, uy = ux/n, uy/n
    off = dist(L[IDX['lEyeInner']], L[IDX['rEyeInner']])*0.35
    hits = []
    for k in (-1, 0, 1):
        x, y = top[0]-uy*off*k, top[1]+ux*off*k
        if at(x, y) != SEG_FACE: continue
        last, miss, end = (x, y), 0, -1
        for _ in range(int(n*0.8)):
            x += ux; y += uy; c = at(x, y)
            if c == SEG_FACE: last, miss = (x, y), 0; continue
            miss += 1
            if miss > 4: end = c; break
        if end == SEG_HAIR: hits.append(last)
    if not hits: return None
    hits.sort(key=lambda p: p[1]); return hits[len(hits)//2]

def extract(img_bgr, landmarker, segmenter):
    import mediapipe as mp, cv2
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Mirror analyses on a canvas at most 520px wide; match it
    if rgb.shape[1] > 520:
        s = 520/rgb.shape[1]; rgb = cv2.resize(rgb, (520, int(round(rgb.shape[0]*s))), interpolation=cv2.INTER_AREA)
    h, w = rgb.shape[:2]
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    res = landmarker.detect(mp_img)
    if not res.face_landmarks: return None
    L = [(p.x*w, p.y*h) for p in res.face_landmarks[0]]
    mask = None
    try:
        sres = segmenter.segment(mp_img)
        if sres.category_mask is not None:
            mask = sres.category_mask.numpy_view().copy()
            if mask.shape != (h, w): mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    except Exception:
        mask = None
    trich = find_trichion(L, mask)

    roll, rot = face_frame(L)
    P = {k: rot(L[v]) for k, v in IDX.items()}
    gl, sn, me = P['glabella'][1], P['noseBottom'][1], P['chin'][1]
    mid, low = abs(sn-gl), abs(me-sn)
    if trich is not None:
        up = abs(gl-rot(trich)[1]); tot = up+mid+low
        t = np.array([up/tot, mid/tot, low/tot]); third_dev = float(np.mean(np.abs(t-1/3)))
        upper_third, lower_third = float(t[0]), float(t[2])
    else:
        third_dev = upper_third = lower_third = np.nan
    eyeW = dist(P['lEyeOuter'], P['lEyeInner'])
    facew = dist(P['lFace'], P['rFace'])/eyeW
    eyespace = dist(P['lEyeInner'], P['rEyeInner'])/eyeW
    lip = abs(P['upperLip'][1]-sn) / max(1e-6, abs(me-P['lowerLip'][1]))
    midX = (P['forehead'][0]+P['chin'][0])/2
    asym = 0.0
    for l, r in [('lEyeOuter','rEyeOuter'),('mouthL','mouthR'),('lFace','rFace'),('lBrow','rBrow'),('lCheek','rCheek')]:
        dl, dr = abs(P[l][0]-midX), abs(P[r][0]-midX); asym += abs(dl-dr)/(((dl+dr)/2) or 1)
    symmetry = max(0.0, 1-asym/5)

    # skin patches (unrotated, same as Mirror)
    rad = max(6, int(round(w*0.03)))
    lums, texs = [], []
    for key in ['foreheadMid','lCheek','rCheek','lCheek2','rCheek2']:
        cx, cy = int(round(L[IDX[key]][0])), int(round(L[IDX[key]][1]))
        x0, y0 = max(0, cx-rad), max(0, cy-rad)
        patch = rgb[y0:y0+2*rad, x0:x0+2*rad]
        if patch.size == 0: continue
        Lc, _ = rgb2lab_L_a(patch.reshape(-1, 3)); lums.append(Lc)
        gray = patch.astype(np.float64) @ np.array([0.299, 0.587, 0.114])
        if gray.shape[1] > 1: texs.append(np.abs(np.diff(gray, axis=1)).ravel())
    if lums:
        skin_even = float(max(0, min(1, 1-np.std(np.concatenate(lums))/25)))
        skin_smooth = float(max(0, min(1, 1-np.mean(np.concatenate(texs))/30))) if texs else np.nan
    else:
        skin_even = skin_smooth = np.nan
    return np.array([third_dev, upper_third, lower_third, mid/(mid+low), facew, eyespace,
                     lip, symmetry, skin_even, skin_smooth], dtype=float)

def build_tasks():
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    for m in (LM_MODEL, SEG_MODEL):
        if not os.path.exists(m): sys.exit(f"missing {m}: run run_benchmark.py, which downloads it")
    lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=LM_MODEL),
        running_mode=vision.RunningMode.IMAGE, num_faces=1))
    seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=python.BaseOptions(model_asset_path=SEG_MODEL),
        running_mode=vision.RunningMode.IMAGE, output_category_mask=True, output_confidence_masks=False))
    return lm, seg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="SCUT-FBP5500 root folder")
    ap.add_argument("--labels", default="train_test_files/All_labels.txt")
    ap.add_argument("--images", default="Images")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "score_model.json"))
    args = ap.parse_args()

    import cv2
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import train_test_split
    from scipy.stats import pearsonr

    pairs = []
    with open(os.path.join(args.root, args.labels)) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2: pairs.append((parts[0], float(parts[1])))
    print(f"{len(pairs)} labeled faces")

    cache = "features_v2.npz"          # v2 = trichion + roll-corrected features
    if os.path.exists(cache):
        d = np.load(cache); X, y = d["X"], d["y"]; print(f"loaded cached features: {X.shape}")
    else:
        lm, seg = build_tasks()
        X, y, miss = [], [], 0
        for i, (fn, score) in enumerate(pairs):
            img = cv2.imread(os.path.join(args.root, args.images, fn))
            feat = None if img is None else extract(img, lm, seg)
            if feat is None: miss += 1; continue
            X.append(feat); y.append(score)
            if (i+1) % 250 == 0: print(f"  {i+1}/{len(pairs)}")
        X, y = np.array(X), np.array(y)
        np.savez(cache, X=X, y=y)
        print(f"extracted {len(X)} usable, skipped {miss}")

    found = float(np.mean(~np.isnan(X[:, 0])))
    print(f"hairline found on {found:.1%} of faces")
    mean = np.nanmean(X, axis=0); std = np.nanstd(X, axis=0); std[std == 0] = 1
    Z = (X-mean)/std; Z = np.where(np.isnan(Z), 0.0, Z)     # missing marker = dataset average (Mirror does the same)

    Ztr, Zte, ytr, yte = train_test_split(Z, y, test_size=0.2, random_state=42)
    reg = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Ztr, ytr)
    pred = reg.predict(Zte)
    r = float(pearsonr(pred, yte)[0]); r2 = float(reg.score(Zte, yte))
    print("\n=== Benchmark (held-out 20%) ===")
    print(f"Pearson r : {r:.3f}\nR^2       : {r2:.3f}\n(alpha {reg.alpha_:.3g})")
    print("\nStandardized coefficients (rating points per 1 SD of the marker):")
    for name, c in sorted(zip(FEATURES, reg.coef_), key=lambda t: -abs(t[1])):
        print(f"  {name:12s} {c:+.4f}")

    # refit on everything for the shipped model; percentiles from its predictions on all faces
    final = RidgeCV(alphas=[reg.alpha_]).fit(Z, y)
    cdf = np.quantile(final.predict(Z), np.linspace(0, 1, 201)).round(4).tolist()
    model = dict(version=2, dataset="SCUT-FBP5500", n=int(len(y)), features=FEATURES,
                 mean=mean.round(6).tolist(), std=std.round(6).tolist(),
                 coef=final.coef_.round(6).tolist(), intercept=round(float(final.intercept_), 6),
                 cdf=cdf, r_test=round(r, 4), r2_test=round(r2, 4), hairline_found=round(found, 4),
                 rating_mean=round(float(y.mean()), 4), rating_sd=round(float(y.std()), 4))
    with open(args.out, "w") as f: json.dump(model, f)
    print(f"\nwrote {os.path.abspath(args.out)}  — commit it (mirror/score_model.json) and Mirror switches to percentiles")

if __name__ == "__main__":
    main()
