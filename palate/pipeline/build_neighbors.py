#!/usr/bin/env python3
"""Palate lab: "fans of X also liked Y" lists for testing a better model.

Runs on GitHub Actions (the Claude session can't reach these servers). Builds
palate/data/lab/neighbors.json with two kinds of neighbour lists for titles
already in palate/data/catalog.json:

  ml    movie -> its 40 most similar movies, learned from MovieLens 32M ratings
        (item-item cosine on ratings with each person's average and each movie's
        quality removed, shrunk when few people rated both). Only the derived
        similarities are published, never ratings.
  tmdb  movie or show -> TMDB's "recommendations" list (page 1), kept only where
        the recommended title is in the catalog.

Nothing personal is in the file. Your ratings stay on your phone.

Env: TMDB_API_KEY (repo secret). Flags: --skip-ml, --skip-tmdb, --limit N.
"""
import argparse, io, json, os, sys, time, zipfile, datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_catalog import TMDB, ML_URLS, CAT_PATH, DATA  # reuse the same client and paths

OUT_DIR = os.path.join(DATA, "lab")
OUT = os.path.join(OUT_DIR, "neighbors.json")
TOPK = 40
SHRINK = 50.0         # similarity * n / (n + SHRINK), n = people who rated both
MIN_ITEM = 40
MIN_USER = 20
BLOCK = 800
THREADS = 16


def catalog_ids():
    with open(CAT_PATH) as f:
        cat = json.load(f)
    i = cat["fields"].index("id")
    return [r[i] for r in cat["items"]]


def ml_neighbors(movie_ids):
    import pandas as pd
    from scipy.sparse import csr_matrix

    want = {int(x[1:]) for x in movie_ids if x.startswith("m")}
    blob = None
    for url in ML_URLS:
        try:
            print("downloading", url, flush=True)
            r = requests.get(url, timeout=900); r.raise_for_status(); blob = r.content; break
        except Exception as e:
            print("  failed:", e, flush=True)
    if blob is None:
        raise RuntimeError("could not download MovieLens")
    z = zipfile.ZipFile(io.BytesIO(blob))
    base = z.namelist()[0].split("/")[0]
    links = pd.read_csv(z.open(f"{base}/links.csv"), usecols=["movieId", "tmdbId"]).dropna()
    links["tmdbId"] = links["tmdbId"].astype(int)
    links = links[links["tmdbId"].isin(want)]
    rt = pd.read_csv(z.open(f"{base}/ratings.csv"), usecols=["userId", "movieId", "rating"],
                     dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32})
    del blob, z
    rt = rt[rt["movieId"].isin(set(links["movieId"]))]
    for _ in range(2):
        ic = rt["movieId"].value_counts(); rt = rt[rt["movieId"].isin(ic[ic >= MIN_ITEM].index)]
        uc = rt["userId"].value_counts(); rt = rt[rt["userId"].isin(uc[uc >= MIN_USER].index)]
    print("ratings used:", len(rt), "movies:", rt["movieId"].nunique(), "users:", rt["userId"].nunique(), flush=True)

    rt = rt.copy()
    rt["r"] = rt["rating"] - rt.groupby("userId")["rating"].transform("mean")
    g = rt.groupby("movieId")["r"]
    bias = g.sum() / (g.count() + 25.0)
    rt["r"] = (rt["r"] - rt["movieId"].map(bias)).astype(np.float32)

    u_codes, u_idx = np.unique(rt["userId"].values, return_inverse=True)
    m_codes, m_idx = np.unique(rt["movieId"].values, return_inverse=True)
    n_items = len(m_codes)
    X = csr_matrix((rt["r"].values, (u_idx, m_idx)), shape=(len(u_codes), n_items), dtype=np.float32).tocsc()
    Bn = X.copy(); Bn.data[:] = 1.0
    del rt
    norm = np.sqrt(np.asarray(X.multiply(X).sum(axis=0)).ravel()) + 1e-6
    XT, BT = X.T.tocsr(), Bn.T.tocsr()

    ml2tmdb = dict(zip(links["movieId"], links["tmdbId"]))
    tid = ["m" + str(ml2tmdb[int(m)]) for m in m_codes]
    out = {}
    t0 = time.time()
    for j0 in range(0, n_items, BLOCK):
        j1 = min(n_items, j0 + BLOCK)
        G = (XT[j0:j1] @ X).toarray()
        C = (BT[j0:j1] @ Bn).toarray()
        S = G / (norm[j0:j1, None] * norm[None, :]) * (C / (C + SHRINK))
        for a in range(j1 - j0):
            S[a, j0 + a] = -9
        top = np.argpartition(-S, TOPK, axis=1)[:, :TOPK]
        for a in range(j1 - j0):
            idx = top[a][np.argsort(-S[a, top[a]])]
            out[tid[j0 + a]] = [[tid[b], round(float(S[a, b]), 3)] for b in idx if S[a, b] > 0]
        print(f"  ml block {j0}-{j1} of {n_items} ({time.time()-t0:.0f}s)", flush=True)
    return out


def tmdb_neighbors(ids, limit=None):
    key = os.environ.get("TMDB_API_KEY", "")
    if not key:
        raise RuntimeError("TMDB_API_KEY missing")
    api = TMDB(key)
    have = set(ids)
    todo = ids[:limit] if limit else ids
    out = {}

    def one(x):
        kind = "movie" if x.startswith("m") else "tv"
        j = api.get(f"/{kind}/{x[1:]}/recommendations", page=1)
        if not j:
            return x, []
        pre = "m" if kind == "movie" else "t"
        recs = []
        for r in j.get("results", []):
            rk = r.get("media_type") or kind
            rid = ("m" if rk == "movie" else "t") + str(r.get("id"))
            if rid in have and rid != x:
                recs.append(rid)
        return x, recs

    t0 = time.time()
    with ThreadPoolExecutor(THREADS) as ex:
        futs = [ex.submit(one, x) for x in todo]
        for n, f in enumerate(as_completed(futs), 1):
            x, recs = f.result()
            if recs:
                out[x] = recs
            if n % 1000 == 0:
                print(f"  tmdb {n}/{len(todo)} ({time.time()-t0:.0f}s, {api.calls} calls)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ml", action="store_true")
    ap.add_argument("--skip-tmdb", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    ids = catalog_ids()
    print("catalog titles:", len(ids), flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    prev = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            prev = json.load(f)
    res = {"built": dt.date.today().isoformat(), "topk": TOPK, "shrink": SHRINK}
    res["ml"] = prev.get("ml", {}) if a.skip_ml else ml_neighbors(ids)
    res["tmdb"] = prev.get("tmdb", {}) if a.skip_tmdb else tmdb_neighbors(ids, a.limit)
    with open(OUT, "w") as f:
        json.dump(res, f, separators=(",", ":"))
    print("wrote", OUT, os.path.getsize(OUT) // 1024, "KB;", len(res["ml"]), "ml lists,", len(res["tmdb"]), "tmdb lists", flush=True)


if __name__ == "__main__":
    main()
