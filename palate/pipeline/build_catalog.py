#!/usr/bin/env python3
"""Palate catalog builder. Runs on GitHub Actions, never on the phone.

Builds two public data files the app reads:
  palate/data/catalog.json    movies + TV with features (genres, keywords, people,
                              crowd score) and a "people like you" taste vector per
                              movie learned from MovieLens.
  palate/data/providers.json  US streaming services per title (TMDB / JustWatch).

Nothing personal is in either file. Your ratings never leave the phone.

Env:
  TMDB_API_KEY  v3 API key or v4 read access token (repo secret).
Flags:
  --rebuild-crowd   recompute MovieLens taste vectors (otherwise reused).
  --limit N         only fetch N titles (testing).

Rules we follow: TMDB data is refreshed before it is 6 months old (we refetch
anything older than REFRESH_DAYS), and the app shows the TMDB attribution.
MovieLens is used for research/non-commercial purposes; only derived,
low-dimensional factors are published, never the ratings themselves.
"""
import argparse, base64, io, json, os, sys, time, zipfile, datetime as dt
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CAT_PATH = os.path.join(DATA, "catalog.json")
PROV_PATH = os.path.join(DATA, "providers.json")
OV_PATH = os.path.join(DATA, "overviews.json")
API = "https://api.themoviedb.org/3"
REGION = "US"
REFRESH_DAYS = 150          # well inside TMDB's 6-month cache limit
ML_URLS = ["https://files.grouplens.org/datasets/movielens/ml-32m.zip",
           "https://files.grouplens.org/datasets/movielens/ml-25m.zip"]
ML_MIN_ITEM = 40            # a movie needs this many MovieLens ratings to get a taste vector
ML_MIN_USER = 20
K = 32                      # taste-vector dimensions
MIN_FEATURE_ITEMS = 3       # keywords/people on fewer titles than this are dropped
TV_PAGES = 150              # 20 shows per page, most-voted first
NEW_PAGES = 40              # recent movies MovieLens doesn't cover yet
THREADS = 16

today = dt.date.today()
DAY0 = dt.date(2020, 1, 1)
def day_num(d=None): return ((d or today) - DAY0).days


# ---------------------------------------------------------------- TMDB client
class TMDB:
    def __init__(self, key):
        self.s = requests.Session()
        if key.startswith("eyJ"):        # v4 read access token
            self.s.headers["Authorization"] = "Bearer " + key
            self.params = {}
        else:
            self.params = {"api_key": key}
        self.calls = 0

    def get(self, path, **params):
        p = dict(self.params); p.update(params)
        for attempt in range(6):
            try:
                r = self.s.get(API + path, params=p, timeout=30)
            except requests.RequestException:
                time.sleep(2 ** attempt); continue
            self.calls += 1
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 2)) + 0.5); continue
            if r.status_code == 404:
                return None
            if r.status_code >= 500:
                time.sleep(2 ** attempt); continue
            r.raise_for_status()
            return r.json()
        return None


# ---------------------------------------------------------------- MovieLens taste vectors
def build_crowd_vectors():
    """Returns {tmdb_id: (int8 vector, scale-free), ...}, bias dict, and scale."""
    import pandas as pd
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import svds

    blob = None
    for url in ML_URLS:
        try:
            print("downloading", url, flush=True)
            r = requests.get(url, timeout=600); r.raise_for_status(); blob = r.content; break
        except Exception as e:
            print("  failed:", e, flush=True)
    if blob is None:
        raise RuntimeError("could not download MovieLens")
    z = zipfile.ZipFile(io.BytesIO(blob))
    base = z.namelist()[0].split("/")[0]
    links = pd.read_csv(z.open(f"{base}/links.csv"), usecols=["movieId", "tmdbId"]).dropna()
    links["tmdbId"] = links["tmdbId"].astype(int)
    rt = pd.read_csv(z.open(f"{base}/ratings.csv"), usecols=["userId", "movieId", "rating"],
                     dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32})
    del blob, z
    print("ratings:", len(rt), flush=True)

    # keep movies and users with enough data (two passes)
    for _ in range(2):
        ic = rt["movieId"].value_counts(); rt = rt[rt["movieId"].isin(ic[ic >= ML_MIN_ITEM].index)]
        uc = rt["userId"].value_counts(); rt = rt[rt["userId"].isin(uc[uc >= ML_MIN_USER].index)]
    print("after filter:", len(rt), "ratings,", rt["movieId"].nunique(), "movies,",
          rt["userId"].nunique(), "users", flush=True)

    # remove user generosity and movie quality, keep only taste (who likes what)
    rt = rt.copy()
    rt["r"] = rt["rating"] - rt.groupby("userId")["rating"].transform("mean")
    g = rt.groupby("movieId")["r"]
    bias = (g.sum() / (g.count() + 25.0))                   # shrunk movie quality vs. raters' own averages
    rt["r"] = rt["r"] - rt["movieId"].map(bias).astype(np.float32)

    u_codes, u_idx = np.unique(rt["userId"].values, return_inverse=True)
    m_codes, m_idx = np.unique(rt["movieId"].values, return_inverse=True)
    M = csr_matrix((rt["r"].values.astype(np.float32), (u_idx, m_idx)),
                   shape=(len(u_codes), len(m_codes)))
    del rt
    print("svd...", flush=True)
    _, s, vt = svds(M, k=K)
    order = np.argsort(-s); s = s[order]; vt = vt[order]
    Q = (vt.T * np.sqrt(s / s[0]))                          # movies x K
    # sign-fix so the build is reproducible
    for j in range(K):
        if Q[np.argmax(np.abs(Q[:, j])), j] < 0: Q[:, j] *= -1
    norms = np.linalg.norm(Q, axis=1)
    Q = Q / np.median(norms) * 1.2                           # typical movie ~ a couple of binary features
    scale = float(np.percentile(np.abs(Q), 99.9)) / 127.0
    Qi = np.clip(np.round(Q / scale), -127, 127).astype(np.int8)

    ml2tmdb = dict(zip(links["movieId"], links["tmdbId"]))
    out, biases = {}, {}
    for i, mid in enumerate(m_codes):
        t = ml2tmdb.get(int(mid))
        if t is None: continue
        out[int(t)] = base64.b64encode(Qi[i].tobytes()).decode()
        biases[int(t)] = round(float(bias.get(mid, 0.0)), 2)
    print("taste vectors:", len(out), flush=True)
    return out, biases, scale


# ---------------------------------------------------------------- TMDB fetch
def movie_record(d):
    crew = (d.get("credits") or {}).get("crew") or []
    cast = (d.get("credits") or {}).get("cast") or []
    directors = [c["name"] for c in crew if c.get("job") == "Director"][:2]
    writers = [c["name"] for c in crew if c.get("job") in ("Screenplay", "Writer")][:1]
    kws = [k["name"] for k in ((d.get("keywords") or {}).get("keywords") or [])][:12]
    rd = d.get("release_date") or ""
    return {
        "id": "m%d" % d["id"], "t": d.get("title") or d.get("original_title") or "",
        "ot": d.get("original_title") if d.get("original_title") != d.get("title") else None,
        "y": int(rd[:4]) if rd[:4].isdigit() else None,
        "g": [x["id"] for x in d.get("genres") or []],
        "kw": kws, "dir": directors + writers, "cast": [c["name"] for c in cast[:4]],
        "lang": d.get("original_language"), "rt": d.get("runtime") or None,
        "va": round(d.get("vote_average") or 0, 1), "vc": d.get("vote_count") or 0,
        "pop": round(d.get("popularity") or 0, 1), "p": d.get("poster_path"),
        "ov": (d.get("overview") or "")[:240],
        "prov": providers_of(d), "f": day_num(),
    }

def tv_record(d):
    cast = (d.get("credits") or {}).get("cast") or []
    creators = [c["name"] for c in d.get("created_by") or []][:2]
    kws = [k["name"] for k in ((d.get("keywords") or {}).get("results") or [])][:12]
    fa = d.get("first_air_date") or ""
    return {
        "id": "t%d" % d["id"], "t": d.get("name") or d.get("original_name") or "",
        "ot": d.get("original_name") if d.get("original_name") != d.get("name") else None,
        "y": int(fa[:4]) if fa[:4].isdigit() else None,
        "g": [x["id"] for x in d.get("genres") or []],
        "kw": kws, "dir": creators, "cast": [c["name"] for c in cast[:4]],
        "lang": d.get("original_language"), "rt": None,
        "ns": d.get("number_of_seasons"), "st": d.get("status"),
        "va": round(d.get("vote_average") or 0, 1), "vc": d.get("vote_count") or 0,
        "pop": round(d.get("popularity") or 0, 1), "p": d.get("poster_path"),
        "ov": (d.get("overview") or "")[:240],
        "prov": providers_of(d), "f": day_num(),
    }

def providers_of(d):
    wp = ((d.get("watch/providers") or {}).get("results") or {}).get(REGION) or {}
    return sorted({p["provider_id"] for p in (wp.get("flatrate") or []) + (wp.get("free") or []) + (wp.get("ads") or [])})

PROVIDER_NAMES = {}

def fetch_detail(api, kind, tid):
    d = api.get(f"/{kind}/{tid}", append_to_response="keywords,credits,watch/providers", language="en-US")
    if not d: return None
    wp = ((d.get("watch/providers") or {}).get("results") or {}).get(REGION) or {}
    for p in (wp.get("flatrate") or []) + (wp.get("free") or []) + (wp.get("ads") or []):
        PROVIDER_NAMES[p["provider_id"]] = p["provider_name"]
    return movie_record(d) if kind == "movie" else tv_record(d)

def fetch_providers(api, kind, tid):
    d = api.get(f"/{kind}/{tid}/watch/providers")
    if d is None: return None
    wp = (d.get("results") or {}).get(REGION) or {}
    allp = (wp.get("flatrate") or []) + (wp.get("free") or []) + (wp.get("ads") or [])
    for p in allp: PROVIDER_NAMES[p["provider_id"]] = p["provider_name"]
    return sorted({p["provider_id"] for p in allp})

def discover(api, kind, pages, **params):
    ids = []
    for page in range(1, pages + 1):
        j = api.get(f"/discover/{kind}", page=page, **params)
        if not j or not j.get("results"): break
        ids += [r["id"] for r in j["results"]]
        if page >= (j.get("total_pages") or 0): break
    return ids

def run_pool(fn, jobs, label):
    out, done = {}, 0
    with ThreadPoolExecutor(THREADS) as ex:
        futs = {ex.submit(fn, *j): j for j in jobs}
        for f in as_completed(futs):
            j = futs[f]; done += 1
            try: out[j] = f.result()
            except Exception as e: print("  error", j, e, flush=True); out[j] = None
            if done % 1000 == 0: print(f"  {label}: {done}/{len(jobs)}", flush=True)
    return out


# ---------------------------------------------------------------- compaction
def compact(records, crowd, biases, scale, genre_names):
    kw_count = Counter(k for r in records for k in set(r["kw"]))
    pp_count = Counter(p for r in records for p in set(r["dir"] + r["cast"]))
    kw_list = sorted(k for k, c in kw_count.items() if c >= MIN_FEATURE_ITEMS)
    pp_list = sorted(p for p, c in pp_count.items() if c >= MIN_FEATURE_ITEMS)
    kw_i = {k: i for i, k in enumerate(kw_list)}
    pp_i = {p: i for i, p in enumerate(pp_list)}
    items = []
    for r in sorted(records, key=lambda r: -r["vc"]):
        tid = int(r["id"][1:])
        emb = crowd.get(tid) if r["id"][0] == "m" else None
        items.append([
            r["id"], r["t"], r["y"], r["g"],
            [kw_i[k] for k in r["kw"] if k in kw_i],
            [pp_i[p] for p in r["dir"] if p in pp_i],
            [pp_i[p] for p in r["cast"] if p in pp_i],
            r["lang"], r["rt"], r["va"], r["vc"], r["p"],
            emb, biases.get(tid) if r["id"][0] == "m" else None,
            r.get("ot"), r["f"],
        ])
    return {
        "v": 1, "built": today.isoformat(), "region": REGION,
        "fields": ["id", "title", "year", "genres", "keywords", "makers", "cast", "lang",
                   "runtime", "voteAvg", "voteCount", "poster", "taste", "crowdBias",
                   "originalTitle", "fetchedDay"],
        "embScale": scale, "embDim": K,
        "genres": genre_names, "keywords": kw_list, "people": pp_list, "items": items,
        "attribution": "This product uses the TMDB API but is not endorsed or certified by TMDB. "
                       "Taste vectors derived from MovieLens (GroupLens Research, University of Minnesota).",
    }

def expand(cat):
    """Turn a compact catalog back into records so a rebuild can reuse fetched data."""
    kw, pp = cat["keywords"], cat["people"]
    recs = {}
    for it in cat["items"]:
        (iid, t, y, g, k, mk, cs, lang, rt, va, vc, p, emb, mb, ot, f) = it
        recs[iid] = {"id": iid, "t": t, "y": y, "g": g, "kw": [kw[i] for i in k],
                     "dir": [pp[i] for i in mk], "cast": [pp[i] for i in cs], "lang": lang,
                     "rt": rt, "va": va, "vc": vc, "p": p, "ot": ot, "ov": "", "f": f,
                     "_emb": emb, "_mb": mb, "prov": []}
    return recs


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-crowd", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    key = os.environ.get("TMDB_API_KEY", "").strip()
    if not key: sys.exit("TMDB_API_KEY is not set (add it as a repo secret).")
    api = TMDB(key)
    os.makedirs(DATA, exist_ok=True)

    old = {}
    old_cat = None
    if os.path.exists(CAT_PATH):
        old_cat = json.load(open(CAT_PATH)); old = expand(old_cat)
        try:
            for iid, v in json.load(open(OV_PATH)).items():
                if iid in old: old[iid]["ov"] = v
        except Exception: pass
        try:
            for iid, v in json.load(open(PROV_PATH))["items"].items():
                if iid in old: old[iid]["prov"] = v
        except Exception: pass
        print("existing catalog:", len(old), "titles", flush=True)

    # taste vectors: reuse unless missing or forced
    if a.rebuild_crowd or not old_cat or not any(r["_emb"] for r in old.values()):
        crowd, biases, scale = build_crowd_vectors()
    else:
        crowd = {int(r["id"][1:]): r["_emb"] for r in old.values() if r["_emb"]}
        biases = {int(r["id"][1:]): r["_mb"] for r in old.values() if r["_mb"] is not None}
        scale = old_cat["embScale"]
        print("reusing", len(crowd), "taste vectors", flush=True)

    # which titles: every MovieLens movie with a taste vector, recent movies, most-voted TV
    movie_ids = set(crowd.keys())
    since = (today - dt.timedelta(days=4 * 365)).isoformat()
    movie_ids |= set(discover(api, "movie", NEW_PAGES, sort_by="popularity.desc",
                              **{"primary_release_date.gte": since, "vote_count.gte": 40}))
    movie_ids |= set(discover(api, "movie", 25, sort_by="vote_count.desc"))
    tv_ids = set(discover(api, "tv", TV_PAGES, sort_by="vote_count.desc", **{"vote_count.gte": 80}))
    jobs = [("movie", i) for i in sorted(movie_ids)] + [("tv", i) for i in sorted(tv_ids)]
    if a.limit: jobs = jobs[:a.limit]
    print("titles:", len(jobs), flush=True)

    def oid(kind, tid): return ("m" if kind == "movie" else "t") + str(tid)
    stale = [j for j in jobs if oid(*j) not in old or day_num() - (old[oid(*j)]["f"] or 0) > REFRESH_DAYS]
    stale_set = set(stale)
    fresh = [j for j in jobs if j not in stale_set]
    print("fetch details:", len(stale), " refresh providers only:", len(fresh), flush=True)

    det = run_pool(lambda k, i: fetch_detail(api, k, i), stale, "details")
    prv = run_pool(lambda k, i: fetch_providers(api, k, i), fresh, "providers")

    records, prov_map = [], {}
    for j in jobs:
        r = det.get(j) if j in det else None
        if r is None:
            r = old.get(oid(*j))
            if r is None: continue
            if j in prv and prv[j] is not None: r["prov"] = prv[j]
        records.append(r)
        if r.get("prov"): prov_map[r["id"]] = r["prov"]

    gm = api.get("/genre/movie/list") or {"genres": []}
    gt = api.get("/genre/tv/list") or {"genres": []}
    genre_names = {str(g["id"]): g["name"] for g in gm["genres"] + gt["genres"]}

    cat = compact(records, crowd, biases, scale, genre_names)
    tmp = CAT_PATH + ".tmp"
    with open(tmp, "w") as fh: json.dump(cat, fh, separators=(",", ":"), ensure_ascii=False)
    os.replace(tmp, CAT_PATH)

    ovs = {r["id"]: r["ov"] for r in records if r.get("ov")}
    with open(OV_PATH + ".tmp", "w") as fh: json.dump(ovs, fh, separators=(",", ":"), ensure_ascii=False)
    os.replace(OV_PATH + ".tmp", OV_PATH)

    used = sorted({p for v in prov_map.values() for p in v})
    old_names = {}
    if os.path.exists(PROV_PATH):
        try: old_names = json.load(open(PROV_PATH)).get("names", {})
        except Exception: pass
    names = {str(p): PROVIDER_NAMES.get(p) or old_names.get(str(p)) or f"Service {p}" for p in used}
    prov = {"v": 1, "built": today.isoformat(), "region": REGION, "names": names, "items": prov_map}
    with open(PROV_PATH + ".tmp", "w") as fh: json.dump(prov, fh, separators=(",", ":"), ensure_ascii=False)
    os.replace(PROV_PATH + ".tmp", PROV_PATH)

    mb = os.path.getsize(CAT_PATH) / 1e6
    print(f"done: {len(cat['items'])} titles, {len(cat['keywords'])} keywords, {len(cat['people'])} people, "
          f"{mb:.1f} MB catalog, {api.calls} TMDB calls", flush=True)

if __name__ == "__main__":
    main()
