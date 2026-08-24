import json
from pathlib import Path

import pandas as pd

from .data import first_play_flags

ROOT = Path(__file__).parent.parent / "taxonomy"
MOODS = {"sad", "excited", "normal", "unknown"}
SOCIALS = {"alone", "social", "both", "unknown"}
REQ = {"artist", "mood", "social", "confidence", "splits_moods", "genre", "reasoning"}

MANUAL_PATCH = [
    {"artist": "The Buggles", "mood": "normal", "social": "alone", "confidence": 0.6,
     "splits_moods": False, "genre": "new wave, synth-pop", "reasoning": "quirky new-wave one-hit; casual alone listening"},
    {"artist": "Lil Yachty", "mood": "normal", "social": "both", "confidence": 0.5,
     "splits_moods": True, "genre": "trap, mumble rap", "reasoning": "melodic trap spans chill and hype contexts"},
    {"artist": "GoodBooks", "mood": "sad", "social": "social", "confidence": 0.4,
     "splits_moods": False, "genre": "indie rock", "reasoning": "britpop-adjacent indie rock, mainstream-lean sad/social"},
    {"artist": "Siam Jem", "mood": "unknown", "social": "unknown", "confidence": 0.2,
     "splits_moods": False, "genre": "unknown", "reasoning": "not recognizable; left unknown per rules"},
]


def load_results():
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    rows, problems = [], []
    for b in manifest:
        rf = ROOT / "results" / f"batch_{b['id']:02d}.json"
        if not rf.exists():
            problems.append(f"batch_{b['id']:02d}: MISSING result file")
            continue
        try:
            data = json.loads(rf.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            problems.append(f"batch_{b['id']:02d}: unparseable JSON ({e})")
            continue
        got = {r.get("artist") for r in data}
        missing = [a for a in b["artists"] if a not in got]
        if missing:
            problems.append(f"batch_{b['id']:02d}: missing {len(missing)}: {missing[:5]}")
        for r in data:
            a = r.get("artist", "?")
            mood = str(r.get("mood", "unknown")).lower().strip()
            soc = str(r.get("social", "unknown")).lower().strip()
            if mood not in MOODS:
                problems.append(f"batch_{b['id']:02d} '{a}': bad mood '{mood}'")
                mood = "unknown"
            if soc not in SOCIALS:
                problems.append(f"batch_{b['id']:02d} '{a}': bad social '{soc}'")
                soc = "unknown"
            try:
                conf = min(max(float(r.get("confidence", 0.3)), 0.0), 1.0)
            except (TypeError, ValueError):
                conf = 0.3
            rows.append({
                "artist": a,
                "mood": mood,
                "social": soc,
                "confidence": conf,
                "splits_moods": bool(r.get("splits_moods", False)),
                "genre": str(r.get("genre", "unknown")).lower().strip() or "unknown",
                "reasoning": str(r.get("reasoning", ""))[:200],
            })
    have = {r["artist"] for r in rows}
    for r in MANUAL_PATCH:
        if r["artist"] not in have:
            rows.append(r)
    return pd.DataFrame(rows), problems


def merge_with_minutes(music, labels):
    mins = music.groupby("artist")["minutes"].sum()
    plays = music.groupby("artist").size()
    lab = labels.drop_duplicates("artist").set_index("artist")
    df = pd.DataFrame({"minutes": mins, "plays": plays}).join(lab, how="left")
    n = len(df)
    df["classified"] = df["mood"].notna()
    df["mood"] = df["mood"].fillna("unclassified")
    df["social"] = df["social"].fillna("unclassified")
    df["confidence"] = df["confidence"].fillna(0.0)
    df["splits_moods"] = df["splits_moods"].fillna(False)
    df["genre"] = df["genre"].fillna("unknown")
    df["reasoning"] = df["reasoning"].fillna("")
    df["cell"] = df["mood"] + "/" + df["social"]
    return df, n


def propagate_tail(music, df):
    tail = df[~df["classified"]]
    if tail.empty:
        return df, 0
    gap = music.sort_values("ts")["ts"].diff().dt.total_seconds()
    m = music.sort_values("ts").copy()
    m["sid"] = ((gap > 600) | gap.isna()).cumsum()
    seed = df[df["classified"] & (df["mood"] != "unknown") & (df["confidence"] >= 0.5)]
    seed_map = seed["cell"].to_dict()
    votes = {}
    for sid, g in m.groupby("sid"):
        arts = set(g["artist"])
        hit = arts & set(seed_map)
        if not hit:
            continue
        for a in arts - set(seed_map):
            v = votes.setdefault(a, {})
            for s in hit:
                v[seed_map[s]] = v.get(seed_map[s], 0) + 1
    n_prop = 0
    for a, v in votes.items():
        total = sum(v.values())
        if total < 3:
            continue
        best, cnt = max(v.items(), key=lambda kv: kv[1])
        share = cnt / total
        if share >= 0.6 and best in ("sad/alone", "excited/alone", "normal/alone"):
            df.loc[a, ["mood", "social", "cell", "confidence", "reasoning"]] = [
                best.split("/")[0], best.split("/")[1], best,
                round(0.3 * share, 2), f"inferred from session-mates ({share:.0%} of {total} shared sessions)",
            ]
            n_prop += 1
    df["classified"] = df["mood"] != "unclassified"
    return df, n_prop


def load_track_results():
    manifest = json.loads((ROOT / "track_manifest.json").read_text(encoding="utf-8"))
    rows = []
    for b in manifest:
        f = ROOT / "track_results" / f"tracks_{b['id']:02d}.json"
        if not f.exists():
            print(f"  track batch {b['id']} missing")
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        for r in data:
            mood = str(r.get("mood", "unknown")).lower().strip()
            soc = str(r.get("social", "unknown")).lower().strip()
            rows.append({
                "artist": b["artist"],
                "track": r.get("track", "?"),
                "mood": mood if mood in MOODS else "unknown",
                "social": soc if soc in SOCIALS else "unknown",
                "confidence": min(max(float(r.get("confidence", 0.3)), 0.0), 1.0),
                "genre": str(r.get("genre", "unknown")).lower().strip() or "unknown",
                "reasoning": str(r.get("reasoning", ""))[:150],
            })
    return pd.DataFrame(rows)


def save_tracks(music, outdir):
    tr = load_track_results()
    mins = music.groupby(["artist", "track"])["minutes"].sum()
    plays = music.groupby(["artist", "track"]).size()
    tr["minutes"] = tr.set_index(["artist", "track"]).index.map(mins)
    tr["plays"] = tr.set_index(["artist", "track"]).index.map(plays)
    tr["cell"] = tr["mood"] + "/" + tr["social"]
    tr.sort_values("minutes", ascending=False).to_csv(outdir / "tracks_taxonomy.csv", index=False)
    print(f"track labels: {len(tr)} tracks, {tr['artist'].nunique()} artists")
    return tr


def run(music):
    labels, problems = load_results()
    df, n = merge_with_minutes(music, labels)
    n_prop = 0
    if problems:
        print(f"{len(problems)} problems:")
        for p in problems[:15]:
            print("  -", p)
    df, n_prop = propagate_tail(music, df)
    outdir = Path("output/taxonomy")
    outdir.mkdir(parents=True, exist_ok=True)
    out = df.reset_index().rename(columns={"index": "artist"})
    cols = ["artist", "mood", "social", "cell", "confidence", "splits_moods", "genre", "minutes", "plays", "classified", "reasoning"]
    out[cols].to_csv(outdir / "artists_taxonomy.csv", index=False)
    out[cols].to_json(outdir / "artists_taxonomy.json", orient="records", indent=1, force_ascii=False)

    cov = out[out["classified"]]["minutes"].sum() / out["minutes"].sum() * 100
    print(f"artists: {len(out)} | classified: {out['classified'].sum()} (+{n_prop} propagated) | listening-time coverage: {cov:.0f}%")
    print("\ncell distribution (by listening hours):")
    cell_h = out.groupby("cell").apply(lambda g: g["minutes"].sum() / 60, include_groups=False).sort_values(ascending=False)
    for c, h in cell_h.items():
        print(f"  {c:22s} {h:8.1f} h")
    print("\ntop splits_moods:", out[out["splits_moods"]].nlargest(8, "minutes")["artist"].tolist())
    print("\ngenre top 15:", out.loc[out["genre"] != "unknown", "genre"].str.split(", ").explode().value_counts().head(15).to_dict())
    save_tracks(music, outdir)
    return out
