import json
from pathlib import Path

ROOT = Path(__file__).parent.parent / "taxonomy"


def select_artists(labels_df, n=30, top=25):
    splits = labels_df[labels_df["splits_moods"]].nlargest(n, "minutes").index.tolist()
    big = labels_df.nlargest(top, "minutes").index.tolist()
    seen = []
    for a in splits + big:
        if a not in seen:
            seen.append(a)
    return seen[:n]


def track_packet(music, artist, k=30):
    sub = music[music["artist"] == artist]
    top = sub.groupby("track").agg(minutes=("minutes", "sum"), plays=("minutes", "size"))
    top = top.nlargest(k, "minutes")
    skipped = sub.dropna(subset=["skipped"]).groupby("track")["skipped"].mean()
    out = []
    for t, r in top.iterrows():
        out.append({
            "track": t,
            "hours": round(r["minutes"] / 60, 2),
            "plays": int(r["plays"]),
            "skip_rate": round(float(skipped.get(t, 0)), 2),
        })
    return {"artist": artist, "tracks": out}


def build(music, labels_df, n=30):
    tdir = ROOT / "track_batches"
    rdir = ROOT / "track_results"
    tdir.mkdir(parents=True, exist_ok=True)
    rdir.mkdir(exist_ok=True)
    artists = select_artists(labels_df, n=n)
    manifest = []
    for i, a in enumerate(artists):
        pk = track_packet(music, a)
        f = tdir / f"tracks_{i:02d}.json"
        f.write_text(json.dumps(pk, ensure_ascii=False, indent=1), encoding="utf-8")
        manifest.append({"id": i, "artist": a, "file": f.name, "n_tracks": len(pk["tracks"])})
    (ROOT / "track_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(manifest)} track batches, {sum(m['n_tracks'] for m in manifest)} tracks total")
    return manifest
