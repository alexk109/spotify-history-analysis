import json
from pathlib import Path

from .data import sessionize

BATCH_SIZE = 25
ROOT = Path(__file__).parent.parent / "taxonomy"

CLOCK_BUCKETS = [(0, 6, "night"), (6, 10, "morning"), (10, 17, "workday"), (17, 24, "evening")]


def clock_summary(sub):
    total = sub["minutes"].sum()
    parts = []
    for lo, hi, name in CLOCK_BUCKETS:
        share = sub.loc[sub["hour"].between(lo, hi - 1), "minutes"].sum() / total
        if share >= 0.35:
            parts.append(f"{name}-heavy({share:.0%})")
    wk = sub.loc[sub["dow"] >= 5, "minutes"].sum() / total
    parts.append(f"weekend {wk:.0%}")
    return ", ".join(parts)


def packet(sub, name):
    top_tracks = sub.groupby("track")["minutes"].sum().nlargest(10).index.tolist()
    top_albums = sub.groupby("album")["minutes"].sum().nlargest(3).index.tolist()
    skipped = sub["skipped"]
    p = {
        "artist": name,
        "hours": round(sub["minutes"].sum() / 60, 1),
        "plays": int(len(sub)),
        "top_tracks_you_played": top_tracks,
        "top_albums": top_albums,
        "listening_clock": clock_summary(sub),
        "skip_rate": round(float(skipped.mean()), 2) if skipped.notna().any() else None,
        "first_year": int(sub["year"].min()),
    }
    if any(ord(c) > 0x2E80 for c in "".join(top_tracks) + name):
        p["note"] = "non-Latin script: translate names before judging"
    return p


def build_batches(music):
    (ROOT / "batches").mkdir(parents=True, exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    totals = music.groupby("artist")["minutes"].sum().sort_values(ascending=False)
    cum = totals.cumsum() / totals.sum()
    core = totals.index[: (cum <= 0.95).sum() + 1]
    packets = [packet(music[music["artist"] == n], n) for n in core]
    batches = [packets[i : i + BATCH_SIZE] for i in range(0, len(packets), BATCH_SIZE)]
    manifest = []
    for i, b in enumerate(batches):
        bp = ROOT / "batches" / f"batch_{i:02d}.json"
        bp.write_text(json.dumps(b, ensure_ascii=False, indent=1), encoding="utf-8")
        manifest.append({"id": i, "file": bp.name, "artists": [p["artist"] for p in b]})
    (ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest, set(core)


def coplay_for_tail(music, core):
    sess = sessionize(music)
    sess_ids = sess.index
    music_with_sess = music.sort_values("ts").copy()
    gap = music_with_sess["ts"].diff().dt.total_seconds()
    music_with_sess["sid"] = ((gap > 600) | gap.isna()).cumsum()
    tail = set(music["artist"].unique()) - core
    mates = {}
    for sid, g in music_with_sess.groupby("sid"):
        arts = set(g["artist"])
        for a in arts & tail:
            for b in arts & core - {a}:
                mates.setdefault(a, {}).setdefault(b, 0)
                mates[a][b] += 1
    out = {}
    for a, m in mates.items():
        top = sorted(m.items(), key=lambda kv: -kv[1])[:4]
        out[a] = [{"artist": b, "shared_sessions": n} for b, n in top]
    (ROOT / "tail_coplay.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return out


def run(music):
    manifest, core = build_batches(music)
    tail = coplay_for_tail(music, core)
    print(f"{len(manifest)} artist batches ({sum(len(b['artists']) for b in manifest)} artists)")
    print(f"tail artists with co-play anchors: {len(tail)}")
    return manifest
