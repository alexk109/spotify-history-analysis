import json
from pathlib import Path

import pandas as pd

TZ = "America/Phoenix"
MIN_PLAY_MS = 30_000
SESSION_GAP_S = 600


def load_history(data_dir):
    rows = []
    for f in sorted(Path(data_dir).glob("Streaming_History_*.json")):
        with open(f, encoding="utf-8") as fh:
            rows.extend(json.load(fh))
    df = pd.DataFrame(rows).drop_duplicates()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["local_ts"] = df["ts"].dt.tz_convert(TZ)
    df["ms_played"] = df["ms_played"].astype("int64")
    df.drop(columns=["ip_addr"], inplace=True, errors="ignore")
    df["is_podcast"] = df["spotify_episode_uri"].notna()
    df["track"] = df["master_metadata_track_name"].fillna("")
    df["artist"] = df["master_metadata_album_artist_name"].fillna("")
    df["album"] = df["master_metadata_album_album_name"].fillna("")
    df["show"] = df["episode_show_name"].fillna("")
    df["episode"] = df["episode_name"].fillna("")
    df = df[df["ms_played"] >= MIN_PLAY_MS].copy()
    df["minutes"] = df["ms_played"] / 60_000.0
    lt = df["local_ts"].dt.tz_localize(None)
    df["date"] = lt.dt.date
    df["year"] = lt.dt.year
    df["month"] = lt.dt.to_period("M")
    df["hour"] = lt.dt.hour
    df["dow"] = lt.dt.dayofweek
    df["week"] = lt.dt.to_period("W")
    return df.sort_values("ts").reset_index(drop=True)


def split_music_podcasts(df):
    music = df[~df["is_podcast"] & (df["artist"] != "")].copy()
    pods = df[df["is_podcast"] & (df["show"] != "")].copy()
    return music, pods


def sessionize(df, gap_s=SESSION_GAP_S):
    d = df.sort_values("ts")
    gap = d["ts"].diff().dt.total_seconds()
    sid = ((gap > gap_s) | gap.isna()).cumsum()
    g = d.groupby(sid)
    dur = g.apply(
        lambda x: (x["ts"].iloc[-1] - x["ts"].iloc[0]).total_seconds() / 60
        + x["minutes"].iloc[-1],
        include_groups=False,
    )
    out = g.agg(
        start=("ts", "first"),
        n_plays=("track", "size"),
        minutes=("minutes", "sum"),
    )
    out["duration_min"] = dur
    return out


def daily_series(df):
    s = df.groupby("date")["minutes"].sum()
    full = pd.date_range(min(s.index), max(s.index), freq="D").date
    return s.reindex(full, fill_value=0.0)


def first_play_flags(df, col):
    seen = set()
    flags = []
    for v in df[col]:
        flags.append(v not in seen)
        seen.add(v)
    return pd.Series(flags, index=df.index)


def platform_family(p):
    p = str(p).lower()
    if "ios" in p or "iphone" in p or "ipad" in p:
        return "iOS"
    if "android" in p:
        return "Android"
    if "windows" in p:
        return "Desktop (Windows)"
    if "os x" in p or "mac" in p or "darwin" in p:
        return "Desktop (macOS)"
    if "linux" in p:
        return "Desktop (Linux)"
    if "web" in p or "player" in p:
        return "Web Player"
    if "tv" in p:
        return "TV"
    if "cast" in p:
        return "Cast"
    if "wear" in p or "watch" in p:
        return "Wearable"
    if "car" in p or "android_auto" in p:
        return "Car"
    return "Other"


def longest_streak(dates):
    ds = sorted(set(dates))
    best = cur = 0
    prev = None
    for d in ds:
        if prev is not None and (d - prev).days == 1:
            cur += 1
        else:
            cur = 1
        best = max(best, cur)
        prev = d
    return best


def fmt_hours(minutes):
    h = minutes / 60
    if h >= 1000:
        return f"{h:,.0f} h"
    return f"{h:,.1f} h"
