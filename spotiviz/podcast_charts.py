import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .style import GREEN, finish


def chart_top_shows(pods, out, n=15):
    s = pods.groupby("show")["minutes"].sum().sort_values(ascending=False).head(n)[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(s.index, s.values / 60, color=GREEN)
    for i, v in enumerate(s.values):
        ax.text(v / 60 + 0.05, i, f"{v/60:.1f}h", va="center", fontsize=8)
    ax.set_title(f"Top {n} podcast shows by listen time")
    ax.set_xlabel("hours")
    finish(fig, out)


def chart_podcast_over_time(pods, out):
    if pods.empty:
        return
    m = pods.groupby("month")["minutes"].sum()
    fig, ax = plt.subplots(figsize=(12, 4))
    x = [p.to_timestamp() for p in m.index]
    ax.plot(x, m.values / 60, marker="o", ms=3.5, color=GREEN, lw=1.6)
    ax.fill_between(x, m.values / 60, alpha=0.2, color=GREEN)
    ax.set_ylabel("hours / month")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    finish(fig, out, "Podcast listening over time")


def chart_top_episodes(pods, out, n=15):
    if pods.empty:
        return
    e = (
        pods.groupby(["episode", "show"])["minutes"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
    )
    labels = [f"{ep[:58]} | {sh}" for ep, sh in e.index]
    s = pd.Series(e.values / 60, index=labels)[::-1]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(s.index, s.values, color="#7c4dff", alpha=0.85)
    ax.tick_params(labelsize=8)
    ax.set_title(f"{n} most-listened episodes")
    ax.set_xlabel("hours")
    finish(fig, out)


def chart_music_vs_podcasts(music, pods, out):
    if pods.empty:
        return
    years = sorted(set(music["year"].unique()) | set(pods["year"].unique()))
    m = music.groupby("year")["minutes"].sum().reindex(years, fill_value=0)
    p = pods.groupby("year")["minutes"].sum().reindex(years, fill_value=0)
    share = (p / (p + m) * 100).fillna(0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].bar([str(y) for y in years], m.values / 60, label="music", color=GREEN)
    axes[0].bar([str(y) for y in years], p.values / 60, bottom=m.values / 60, label="podcasts", color="#7c4dff")
    axes[0].set_title("Music vs podcasts hours per year")
    axes[0].legend(frameon=False)
    axes[1].bar([str(y) for y in years], share.values, color="#7c4dff")
    axes[1].set_title("Podcast share of total listening")
    axes[1].set_ylabel("%")
    axes[1].set_ylim(0, 100)
    finish(fig, out)


def render_all(music, pods, out_dir):
    from pathlib import Path

    out_dir = Path(out_dir)
    made = []
    for name, fn in [
        ("21_top_podcast_shows", lambda: chart_top_shows(pods, out_dir / "21_top_podcast_shows.png")),
        ("22_podcasts_over_time", lambda: chart_podcast_over_time(pods, out_dir / "22_podcasts_over_time.png")),
        ("23_top_episodes", lambda: chart_top_episodes(pods, out_dir / "23_top_episodes.png")),
        ("24_music_vs_podcasts", lambda: chart_music_vs_podcasts(music, pods, out_dir / "24_music_vs_podcasts.png")),
    ]:
        try:
            fn()
            made.append(name)
            print(f"  [ok] {name}")
        except Exception as e:
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
    return made
