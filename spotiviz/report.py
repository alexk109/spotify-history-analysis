import pandas as pd

from pathlib import Path

from .data import daily_series, first_play_flags, longest_streak, fmt_hours


def _md_table(rows, headers):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def build_report(music, pods, out_path, chart_names):
    d = music.sort_values("ts").copy()
    total_music = music["minutes"].sum()
    total_pods = pods["minutes"].sum() if not pods.empty else 0
    daily = daily_series(music)
    biggest = daily.idxmax()
    streak = longest_streak(music["date"])
    zero_days = (daily == 0).sum()
    d["replay"] = ~first_play_flags(d, "spotify_track_uri")
    replay_share = d.loc[d["replay"], "minutes"].sum() / total_music * 100
    have = music[music["skipped"].notna()]
    skip_rate = have["skipped"].mean() * 100 if not have.empty else float("nan")

    lines = []
    a = lines.append
    a("# Your Spotify Listening — The Full Report\n")
    a(f"*Data span:* {music['ts'].min():%b %d, %Y} – {music['ts'].max():%b %d, %Y} · "
      f"all times converted to **America/Phoenix** · plays counted when ≥ 30s · IPs excluded\n")

    a("## Headline numbers\n")
    a(_md_table([
        ("Total listening", f"{(total_music+total_pods)/60:,.0f} hours"),
        ("Music", f"{total_music/60:,.0f} h ({len(music):,} plays)"),
        ("Podcasts", f"{total_pods/60:,.0f} h ({len(pods):,} plays)" if not pods.empty else "—"),
        ("Unique artists / tracks / albums",
         f"{music['artist'].nunique():,} / {music['spotify_track_uri'].nunique():,} / {music['album'].nunique():,}"),
        ("Active days", f"{music['date'].nunique():,} of {(daily.index.max()-daily.index.min()).days+1:,}")
        ,
        ("Zero-listening days", f"{zero_days:,}"),
        ("Longest daily streak", f"{streak} days"),
        ("Biggest single day", f"{daily.max()/60:.1f} h on {biggest:%b %d, %Y}"),
        ("Avg per active day", f"{music['minutes'].sum()/music['date'].nunique()/60:.1f} h"),
        ("Replay share", f"{replay_share:.0f}% of music minutes were tracks you'd already played"),
        ("Skip rate", f"{skip_rate:.0f}%"),
    ], ["Metric", "Value"]))

    a("\n## All-time top 10 artists\n")
    at = music.groupby("artist")["minutes"].agg(["sum", "size"]).sort_values("sum", ascending=False).head(10)
    a(_md_table([(i, f"{r['sum']/60:.1f} h", f"{r['size']:,}") for i, r in at.iterrows()],
                ["Artist", "Time", "Plays"]))

    a("\n## All-time top 10 tracks\n")
    tt = (music.groupby(["track", "artist"])["minutes"].agg(["sum", "size"])
          .sort_values("sum", ascending=False).head(10))
    a(_md_table([(f"**{t}** — {ar}", f"{r['sum']/60:.1f} h", f"{r['size']:,}")
                 for (t, ar), r in tt.iterrows()],
                ["Track", "Time", "Plays"]))

    a("\n## Year by year\n")
    rows = []
    for y, sub in music.groupby("year"):
        top_a = sub.groupby("artist")["minutes"].sum()
        top_t = sub.groupby("track")["minutes"].sum()
        rows.append((y, f"{sub['minutes'].sum()/60:,.0f} h", len(sub),
                     sub["artist"].nunique(), top_a.idxmax(), top_t.idxmax()))
    a(_md_table(rows, ["Year", "Hours", "Plays", "Unique artists", "#1 Artist", "#1 Track"]))

    if not pods.empty:
        a("\n## Podcast corner\n")
        ps = pods.groupby("show")["minutes"].agg(["sum", "size"]).sort_values("sum", ascending=False).head(8)
        a(_md_table([(i, f"{r['sum']/60:.1f} h", f"{r['size']:,}") for i, r in ps.iterrows()],
                    ["Show", "Time", "Episodes played"]))
        share = total_pods / (total_pods + total_music) * 100
        a(f"\nPodcasts are **{share:.0f}%** of everything you've played.\n")

    gp = Path(__file__).parent.parent / "output" / "taxonomy" / "artists_taxonomy.csv"
    if gp.exists():
        tax = pd.read_csv(gp)
        tax = tax[tax["genre"].notna() & (tax["genre"] != "unknown")]
        if not tax.empty:
            tax["tag"] = tax["genre"].str.split(", ")
            tax = tax.explode("tag")
            gg = (
                tax.groupby("tag")
                .apply(lambda g: pd.Series({"hours": g["minutes"].sum() / 60, "artists": g["artist"].nunique()}), include_groups=False)
                .sort_values("hours", ascending=False)
                .head(15)
            )
            a("\n## Top genres\n")
            a(_md_table([(i, f"{r['hours']:.1f} h", int(r["artists"])) for i, r in gg.iterrows()],
                        ["Genre", "Hours", "Artists"]))

    a("## Charts generated\n")
    a(", ".join(f"`{c}.png`" for c in sorted(chart_names)))
    a("\nOpen `dashboard.html` for the interactive version.")

    out_path.write_text("\n".join(lines), encoding="utf-8")
