from pathlib import Path

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .data import daily_series, first_play_flags, longest_streak, platform_family, sessionize
from .style import GREEN, PALETTE, finish


def _barh_top(ax, counts, title, color=GREEN, unit="minutes"):
    ax.barh(counts.index[::-1], counts.values[::-1], color=color)
    ax.set_title(title)
    ax.set_xlabel(unit)
    maxv = counts.max()
    for i, v in enumerate(counts.values[::-1]):
        label = f"{v/60:,.0f}h" if unit == "minutes" else f"{v:,.0f}"
        ax.text(v + maxv * 0.01, i, f" {label}", va="center", fontsize=8)


def chart_monthly(df, out):
    m = df.groupby("month")["minutes"].sum()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = [p.to_timestamp() for p in m.index]
    avg = df.groupby("year")["minutes"].sum().mean() / 12
    ax.axhline(avg, color="#999", ls="--", lw=1, label=f"overall monthly avg ({avg/60:.0f} h)")
    ax.plot(x, m.values, color=GREEN, lw=1.6)
    ax.fill_between(x, m.values, alpha=0.25, color=GREEN)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(frameon=False)
    finish(fig, out, "Listening over time - minutes per month")


def chart_heatmap(df, out):
    piv = df.pivot_table(index="dow", columns="hour", values="minutes", aggfunc="sum").fillna(0)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    piv = piv.reindex(range(7), fill_value=0).reindex(columns=range(24), fill_value=0)
    cmap = LinearSegmentedColormap.from_list("spot", ["#ffffff", "#c8f3d8", GREEN, "#0e6b31"])
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.pcolormesh(piv.values, cmap=cmap)
    ax.set_xticks(np.arange(24) + 0.5, [f"{h}" for h in range(24)])
    ax.set_yticks(np.arange(7) + 0.5, days)
    ax.invert_yaxis()
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("minutes")
    finish(fig, out, "When you listen - day of week x hour (America/Phoenix)")


def chart_clock(df, out):
    h = df.groupby("hour")["minutes"].sum().reindex(range(24), fill_value=0)
    theta = np.deg2rad((h.index * 15) )
    width = np.deg2rad(14)
    colors = [GREEN if 9 <= i < 17 else ("#169c46" if 17 <= i < 23 else "#7c4dff") for i in h.index]
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="polar")
    ax.bar(theta, h.values, width=width, color=colors, alpha=0.85)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax.set_xticklabels(["00", "03", "06", "09", "12", "15", "18", "21"])
    ax.set_title("Your listening clock\n(green=work hours, dark green=evening, purple=night)", pad=20)
    ax.grid(alpha=0.4)
    finish(fig, out)


def chart_top_artists(df, out, n=20):
    a = df.groupby("artist")["minutes"].sum().sort_values(ascending=False).head(n)
    fig, ax = plt.subplots(figsize=(9, 7))
    _barh_top(ax, a, f"Top {n} artists of all time")


def chart_top_tracks(df, out, n=20):
    t = df.groupby(["track", "artist"])["minutes"].sum().sort_values(ascending=False).head(n)
    labels = [f"{tr} - {ar}" for tr, ar in t.index]
    s = pd.Series(t.values, index=labels)
    fig, ax = plt.subplots(figsize=(10, 7))
    _barh_top(ax, s, f"Top {n} tracks of all time", color="#169c46")


def chart_top_albums(df, out, n=15):
    al = df.groupby(["album", "artist"])["minutes"].sum().sort_values(ascending=False).head(n)
    labels = [f"{al_} - {ar}" for al_, ar in al.index]
    s = pd.Series(al.values, index=labels)
    fig, ax = plt.subplots(figsize=(10, 6))
    _barh_top(ax, s, f"Top {n} albums of all time", color="#0e6b31")


def chart_yearly_artists(df, out, n=8):
    years = sorted(df["year"].unique())
    ncols = 3
    nrows = int(np.ceil(len(years) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.2 * nrows), constrained_layout=True)
    for ax, y in zip(axes.flat, years):
        sub = df[df["year"] == y]
        top = sub.groupby("artist")["minutes"].sum().sort_values(ascending=False).head(n)[::-1]
        ax.barh(top.index, top.values / 60, color=GREEN)
        ax.set_title(str(y), fontsize=11)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("hours", fontsize=8)
    for ax in axes.flat[len(years):]:
        ax.axis("off")
    finish(fig, out, f"Top {n} artists per year")


def chart_skips(df, out):
    have = df[df["skipped"].notna()]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    per_year = have.groupby("year")["skipped"].mean() * 100
    axes[0].plot(per_year.index, per_year.values, marker="o", color=GREEN)
    axes[0].set_title("Skip rate by year")
    axes[0].set_ylabel("% skipped")
    axes[0].set_ylim(0, 100)
    by_hour = have.groupby("hour")["skipped"].mean().reindex(range(24), fill_value=np.nan) * 100
    axes[1].bar(by_hour.index, by_hour.values, color="#7c4dff")
    axes[1].set_title("Skip rate by hour of day")
    axes[1].set_xlabel("hour")
    art = have.groupby("artist").agg(plays=("skipped", "size"), rate=("skipped", "mean"))
    art = art[art["plays"] >= 50]
    art = art[(art["rate"] > 0) | (art["plays"] >= 100)]
    loyal = art["rate"].nsmallest(8)[::-1]
    impat = art["rate"].nlargest(8)
    both = pd.concat([impat, loyal])
    colors = ["#e74c3c"] * len(impat) + [GREEN] * len(loyal)
    axes[2].barh(range(len(both)), both.values * 100, color=colors)
    axes[2].set_yticks(range(len(both)), both.index, fontsize=8)
    axes[2].set_title("Most impatient vs most loyal\n(artists with 50+ plays)")
    axes[2].set_xlabel("% of plays skipped")
    axes[2].set_xlim(0, 100)
    finish(fig, out, "Skip behavior")


def chart_platforms(df, out):
    fam = df["platform"].map(platform_family)
    mins = df.assign(fam=fam).groupby("fam")["minutes"].sum().sort_values()
    plays = df.assign(fam=fam).groupby("fam").size()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].barh(mins.index, mins.values / 60, color=GREEN)
    axes[0].set_title("Hours by device family")
    axes[1].barh(plays.index[::-1], plays.values[::-1], color="#169c46")
    axes[1].set_title("Plays by device family")
    for ax in axes:
        ax.tick_params(labelsize=9)
    finish(fig, out, "Where you listen")


def chart_discovery(df, out):
    d = df.sort_values("ts").copy()
    d["new_artist"] = first_play_flags(d, "artist")
    d["new_track"] = first_play_flags(d, "spotify_track_uri")
    per_month = d.groupby("month")[["new_artist", "new_track"]].sum()
    cum_artist = per_month["new_artist"].cumsum()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = [p.to_timestamp() for p in per_month.index]
    ax.bar(x, per_month["new_artist"], width=22, color=GREEN, alpha=0.55, label="new artists / month")
    ax2 = ax.twinx()
    ax2.plot(x, cum_artist, color="#7c4dff", lw=1.8, label="total unique artists (cumulative)")
    ax2.grid(False)
    lines = [ax.containers[0], ax2.lines[0]]
    ax.legend(lines, [l.get_label() for l in lines], frameon=False, loc="upper left")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    finish(fig, out, "Discovery curve - how much new music you absorb")


def chart_longtail(df, out):
    pc = df.groupby("spotify_track_uri").size().value_counts().sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    total = pc.sum()
    axes[0].bar(pc.index[:30], pc.values[:30], color=GREEN)
    axes[0].set_title(f"How many times tracks get played ({total:,} unique tracks)")
    axes[0].set_xlabel("number of plays per track")
    axes[0].set_ylabel("how many tracks")
    bins = [1, 2, 3, 5, 10, 25, 100, 10**9]
    labels = ["1", "2", "3-4", "5-9", "10-24", "25-99", "100+"]
    cut = pd.cut(df.groupby("spotify_track_uri").size(), bins, right=False, labels=labels)
    vc = cut.value_counts().reindex(labels)
    axes[1].pie(vc.values, labels=labels, autopct="%1.0f%%", colors=PALETTE, wedgeprops={"width": 0.45})
    axes[1].set_title("Share of catalog by replay depth")
    finish(fig, out, "The long tail of your library")


def chart_bump(df, out, n=8, last_years=5):
    max_year = df["year"].max()
    sub = df[df["year"] > max_year - last_years]
    yearly = sub.groupby(["year", "artist"])["minutes"].sum().reset_index()
    yearly["rank"] = yearly.groupby("year")["minutes"].rank(ascending=False, method="min")
    artists = yearly[yearly["rank"] <= n]["artist"].unique()
    piv = (
        yearly[yearly["artist"].isin(artists)]
        .pivot(index="year", columns="artist", values="rank")
        .sort_index()
    )
    piv = piv.where(piv <= n)
    years = piv.index.values
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    labels = []
    for i, artist in enumerate(piv.columns):
        r = piv[artist]
        c = PALETTE[i % len(PALETTE)]
        ax.plot(years, r, lw=2.2, alpha=0.8, color=c, marker="o", ms=4.5)
        lv = r.last_valid_index()
        if lv is not None:
            labels.append((artist, float(r[lv]), c, float(lv)))
    x_lab = years.max() + 0.12
    labels.sort(key=lambda t: t[1])
    ys = [y for _, y, _, _ in labels]
    gap = 0.38
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < gap:
            ys[i] = ys[i - 1] + gap
    if ys and ys[-1] > n + 0.5:
        ys[-1] = n + 0.5
        for i in range(len(ys) - 2, -1, -1):
            if ys[i] > ys[i + 1] - gap:
                ys[i] = ys[i + 1] - gap
    for (artist, y_true, c, lv), y_lab in zip(labels, ys):
        if abs(y_lab - y_true) > 0.05 or lv != years.max():
            ax.plot([lv, x_lab - 0.02], [y_true, y_lab], lw=0.7, ls=":",
                    color=c, alpha=0.45, zorder=1)
        ax.annotate(artist, (x_lab, y_lab), xytext=(4, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=8.5, color=c, fontweight="bold")
    ax.invert_yaxis()
    ax.set_yticks(range(1, n + 1))
    ax.set_xticks(years)
    ax.set_xlim(years.min() - 0.2, years.max() + 1.15)
    ax.set_title(f"Artist rank shifts, {years.min()}–{years.max()} (top {n} by minutes)")
    ax.set_ylabel("rank")
    finish(fig, out)


def chart_calendar(df, out):
    daily = df.groupby("date")["minutes"].sum()
    idx = pd.date_range(min(daily.index), max(daily.index), freq="D")
    daily = daily.reindex(idx.date, fill_value=0.0)
    years = sorted({d.year for d in daily.index})
    grid = np.full((len(years), 366), np.nan)
    for d, v in zip(daily.index, daily.values):
        doy = (pd.Timestamp(d) - pd.Timestamp(d.year, 1, 1)).days
        grid[years.index(d.year), doy] = v
    for yi, y in enumerate(years):
        ndays = 366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365
        grid[yi, ndays:] = np.nan
    cmap = LinearSegmentedColormap.from_list("spot", ["#f7faf8", "#c8f3d8", GREEN, "#0e6b31"])
    cmap.set_bad("#ffffff")
    fig, ax = plt.subplots(figsize=(13, 4.2))
    im = ax.pcolormesh(np.arange(366), np.arange(len(years)), grid, cmap=cmap)
    ax.set_yticks(np.arange(len(years)) + 0.5, [str(y) for y in years])
    ax.invert_yaxis()
    ax.grid(False)
    months = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    ax.set_xticks(months, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
    cb.set_label("minutes / day")
    finish(fig, out, "Every day you listened, 2018 - today")


def chart_sessions(df, out):
    s = sessionize(df[df["year"] >= 2019])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4), constrained_layout=True)
    d = s["duration_min"].clip(upper=240)
    axes[0].hist(d, bins=48, color=GREEN, alpha=0.8)
    med = s["duration_min"].median()
    axes[0].axvline(med, color="#333", ls="--", lw=1)
    axes[0].text(med + 3, axes[0].get_ylim()[1] * 0.9, f"median {med:.0f} min")
    axes[0].set_title("Listening session length")
    axes[0].set_xlabel("minutes (240+ clipped)")
    per_year = s.groupby(s["start"].dt.year).agg(n=("duration_min", "size"), avg=("duration_min", "mean"))
    ax2 = axes[1]
    ax2.bar(per_year.index.astype(str), per_year["n"], color="#169c46", alpha=0.5, label="sessions")
    ax2.set_ylabel("sessions / year")
    ax3 = ax2.twinx()
    ax3.plot(range(len(per_year)), per_year["avg"], color="#7c4dff", marker="o", lw=2, label="avg length")
    ax3.set_ylabel("avg session minutes")
    ax3.grid(False)
    ax2.set_title("Sessions per year and average length")
    finish(fig, out)


def chart_weekday_weekend(df, out):
    wk = df[df["dow"] < 5].groupby("hour")["minutes"].sum() / df[df["dow"] < 5]["date"].nunique()
    we = df[df["dow"] >= 5].groupby("hour")["minutes"].sum() / max(df[df["dow"] >= 5]["date"].nunique(), 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(wk.index, wk.values, marker="o", ms=3.5, color=GREEN, label="weekday avg")
    ax.plot(we.index, we.values, marker="o", ms=3.5, color="#7c4dff", label="weekend avg")
    ax.fill_between(wk.index, wk.values, alpha=0.12, color=GREEN)
    ax.fill_between(we.index, we.values, alpha=0.12, color="#7c4dff")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("hour of day")
    ax.set_ylabel("minutes / day")
    ax.legend(frameon=False)
    finish(fig, out, "Weekday vs weekend rhythm")


def chart_reasons(df, out):
    have = df.dropna(subset=["reason_end"])
    ends = have.groupby("reason_end").size().sort_values()
    starts = have.groupby("reason_start").size().sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    axes[0].barh(starts.index, starts.values, color="#169c46")
    axes[0].set_title("How playback starts")
    axes[1].barh(ends.index, ends.values, color="#0e6b31")
    axes[1].set_title("How playback ends")
    for ax in axes:
        ax.set_xlabel("plays")
    finish(fig, out, "Playback anatomy")


def chart_diversity(df, out):
    g = df.groupby("month").agg(artists=("artist", "nunique"), tracks=("spotify_track_uri", "nunique"), plays=("track", "size"))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = [p.to_timestamp() for p in g.index]
    ax.plot(x, g["artists"], color=GREEN, lw=1.6, label="unique artists")
    ax.plot(x, g["tracks"], color="#7c4dff", lw=1.6, label="unique tracks")
    ax.set_ylabel("uniques per month")
    ax2 = ax.twinx()
    ax2.plot(x, g["artists"] / g["plays"], color="#ff8a3d", lw=1.2, ls="--", label="artists/plays ratio")
    ax2.set_ylabel("variety ratio (unique artists per play)")
    ax2.grid(False)
    lines = ax.get_lines() + ax2.lines
    ax.legend(lines, [l.get_label() for l in lines], frameon=False, loc="upper left")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    finish(fig, out, "Musical variety over time")


def chart_replay_share(df, out):
    d = df.sort_values("ts").copy()
    d["is_replay"] = ~first_play_flags(d, "spotify_track_uri")
    monthly = d.groupby("month").apply(
        lambda x: (x.loc[x["is_replay"], "minutes"].sum() / x["minutes"].sum()) * 100,
        include_groups=False,
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    x = [p.to_timestamp() for p in monthly.index]
    ax.fill_between(x, monthly.values, alpha=0.3, color="#ff8a3d")
    ax.plot(x, monthly.values, color="#ff8a3d", lw=1.6)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of minutes that are replays")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    finish(fig, out, "Old favorites vs fresh finds")


def chart_top_days(df, out, n=12):
    daily = df.groupby("date")["minutes"].sum().sort_values(ascending=False).head(n)[::-1]
    labels = [d.strftime("%b %d, %Y") for d in daily.index]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(labels, daily.values / 60, color=GREEN)
    for i, v in enumerate(daily.values):
        ax.text(v / 60 + 0.05, i, f"{v/60:.1f}h", va="center", fontsize=8)
    ax.set_title(f"{n} biggest listening days ever")
    ax.set_xlabel("hours")
    finish(fig, out)


def chart_shuffle(df, out):
    have = df[df["shuffle"].notna()]
    if have.empty:
        return None
    per_year = have.groupby("year")["shuffle"].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([str(y) for y in per_year.index], per_year.values, color="#2196f3", alpha=0.8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of plays on shuffle")
    finish(fig, out, "Shuffle reliance by year")


def chart_genres(df, out):
    p = Path(__file__).parent.parent / "output" / "taxonomy" / "artists_taxonomy.csv"
    if not p.exists():
        print("  [skip] genres (no taxonomy output)")
        return
    tax = pd.read_csv(p).set_index("artist")
    m = df[df["artist"].isin(tax.index)].copy()
    m["tag"] = tax.loc[m["artist"], "genre"].values
    m = m[m["tag"].notna() & (m["tag"] != "unknown")]
    m["tag"] = m["tag"].str.split(", ")
    m = m.explode("tag")
    g = m.groupby("tag")["minutes"].sum().sort_values(ascending=False)
    if g.empty:
        print("  [skip] genres (empty)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    top = g.head(15)[::-1]
    axes[0].barh(top.index, top.values / 60, color=GREEN)
    axes[0].set_title("Top genres of all time")
    axes[0].set_xlabel("hours")
    top6 = g.head(6).index
    yearly = (
        m[m["tag"].isin(top6)]
        .pivot_table(index="year", columns="tag", values="minutes", aggfunc="sum")
        .fillna(0)
        .sort_index()
    )
    axes[1].stackplot(yearly.index, [yearly[c] / 60 for c in top6], labels=top6, colors=PALETTE[:6], alpha=0.88)
    axes[1].set_title("Top 6 genres over the years")
    axes[1].set_ylabel("hours")
    axes[1].legend(loc="upper left", fontsize=8, frameon=False)
    finish(fig, out, "Genre profile (from taxonomy labels)")


ALL_CHARTS = [
    ("01_monthly_listening", chart_monthly),
    ("02_when_you_listen", chart_heatmap),
    ("03_listening_clock", chart_clock),
    ("04_top_artists", chart_top_artists),
    ("05_top_tracks", chart_top_tracks),
    ("06_top_albums", chart_top_albums),
    ("07_top_artists_per_year", chart_yearly_artists),
    ("08_skip_behavior", chart_skips),
    ("09_platforms", chart_platforms),
    ("10_discovery_curve", chart_discovery),
    ("11_long_tail", chart_longtail),
    ("12_artist_rank_shifts", chart_bump),
    ("13_daily_calendar", chart_calendar),
    ("14_sessions", chart_sessions),
    ("15_weekday_vs_weekend", chart_weekday_weekend),
    ("16_playback_anatomy", chart_reasons),
    ("17_variety_over_time", chart_diversity),
    ("18_replay_share", chart_replay_share),
    ("19_biggest_days", chart_top_days),
    ("20_shuffle", chart_shuffle),
    ("25_top_genres", chart_genres),
]


def render_all(music, out_dir):
    out_dir = Path(out_dir)
    made = []
    for name, fn in ALL_CHARTS:
        try:
            fn(music, out_dir / f"{name}.png")
            made.append(name)
            print(f"  [ok] {name}")
        except Exception as e:
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
    return made
