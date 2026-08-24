from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .data import first_play_flags, platform_family

GREEN = "#1DB954"
DARK = "#191414"
FONT = "Inter, Segoe UI, sans-serif"


def _fig_html(fig, div_only=False):
    return fig.to_html(full_html=False, include_plotlyjs=("inline" if not div_only else False),
                       config={"displayModeBar": False})


def _kpi(label, value):
    return f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div></div>'


def _section(title, body):
    return f'<h2>{title}</h2>{body}'


def build(music, pods, out_path):
    out_path = Path(out_path)
    figs = []
    first = True

    def add(fig):
        nonlocal first
        figs.append(_fig_html(fig, div_only=not first))
        first = False

    m = music.groupby("month")["minutes"].sum().reset_index()
    m["month"] = m["month"].astype(str)
    m["hours"] = m["minutes"] / 60
    f1 = px.area(m, x="month", y="hours", title="Listening per month (drag to zoom)")
    f1.update_traces(line_color=GREEN, fillcolor="rgba(29,185,84,.25)")
    f1.update_layout(xaxis_rangeslider_visible=True)
    add(f1)

    piv = music.pivot_table(index="dow", columns="hour", values="minutes", aggfunc="sum").fillna(0)
    piv = piv.reindex(index=range(7), columns=range(24), fill_value=0)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    z = [[float(piv.iloc[d, h]) / 60 for h in range(24)] for d in range(7)]
    f2 = go.Figure(go.Heatmap(z=z, x=list(range(24)), y=days, colorscale="Greens",
                              hovertemplate="%{y} %{x}:00 - %{z:.1f} h<extra></extra>"))
    f2.update_layout(title="When you listen (local time)")
    add(f2)

    art = music.groupby("artist")["minutes"].sum().nlargest(60).reset_index()
    art.columns = ["artist", "minutes"]
    art["hours"] = art["minutes"] / 60
    f3 = px.treemap(art, path=["artist"], values="hours", color="hours", color_continuous_scale="Greens",
                    title="Artist universe (top 60 by hours)")
    add(f3)

    top_art = music.groupby("artist")["minutes"].sum().sort_values(ascending=True).tail(20) / 60
    f4 = go.Figure(go.Bar(x=top_art.values, y=top_art.index, orientation="h", marker_color=GREEN,
                          hovertemplate="%{y}: %{x:.1f} h<extra></extra>"))
    f4.update_layout(title="Top 20 artists of all time")
    add(f4)

    tt = (music.groupby(["track", "artist"])["minutes"].sum()
          .sort_values(ascending=True).tail(20).reset_index())
    tt["label"] = tt["track"] + " — " + tt["artist"]
    tt["hours"] = tt["minutes"] / 60
    f5 = go.Figure(go.Bar(x=tt["hours"], y=tt["label"], orientation="h", marker_color="#169c46",
                          hovertemplate="%{y}: %{x:.1f} h<extra></extra>"))
    f5.update_layout(title="Top 20 tracks of all time", height=650)
    add(f5)

    d = music.sort_values("ts").copy()
    d["new"] = first_play_flags(d, "artist")
    disc = d.groupby("month")["new"].sum().cumsum().reset_index()
    disc["month"] = disc["month"].astype(str)
    f6 = px.line(disc, x="month", y="new", title="Unique artists discovered (cumulative)")
    f6.update_traces(line_color="#7c4dff")
    add(f6)

    have = music[music["skipped"].notna()]
    if not have.empty:
        sk = have.groupby("month")["skipped"].mean().reset_index()
        sk["month"] = sk["month"].astype(str)
        sk["skip_rate"] = sk["skipped"] * 100
        f7 = px.area(sk, x="month", y="skip_rate", title="Skip rate % per month")
        f7.update_traces(line_color="#e74c3c", fillcolor="rgba(231,76,60,.15)")
        add(f7)

    plat = music.assign(fam=music["platform"].map(platform_family))
    pf = plat.groupby("fam")["minutes"].sum().nlargest(10).reset_index()
    pf.columns = ["platform", "minutes"]
    pf["hours"] = pf["minutes"] / 60
    f8 = px.pie(pf, names="platform", values="hours", hole=0.45, title="Devices", color_discrete_sequence=px.colors.sequential.Greens_r)
    add(f8)

    if not pods.empty:
        ps = pods.groupby("show")["minutes"].sum().sort_values(ascending=True).tail(12) / 60
        f9 = go.Figure(go.Bar(x=ps.values, y=ps.index, orientation="h", marker_color="#7c4dff",
                              hovertemplate="%{y}: %{x:.1f} h<extra></extra>"))
        f9.update_layout(title="Podcast shows by hours")
        add(f9)

    uris = music.dropna(subset=["spotify_track_uri"]).groupby(["track", "artist", "spotify_track_uri"])["minutes"].sum()
    rows = []
    for (tr, ar, uri), mins in uris.nlargest(50).items():
        tid = uri.split(":")[-1]
        rows.append(f"<tr><td>{tr}</td><td>{ar}</td><td>{mins/60:.1f} h</td>"
                    f"<td><a href='https://open.spotify.com/track/{tid}'>play</a></td></tr>")
    table = ("<table><tr><th>Track</th><th>Artist</th><th>Time</th><th></th></tr>"
             + "".join(rows) + "</table>")

    total_min = music["minutes"].sum() + pods["minutes"].sum()
    kpis = "".join([
        _kpi("total listening", f"{total_min/60:,.0f} h"),
        _kpi("music plays", f"{len(music):,}"),
        _kpi("unique artists", f"{music['artist'].nunique():,}"),
        _kpi("unique tracks", f"{music['spotify_track_uri'].nunique():,}"),
        _kpi("active days", f"{music['date'].nunique():,}"),
        _kpi("top artist", music.groupby('artist')['minutes'].sum().idxmax()),
    ])

    css = """
    body{font-family:%s;margin:0;background:#fafafa;color:#191414}
    .wrap{max-width:1100px;margin:0 auto;padding:24px}
    header{background:%s;color:#fff;padding:28px 24px}
    header h1{margin:0 0 6px;font-size:26px}
    header p{margin:0;opacity:.7;font-size:13px}
    .kpis{display:flex;flex-wrap:wrap;gap:14px;margin:18px 0}
    .kpi{background:#fff;border-radius:12px;padding:14px 22px;box-shadow:0 1px 4px rgba(0,0,0,.08)}
    .kpi .v{font-size:22px;font-weight:700;color:%s}
    .kpi .l{font-size:11px;text-transform:uppercase;letter-spacing:.08em;opacity:.55}
    h2{border-left:5px solid %s;padding-left:10px;margin-top:40px;font-size:20px}
    table{width:100%%;border-collapse:collapse;background:#fff;border-radius:8px;overflow:hidden}
    th,td{padding:7px 12px;border-bottom:1px solid #eee;font-size:14px;text-align:left}
    th{background:%s;color:#fff}
    a{color:%s}
    """ % (FONT, DARK, GREEN, GREEN, GREEN, GREEN)
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Spotify Wrapped Pro</title><style>{css}</style></head><body>
<header><div class='wrap'><h1>Your Spotify, the full picture</h1>
<p>{music['ts'].min():%b %Y} – {music['ts'].max():%b %Y} · America/Phoenix · plays ≥ 30s</p></div></header>
<div class='wrap'><div class='kpis'>{kpis}</div>
{''.join(_section(t, b) for t, b in zip([
 'Timeline','Listening clock','Artist universe','All-time artists','All-time tracks',
 'Discovery','Skip rate','Devices','Podcasts'], figs))}
<h2>Top 50 tracks with links</h2>{table}
</div></body></html>"""
    out_path.write_text(html, encoding="utf-8")
