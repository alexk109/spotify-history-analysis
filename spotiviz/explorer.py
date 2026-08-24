import itertools
import json
from pathlib import Path

import pandas as pd

from .data import first_play_flags, platform_family

N_ENTITIES = 150
PLOTLY_REL = "venv/lib/python3.10/site-packages/plotly/package_data/plotly.min.js"


def _sparse(series, rnd=1):
    return {str(k): round(float(v), rnd) for k, v in series.items() if v > 0}


def _yearly(df, col="minutes", agg="sum"):
    g = df.groupby("year")[col]
    s = g.sum() if agg == "sum" else g.count()
    return {str(int(k)): round(float(v), 1) for k, v in s.items()}


def _exact_subset_counts(sets_by_year):
    years = sorted(sets_by_year)
    out = {}
    for r in range(len(years) + 1):
        for combo in itertools.combinations(years, r):
            mask = sum(1 << years.index(y) for y in combo)
            u = set()
            for y in combo:
                u |= sets_by_year[y]
            out[mask] = len(u)
    return out


def _entity_record(sub, name, artist, album, uri, day0):
    days = ((sub["local_ts"].dt.normalize() - day0).dt.days).astype(int)
    return {
        "n": name,
        "a": artist,
        "al": album,
        "u": uri or "",
        "my": _yearly(sub),
        "py": _yearly(sub, agg="count"),
        "sy": {str(int(k)): round(float(v) * 100) for k, v in sub.groupby("year")["skipped"].mean().items()},
        "m": _sparse(sub.groupby("month")["minutes"].sum()),
        "d": days.tolist(),
        "h": sub["local_ts"].dt.hour.tolist(),
        "tt": _subtracks(sub),
    }


def _subtracks(sub, n=10):
    top = sub.groupby("track")["minutes"].sum().nlargest(n)
    out = []
    for t in top.index:
        s = sub[sub["track"] == t]
        out.append([
            t,
            _yearly(s),
            (s.dropna(subset=["spotify_track_uri"])["spotify_track_uri"].iloc[0] if s["spotify_track_uri"].notna().any() else ""),
        ])
    return out


def _taxonomy_embed(music):
    base = Path(__file__).parent.parent
    ap = base / "output/taxonomy/artists_taxonomy.json"
    if not ap.exists():
        return {"artistCell": {}, "cellHour": {}, "trackCell": {}}
    tax = json.loads(ap.read_text(encoding="utf-8"))
    artist_cell = {
        r["artist"]: {"m": r["mood"], "s": r["social"], "c": r["confidence"], "g": r["genre"], "my": {}}
        for r in tax if r.get("classified")
    }
    m = music[music["artist"].isin(artist_cell)].copy()
    m["cell"] = m["artist"].map(lambda a: artist_cell[a]["m"] + "/" + artist_cell[a]["s"])
    cell_hour = {}
    for (cell, y), g in m.groupby(["cell", "year"]):
        h = g.groupby("hour")["minutes"].sum().reindex(range(24), fill_value=0)
        cell_hour.setdefault(cell, {})[str(int(y))] = [round(float(v), 1) for v in h]
    for (a, y), v in m.groupby(["artist", "year"])["minutes"].sum().items():
        artist_cell[a]["my"][str(int(y))] = round(float(v), 1)
    track_cell = {}
    tp = base / "output/taxonomy/tracks_taxonomy.csv"
    if tp.exists():
        import csv
        with open(tp, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                track_cell[f'{r["artist"]}|||{r["track"]}'] = [r["mood"], r["social"]]
    return {"artistCell": artist_cell, "cellHour": cell_hour, "trackCell": track_cell}


def prepare(music, pods):
    years = sorted(int(y) for y in music["year"].unique())
    day0 = music["local_ts"].min().normalize()
    d = music.sort_values("ts").copy()
    d["new"] = first_play_flags(d, "artist")
    fam = music["platform"].map(platform_family)
    dev = music.assign(fam=fam).groupby(["year", "fam"])["minutes"].sum()

    heat = {str(y): [0.0] * 168 for y in years}
    for (y, dow, hr), v in music.groupby(["year", "dow", "hour"])["minutes"].sum().items():
        heat[str(int(y))][int(dow) * 24 + int(hr)] = round(float(v), 1)

    artist_sets = {y: set(music.loc[music["year"] == y, "artist"]) for y in years}
    track_sets = {y: set(music.loc[music["year"] == y, "spotify_track_uri"].fillna(music["track"])) for y in years}
    day_sets = {y: set(music.loc[music["year"] == y, "date"]) for y in years}

    top_a = music.groupby("artist")["minutes"].sum().nlargest(N_ENTITIES)
    top_t = music.groupby(["track", "artist"], dropna=False)["minutes"].sum().nlargest(N_ENTITIES)
    artists = []
    for name in top_a.index:
        artists.append(_entity_record(music[music["artist"] == name], name, "", "", "", day0))
    tracks = []
    for (t, a) in top_t.index:
        sub = music[(music["track"] == t) & (music["artist"] == a)]
        uri = ""
        if sub["spotify_track_uri"].notna().any():
            uri = sub.dropna(subset=["spotify_track_uri"])["spotify_track_uri"].iloc[0]
        tracks.append(_entity_record(sub, t, a, sub["album"].iloc[0], uri, day0))

    d0n = music["local_ts"].min().normalize()
    daily = music.assign(dn=(music["local_ts"].dt.normalize() - d0n).dt.days).groupby("dn")["minutes"].sum()
    return {
        "years": years,
        "day0": int(day0.value // 86400 // 10**9),
        "global": {"my": _yearly(music), "py": _yearly(music, agg="count")},
        "monthly": _sparse(music.groupby("month")["minutes"].sum()),
        "daily": {str(int(k)): round(float(v), 1) for k, v in daily.items()},
        "heat": heat,
        "newArtists": {str(k): int(v) for k, v in d.groupby("month")["new"].sum().items()},
        "devices": {str(int(y)): {f: round(float(v), 1) for (_, f), v in g.items()} for y, g in dev.groupby(level=0)},
        "uniques": {
            "artists": _exact_subset_counts(artist_sets),
            "tracks": _exact_subset_counts(track_sets),
            "days": _exact_subset_counts(day_sets),
        },
        "artists": artists,
        "tracks": tracks,
        "pods": {
            "shows": {s: _yearly(g) for s, g in pods.groupby("show")},
            "monthly": _sparse(pods.groupby("month")["minutes"].sum()) if not pods.empty else {},
        },
        **_taxonomy_embed(music),
    }


SHARED_JS = r"""
const GREEN="#1DB954",MUT="#a7a7a7",GRID="#282828";
const $=id=>document.getElementById(id);
const DAY0=D.day0*86400000;
function fmtH(m){const h=m/60;return h>=100?Math.round(h).toLocaleString()+" h":h>=10?h.toFixed(1)+" h":h.toFixed(2)+" h"}
function inYears(y){return state.years.has(+y)}
function sumDict(o){let s=0;for(const k in o)if(inYears(k))s+=o[k];return s}
function barL(labels){let m=0;labels.forEach(l=>{const n=String(l).length;if(n>m)m=n});return Math.min(250,24+m*6.4)}
function lay(h,extra){return Object.assign({height:h,margin:{l:56,r:14,t:26,b:40},paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"rgba(0,0,0,0)",font:{color:MUT,size:12},xaxis:{gridcolor:GRID,zeroline:false},yaxis:{gridcolor:GRID,zeroline:false}},extra||{})}
function barlay(labels,h,extra){return lay(h,Object.assign({margin:{l:barL(labels),r:14,t:26,b:40},yaxis:{gridcolor:"rgba(0,0,0,0)",tickfont:{size:11.5}}},extra||{}))}
function monthsInFilter(){return Object.keys(D.monthly).filter(m=>inYears(m.slice(0,4))).sort()}
function draw(id,data,layout,onClick){const gd=$(id);if(!gd)return;gd.removeAllListeners&&gd.removeAllListeners("plotly_click");Plotly.react(gd,data,layout,{displayModeBar:false,responsive:true});if(onClick)gd.on("plotly_click",ev=>{const p=ev.points[0];if(p&&p.customdata!==undefined&&p.customdata!==null)onClick(p.customdata)})}
function findEntity(code){const [kind,idx]=code.split(":");return {kind:kind==="A"?"artists":"tracks",rec:D[kind==="A"?"artists":"tracks"][+idx]}}
function entitySubtitle(kind,rec){const ac=D.artistCell[rec.n],tc=kind==="tracks"?D.trackCell[rec.a+"|||"+rec.n]:null;
const badge=ac?' · <span style="color:var(--green);font-weight:700">'+ac.m+"/"+ac.s+"</span> ("+ac.c+") · "+ac.g:(tc?' · <span style="color:var(--green);font-weight:700">'+tc[0]+"/"+tc[1]+"</span>":"");
return (kind==="tracks"?"by "+rec.a+(rec.al?" · "+rec.al:""):"artist")+badge}
function kpiHTML(items){return '<div class="kpis">'+items.map(i=>'<div class="kpi"><div class="v">'+i[1]+'</div><div class="l">'+i[0]+"</div></div>").join("")+"</div>"}
"""

EXPLORER_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Spotify Explorer</title>
<style>
:root{--bg:#121212;--card:#181818;--card2:#202020;--green:#1DB954;--txt:#fff;--mut:#a7a7a7;--line:#282828}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);font:14px/1.5 Inter,Segoe UI,system-ui,sans-serif}
.bar{position:sticky;top:0;z-index:10;background:rgba(18,18,18,.95);backdrop-filter:blur(6px);border-bottom:1px solid var(--line);padding:12px 22px}
.bar h1{margin:0 0 8px;font-size:19px}
.bar h1 span{color:var(--green)}
.chips,.chips2{display:flex;flex-wrap:wrap;gap:7px;align-items:center}
.chips2{margin-top:8px}
.chip{border:1px solid var(--line);background:var(--card);color:var(--mut);border-radius:999px;padding:4px 13px;cursor:pointer;font-size:12.5px;user-select:none}
.chip.on{background:var(--green);border-color:var(--green);color:#000;font-weight:700}
.chip.presets{border-style:dashed}
input#q{background:var(--card);border:1px solid var(--line);color:var(--txt);border-radius:999px;padding:5px 14px;font-size:13px;width:250px}
.wrap{max-width:1180px;margin:0 auto;padding:18px 22px 60px}
.kpis{display:flex;flex-wrap:wrap;gap:12px;margin:14px 0}
.kpi{background:var(--card);border-radius:10px;padding:12px 20px;min-width:130px}
.kpi .v{font-size:21px;font-weight:800;color:var(--green)}
.kpi .l{font-size:10.5px;text-transform:uppercase;letter-spacing:.09em;color:var(--mut)}
h2{font-size:17px;margin:34px 0 4px}
h2 small{color:var(--mut);font-weight:400;font-size:12px}
.card{background:var(--card);border-radius:12px;padding:8px 6px;margin-top:8px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:900px){.grid2{grid-template-columns:1fr}}
.hint{color:var(--mut);font-size:11.5px;margin:2px 0 0}
a{color:var(--green)}
.cellgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;max-width:600px;margin-top:8px}
.cbtn{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px;cursor:pointer;user-select:none}
.cbtn.on{border-color:var(--green);background:#15251c}
.cbtn .h{font-size:17px;font-weight:800;color:var(--green)}
.cbtn .n{font-size:11.5px;color:var(--mut)}
</style></head><body>
<div class="bar"><h1>Spotify <span>Explorer</span></h1>
<div class="chips" id="chips"></div>
<div class="chips2"><span class="hint">quick:</span>
<span class="chip presets" data-preset="2024,2025,2026">24–26</span>
<span class="chip presets" data-preset="2025,2026">25–26</span>
<input id="q" list="qlist" placeholder="search artist or track…"><datalist id="qlist"></datalist>
</div></div>
<div class="wrap">
<div class="kpis" id="kpis"></div>
<div class="grid2">
<div><h2>Top artists <small>click one to open its page</small></h2><div class="card"><div id="ta"></div></div></div>
<div><h2>Top tracks <small>click one to open its page</small></h2><div class="card"><div id="tt"></div></div></div>
</div>
<h2>Genres <small>from taxonomy labels · click a genre for its artists</small></h2>
<div class="grid2">
<div class="card"><div id="genres"></div></div>
<div class="card"><div id="genretime"></div></div>
</div>
<div class="card"><div class="hint" id="genreartists-title" style="margin:6px 10px">click a genre above to see its artists</div><div id="genreartists"></div></div>
<h2>Listening over time <small>hours per day · drag the slider to zoom</small></h2><div class="card"><div id="tl"></div></div>
<h2>When you listen <small>local time (Phoenix)</small></h2><div class="card"><div id="heat"></div></div>
<h2>Mood × social cells <small>your taxonomy · click a cell to explore</small></h2>
<div class="cellgrid" id="cellgrid"></div><div id="cellview"></div>
<h2>Podcast shows</h2><div class="card"><div id="pods"></div></div>
<div class="grid2">
<div><h2>Devices</h2><div class="card"><div id="dev"></div></div></div>
<div><h2>Artists discovered <small>cumulative</small></h2><div class="card"><div id="disc"></div></div></div>
</div>
</div>
<script>__PLOTLY_JS__</script>
<script>const D = __DATA__;</script>
<script>
__SHARED_JS__
const state={years:new Set(D.years)};
function maskOf(){let m=0;D.years.forEach((y,i)=>{if(state.years.has(y))m|=1<<i});return m}
function openDrill(code){const y=[...state.years].join(",");window.open("drill.html?e="+encodeURIComponent(code)+"&y="+encodeURIComponent(y),"_blank")}
function renderChips(){const c=$("chips");c.innerHTML="";const all=document.createElement("span");all.className="chip"+(state.years.size===D.years.length?" on":"");all.textContent="All time";all.onclick=()=>{state.years=new Set(D.years);refresh()};c.appendChild(all);D.years.forEach(y=>{const s=document.createElement("span");s.className="chip"+(state.years.has(y)?" on":"");s.textContent=y;s.onclick=()=>{state.years.has(y)?state.years.delete(y):state.years.add(y);if(!state.years.size)state.years=new Set(D.years);refresh()};c.appendChild(s)})}
function renderKPIs(){const g=D.global,m=maskOf(),u=D.uniques;
const items=[["listening",fmtH(sumDict(g.my))],["plays",Math.round(sumDict(g.py)).toLocaleString()],["unique artists",u.artists[m].toLocaleString()],["unique tracks",u.tracks[m].toLocaleString()],["active days",u.days[m].toLocaleString()]];
const ys=[...state.years].sort();items.push(["period",ys.length===D.years.length?"all time":ys.join(" · ")]);
$("kpis").innerHTML=kpiHTML(items)}
function renderTimeline(){const days=Object.keys(D.daily).map(Number).filter(d=>inYears(new Date(DAY0+d*86400000).getFullYear())).sort((a,b)=>a-b);
draw("tl",[{x:days.map(d=>new Date(DAY0+d*86400000)),y:days.map(d=>+(D.daily[d]/60).toFixed(2)),type:"scatter",mode:"lines",fill:"tozeroy",line:{color:GREEN,width:1},fillcolor:"rgba(29,185,84,.18)"}],
lay(320,{xaxis:{gridcolor:GRID,type:"date",rangeslider:{visible:true}},yaxis:{gridcolor:GRID,title:"hours / day"},height:380}))}
function renderHeat(){const z=[[],[],[],[],[],[],[]];const days=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"];
for(let d=0;d<7;d++)for(let h=0;h<24;h++)z[d][h]=+(D.years.reduce((s,y)=>s+(inYears(y)?D.heat[y][d*24+h]:0),0)/60).toFixed(2);
draw("heat",[{z:z,x:[...Array(24).keys()],y:days,type:"heatmap",colorscale:"Greens",hovertemplate:"%{y} %{x}:00 — %{z:.1f} h<extra></extra>",colorbar:{thickness:10}}],lay(280,{margin:{l:56,r:14,t:26,b:40},yaxis:{autorange:"reversed",gridcolor:"rgba(0,0,0,0)"},xaxis:{gridcolor:"rgba(0,0,0,0)",dtick:2}}))}
function renderTop(kind,div){const list=D[kind];
const rows=list.map((e,i)=>({k:(kind==="artists"?"A:":"T:")+i,n:e.n,a:e.a,v:sumDict(e.my)})).filter(e=>e.v>0).sort((a,b)=>a.v-b.v).slice(-20);
const labels=rows.map(r=>r.n);
draw(div,[{x:rows.map(r=>+(r.v/60).toFixed(1)),y:labels,type:"bar",orientation:"h",marker:{color:GREEN},customdata:rows.map(r=>[r.k,r.a]),hovertemplate:"%{y}"+(kind==="tracks"?" — %{customdata[1]}":"")+": %{x} h<extra></extra>"}],
barlay(labels,600,{xaxis:{title:"hours in selected years",gridcolor:GRID}}),(cd)=>openDrill(cd[0]))}
function renderDevices(){const agg={};D.years.forEach(y=>{const dd=D.devices[y]||{};for(const k in dd)agg[k]=(agg[k]||0)+dd[k]});
const items=Object.entries(agg).sort((a,b)=>b[1]-a[1]).slice(0,8);
draw("dev",[{labels:items.map(i=>i[0]),values:items.map(i=>+(i[1]/60).toFixed(1)),type:"pie",hole:.5,marker:{colors:["#1DB954","#169c46","#0e6b31","#7c4dff","#b3b3b3","#535353","#ff8a3d","#2196f3"]},hovertemplate:"%{label}: %{value} h<extra></extra>"}],lay(320))}
function renderDiscovery(){const ms=monthsInFilter();let c=0;const ys=ms.map(m=>{c+=D.newArtists[m]||0;return c});
draw("disc",[{x:ms,y:ys,type:"scatter",mode:"lines",line:{color:"#7c4dff",width:2}}],lay(320,{yaxis:{gridcolor:GRID,title:"unique artists"},xaxis:{type:"date",gridcolor:GRID}}))}
function renderPods(){const rows=Object.entries(D.pods.shows).map(([s,my])=>({s,v:sumDict(my)})).filter(r=>r.v>0).sort((a,b)=>a.v-b.v).slice(-12);
const labels=rows.map(r=>r.s);
draw("pods",[{x:rows.map(r=>+(r.v/60).toFixed(1)),y:labels,type:"bar",orientation:"h",marker:{color:"#7c4dff"},hovertemplate:"%{y}: %{x} h<extra></extra>"}],
barlay(labels,360,{xaxis:{title:"hours in selected years",gridcolor:GRID}}))}
const CELLS=["sad/alone","sad/social","sad/both","normal/alone","normal/social","normal/both","excited/alone","excited/social","excited/both"];
let curCell=null;
function cellHours(cell){const ch=D.cellHour[cell]||{};let h=Array(24).fill(0);for(const y in ch)if(inYears(y))h=h.map((v,i)=>v+ch[y][i]);return h}
function cellMins(cell){let s=0;for(const a in D.artistCell){const ac=D.artistCell[a];if(ac.m+"/"+ac.s===cell)for(const y in ac.my)if(inYears(y))s+=ac.my[y]}return s}
function renderCellGrid(){const g=$("cellgrid");g.innerHTML="";CELLS.forEach(c=>{const mins=cellMins(c);const b=document.createElement("div");b.className="cbtn"+(curCell===c?" on":"");b.innerHTML='<div class="h">'+(mins/60).toFixed(0)+' h</div><div class="n">'+c+"</div>";b.onclick=()=>{curCell=curCell===c?null:c;renderCellGrid();renderCell()};g.appendChild(b)})}
function renderCell(){const v=$("cellview");if(!curCell){v.innerHTML="";return}
const artists=Object.entries(D.artistCell).filter(([,ac])=>ac.m+"/"+ac.s===curCell).map(([a,ac])=>({a:a,v:Object.entries(ac.my).filter(([y])=>inYears(y)).reduce((s,[,x])=>s+x,0)})).filter(x=>x.v>0).sort((a,b)=>a.v-b.v).slice(-15);
const labels=artists.map(r=>r.a);
v.innerHTML='<div class="card" style="margin-top:10px"><div id="cell-artists"></div></div><div class="card"><div id="cell-clock"></div></div>';
draw("cell-artists",[{x:artists.map(r=>+(r.v/60).toFixed(1)),y:labels,type:"bar",orientation:"h",marker:{color:GREEN},customdata:artists.map(r=>["A:"+D.artists.findIndex(e=>e.n===r.a)]),hovertemplate:"%{y}: %{x} h<extra></extra>"}],
barlay(labels,40+artists.length*24,{xaxis:{title:"hours in selected years",gridcolor:GRID}}),(cd)=>{if(cd[0]!=="A:-1")openDrill(cd[0])});
const h=cellHours(curCell);
draw("cell-clock",[{x:[...Array(24).keys()],y:h.map(x=>+(x/60).toFixed(1)),type:"bar",marker:{color:"#7c4dff"},hovertemplate:"%{x}:00 — %{y} h<extra></extra>"}],lay(240,{xaxis:{title:"hour of day",dtick:3,gridcolor:GRID},yaxis:{title:"hours",gridcolor:GRID}}))}
const GENRE_COLORS=["#1DB954","#7c4dff","#ff8a3d","#2196f3","#f573a0","#ffc107"];
function genreAgg(){const tags={};for(const a in D.artistCell){const ac=D.artistCell[a];if(!ac.g||ac.g==="unknown")continue;
ac.g.split(",").forEach(gg=>{const g=gg.trim();if(!g||g==="unknown")return;const t=tags[g]=tags[g]||{v:0,years:{}};
for(const y in ac.my)if(inYears(y)){t.v+=ac.my[y];t.years[y]=(t.years[y]||0)+ac.my[y]}})}return tags}
function renderGenres(){const tags=genreAgg();const rows=Object.entries(tags).map(([g,t])=>({g:g,v:t.v,y:t.years})).filter(r=>r.v>0).sort((a,b)=>a.v-b.v).slice(-15);
const labels=rows.map(r=>r.g);
draw("genres",[{x:rows.map(r=>+(r.v/60).toFixed(1)),y:labels,type:"bar",orientation:"h",marker:{color:GREEN},customdata:rows.map(r=>[r.g]),hovertemplate:"%{y}: %{x} h<extra></extra>"}],
barlay(labels,560,{xaxis:{title:"hours in selected years",gridcolor:GRID}}),(cd)=>renderGenreArtists(cd[0]));
const top6=Object.entries(tags).map(([g,t])=>({g:g,v:t.v,y:t.years})).sort((a,b)=>b.v-a.v).slice(0,6);
const ys=D.years.filter(y=>state.years.has(y));
const traces=top6.map((t,i)=>({x:ys.map(String),y:ys.map(y=>+((t.years[y]||0)/60).toFixed(2)),type:"scatter",mode:"lines",stackgroup:"one",name:t.g,line:{width:2,color:GENRE_COLORS[i%6]},hovertemplate:t.g+": %{y} h<extra></extra>"}));
draw("genretime",traces,lay(560,{yaxis:{gridcolor:GRID,title:"hours"},xaxis:{gridcolor:GRID},legend:{orientation:"h",y:-0.12}}))}
function renderGenreArtists(tag){const rows=Object.entries(D.artistCell).filter(([,ac])=>ac.g&&ac.g.split(",").map(s=>s.trim()).includes(tag)).map(([a,ac])=>({a:a,v:Object.entries(ac.my).filter(([y])=>inYears(y)).reduce((s,[,x])=>s+x,0)})).filter(r=>r.v>0).sort((a,b)=>a.v-b.v).slice(-12);
const labels=rows.map(r=>r.a);
$("genreartists-title").textContent="Top artists in “"+tag+"”";
draw("genreartists",[{x:rows.map(r=>+(r.v/60).toFixed(1)),y:labels,type:"bar",orientation:"h",marker:{color:"#7c4dff"},customdata:rows.map(r=>["A:"+D.artists.findIndex(e=>e.n===r.a)]),hovertemplate:"%{y}: %{x} h<extra></extra>"}],
barlay(labels,40+labels.length*24,{xaxis:{title:"hours in selected years",gridcolor:GRID}}),(cd)=>{if(cd[0]!=="A:-1")openDrill(cd[0])})}
function refresh(){renderChips();renderKPIs();renderTimeline();renderHeat();renderTop("artists","ta");renderTop("tracks","tt");renderDevices();renderDiscovery();renderPods();renderCellGrid();renderCell();renderGenres()}
renderChips();renderKPIs();renderTimeline();renderHeat();renderTop("artists","ta");renderTop("tracks","tt");renderDevices();renderDiscovery();renderPods();renderCellGrid();renderGenres();
const ql=$("qlist");D.artists.forEach(e=>{const o=document.createElement("option");o.value=e.n;o.label="artist";ql.appendChild(o)});D.tracks.forEach(e=>{const o=document.createElement("option");o.value=e.n;o.label="track — "+e.a;ql.appendChild(o)});
$("q").addEventListener("change",e=>{const v=e.target.value;if(!v)return;
const ai=D.artists.findIndex(x=>x.n===v);if(ai>=0){openDrill("A:"+ai);e.target.value="";return}
const ti=D.tracks.findIndex(x=>x.n===v);if(ti>=0){openDrill("T:"+ti);e.target.value="";return}});
document.querySelectorAll(".presets").forEach(ch=>ch.onclick=()=>{state.years=new Set(ch.dataset.preset.split(",").map(Number));refresh()});
</script></body></html>
"""

DRILL_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Drill-down — Spotify Explorer</title>
<style>
:root{--bg:#121212;--card:#181818;--card2:#202020;--green:#1DB954;--txt:#fff;--mut:#a7a7a7;--line:#282828}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);font:14px/1.5 Inter,Segoe UI,system-ui,sans-serif}
.bar{position:sticky;top:0;z-index:10;background:rgba(18,18,18,.95);backdrop-filter:blur(6px);border-bottom:1px solid var(--line);padding:12px 22px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
.bar a.home{color:var(--green);text-decoration:none;font-weight:700;white-space:nowrap}
.bar .chips{display:flex;flex-wrap:wrap;gap:7px;align-items:center}
.chip{border:1px solid var(--line);background:var(--card);color:var(--mut);border-radius:999px;padding:4px 13px;cursor:pointer;font-size:12.5px;user-select:none}
.chip.on{background:var(--green);border-color:var(--green);color:#000;font-weight:700}
.wrap{max-width:1180px;margin:0 auto;padding:18px 22px 60px}
.kpis{display:flex;flex-wrap:wrap;gap:12px;margin:14px 0}
.kpi{background:var(--card);border-radius:10px;padding:12px 20px;min-width:130px}
.kpi .v{font-size:21px;font-weight:800;color:var(--green)}
.kpi .l{font-size:10.5px;text-transform:uppercase;letter-spacing:.09em;color:var(--mut)}
h1{margin:18px 0 2px;font-size:26px}
h2{font-size:17px;margin:30px 0 4px}
h2 small{color:var(--mut);font-weight:400;font-size:12px}
.sub{color:var(--mut);margin-bottom:6px}
.card{background:var(--card);border-radius:12px;padding:8px 6px;margin-top:8px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:900px){.grid2{grid-template-columns:1fr}}
a{color:var(--green)}
#msg{color:var(--mut);padding:40px 0}
</style></head><body>
<div class="bar"><a class="home" href="explorer.html">← Explorer</a><div class="chips" id="chips"></div></div>
<div class="wrap"><div id="content"><p id="msg">Nothing selected. Open something from the explorer, or search:</p>
<input id="q" list="qlist" placeholder="search artist or track…" style="background:var(--card);border:1px solid var(--line);color:var(--txt);border-radius:999px;padding:6px 14px;font-size:13px"><datalist id="qlist"></datalist>
</div></div>
<script>__PLOTLY_JS__</script>
<script>const D = __DATA__;</script>
<script>
__SHARED_JS__
const params=new URLSearchParams(location.search);
const state={years:new Set((params.get("y")||"").split(",").filter(x=>x&&D.years.includes(+x)).map(Number))};
if(!state.years.size)state.years=new Set(D.years);
let entity=params.get("e")||null;
function setYears(){renderChips();render()}
function renderChips(){const c=$("chips");c.innerHTML="";const all=document.createElement("span");all.className="chip"+(state.years.size===D.years.length?" on":"");all.textContent="All time";all.onclick=()=>{state.years=new Set(D.years);syncURL();setYears()};c.appendChild(all);D.years.forEach(y=>{const s=document.createElement("span");s.className="chip"+(state.years.has(y)?" on":"");s.textContent=y;s.onclick=()=>{state.years.has(y)?state.years.delete(y):state.years.add(y);if(!state.years.size)state.years=new Set(D.years);syncURL();setYears()};c.appendChild(s)})}
function syncURL(){const p=new URLSearchParams();if(entity)p.set("e",entity);if(state.years.size!==D.years.length)p.set("y",[...state.years].join(","));history.replaceState({e:entity},"", "?"+p.toString())}
function navigate(code,push){entity=code;if(push)history.pushState({e:code},"", "?e="+encodeURIComponent(code)+"&y="+encodeURIComponent([...state.years].join(",")));window.scrollTo(0,0);render()}
function render(){const box=$("content");
if(!entity){box.innerHTML='<p id="msg">Nothing selected.</p>';return}
const {kind,rec}=findEntity(entity);
if(!rec){box.innerHTML='<p id="msg">Not found.</p>';return}
document.title=rec.n+" — Spotify Explorer";
const mins=sumDict(rec.my),plays=sumDict(rec.py);
let firstD="—",lastD="—";
const daysIn=rec.d.map((dd,i)=>({d:dd,h:rec.h[i]})).filter(p=>inYears(new Date(DAY0+p.d*86400000).getFullYear()));
if(daysIn.length){const ds=daysIn.map(p=>p.d);firstD=new Date(DAY0+Math.min(...ds)*86400000).toLocaleDateString();lastD=new Date(DAY0+Math.max(...ds)*86400000).toLocaleDateString()}
let num=0,den=0;for(const y in rec.sy){if(inYears(y)){num+=rec.sy[y]*rec.py[y];den+=rec.py[y]}}
const skip=den?(num/den).toFixed(0)+"%":"—";
const link=rec.u?'<a href="https://open.spotify.com/track/'+rec.u.split(":")[2]+'" target="_blank">open in Spotify ↗</a>':"";
const ms=monthsInFilter();
const labels=kind==="artists"?rec.tt.map(t=>t[0]).filter(t=>{const r=rec.tt.find(x=>x[0]===t);return sumDict(r[1])>0}):[];
box.innerHTML='<h1>'+rec.n+'</h1><div class="sub">'+entitySubtitle(kind,rec)+(link?" · "+link:"")+'</div>'
+kpiHTML([["time",mins?fmtH(mins):"—"],["plays",Math.round(plays).toLocaleString()],["skipped",skip],["first play",firstD],["last play",lastD]])
+'<div class="grid2"><div><h2 style="margin-top:6px">Minutes per month</h2><div class="card"><div id="d-tl"></div></div></div>'
+'<div><h2 style="margin-top:6px">Hour × weekday</h2><div class="card"><div id="d-heat"></div></div></div></div>'
+'<h2>Every play <small>each dot = one play, by date &amp; time of day</small></h2><div class="card"><div id="d-strip"></div></div>'
+(kind==="artists"&&rec.tt.length?'<h2>Top tracks for this artist <small>click to open its page</small></h2><div class="card"><div id="d-tt"></div></div>':"");
draw("d-tl",[{x:ms,y:ms.map(k=>+(rec.m[k]||0).toFixed(1)),type:"scatter",mode:"lines",fill:"tozeroy",line:{color:GREEN,width:2},fillcolor:"rgba(29,185,84,.18)"}],lay(280,{xaxis:{type:"date",gridcolor:GRID},yaxis:{gridcolor:GRID,title:"minutes"}}));
const zb=[[],[],[],[],[],[],[]];daysIn.forEach(p=>{const dt=new Date(DAY0+p.d*86400000);const dow=(dt.getDay()+6)%7;zb[dow][p.h]=(zb[dow][p.h]||0)+1});
draw("d-heat",[{z:zb,x:[...Array(24).keys()],y:["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],type:"heatmap",colorscale:"Greens",hovertemplate:"%{y} %{x}:00 — %{z} plays<extra></extra>",colorbar:{thickness:10}}],lay(280,{margin:{l:56,r:14,t:26,b:40},yaxis:{autorange:"reversed",gridcolor:"rgba(0,0,0,0)"},xaxis:{gridcolor:"rgba(0,0,0,0)",dtick:3}}));
draw("d-strip",[{x:daysIn.map(p=>new Date(DAY0+p.d*86400000)),y:daysIn.map(p=>p.h),type:"scatter",mode:"markers",marker:{color:GREEN,size:5,opacity:.55},hovertemplate:"%{x|%b %d, %Y} at %{y}:00<extra></extra>"}],lay(280,{xaxis:{type:"date",gridcolor:GRID},yaxis:{gridcolor:GRID,dtick:3,title:"hour of day"}}));
if(kind==="artists"&&rec.tt.length){const rows=rec.tt.map(t=>({n:t[0],v:sumDict(t[1])})).filter(r=>r.v>0).sort((a,b)=>a.v-b.v);
const tl=rows.map(r=>r.n);
draw("d-tt",[{x:rows.map(r=>+(r.v/60).toFixed(2)),y:tl,type:"bar",orientation:"h",marker:{color:"#169c46"},customdata:rows.map(r=>["T:"+D.tracks.findIndex(t=>t.n===r.n&&t.a===rec.n)]),hovertemplate:"%{y}: %{x} h<extra></extra>"}],
barlay(tl,60+tl.length*24,{xaxis:{title:"hours in selected years",gridcolor:GRID}}),(cd)=>{if(cd[0]!=="T:-1")navigate(cd[0],true)})}}
window.onpopstate=ev=>{entity=(ev.state&&ev.state.e)||params.get("e")||null;render()};
renderChips();render();
const ql=$("qlist");D.artists.forEach(e=>{const o=document.createElement("option");o.value=e.n;o.label="artist";ql.appendChild(o)});D.tracks.forEach(e=>{const o=document.createElement("option");o.value=e.n;o.label="track — "+e.a;ql.appendChild(o)});
const q=$("q");
if(q)q.addEventListener("change",e=>{const v=e.target.value;if(!v)return;
const ai=D.artists.findIndex(x=>x.n===v);if(ai>=0){navigate("A:"+ai,true);e.target.value="";return}
const ti=D.tracks.findIndex(x=>x.n===v);if(ti>=0){navigate("T:"+ti,true);e.target.value="";return}});
</script></body></html>
"""


def build(music, pods, out_path):
    out_path = Path(out_path)
    data = prepare(music, pods)
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    plotly_js = Path(__file__).parent.parent / PLOTLY_REL
    pjs = plotly_js.read_text(encoding="utf-8") if plotly_js.exists() else ""
    out_path.write_text(
        EXPLORER_HTML.replace("__PLOTLY_JS__", pjs).replace("__DATA__", payload).replace("__SHARED_JS__", SHARED_JS),
        encoding="utf-8",
    )
    out_path.parent.joinpath("drill.html").write_text(
        DRILL_HTML.replace("__PLOTLY_JS__", pjs).replace("__DATA__", payload).replace("__SHARED_JS__", SHARED_JS),
        encoding="utf-8",
    )
    return data
