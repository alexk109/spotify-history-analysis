import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from spotiviz.data import load_history, split_music_podcasts
from spotiviz.audio_charts import render_all
from spotiviz.podcast_charts import render_all as render_pods
from spotiviz.dashboard import build
from spotiviz.explorer import build as build_explorer
from spotiviz.report import build_report

ROOT = Path(__file__).parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "Spotify Extended Streaming History"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "output"
if not DATA.exists():
    sys.exit(f"Data folder not found: {DATA}")
PNG = OUT / "png"
PNG.mkdir(parents=True, exist_ok=True)

t0 = time.time()
print("Loading history…")
df = load_history(DATA)
music, pods = split_music_podcasts(df)
print(f"  {len(df):,} real plays total -> {len(music):,} music, {len(pods):,} podcast")
print(f"  span: {music['ts'].min():%Y-%m-%d} .. {music['ts'].max():%Y-%m-%d}")

print("Rendering music charts…")
made = render_all(music, PNG)
print("Rendering podcast charts…")
made += render_pods(music, pods, PNG)

print("Building dashboard…")
build(music, pods, OUT / "dashboard.html")

print("Building interactive explorer…")
build_explorer(music, pods, OUT / "explorer.html")

print("Writing report…")
build_report(music, pods, OUT / "report.md", made)

print(f"Done in {time.time()-t0:.1f}s -> {OUT}")
