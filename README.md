# Spotify history analysis

Turns a Spotify **Extended Streaming History** export into a private, offline analysis pack:
24 static charts, a written report, an interactive explorer with drill-down pages, and a
personal mood × social taxonomy of every artist you've played.

## Run it

```bash
./run.sh
```

That's it. The script creates a venv, installs deps, and runs the whole pipeline.
Requires Python 3.10+ and your export unzipped as `Spotify Extended Streaming History/`
in this folder (request it from spotify.com → account → privacy → extended streaming history).

**No data?** The repo bundles a sanitized demo dataset (`sample_data/` — 27 artists,
2 podcasts, IPs scrubbed). `./run.sh` falls back to it automatically, so a fresh clone
works out of the box.

## Outputs (`output/`)

| File | What it is |
|---|---|
| `explorer.html` | Interactive dashboard: year filters, clickable top lists, mood×social cell grid. Keep `drill.html` next to it — artist/track clicks open it. |
| `drill.html` | Per-artist / per-track detail page (timeline, hour×weekday heat, every-play strip, sub-tracks). URL-driven, back-button safe. |
| `report.md` | Headline stats, all-time tops, year-by-year table. |
| `png/` | 24 high-res charts (listening clock, calendar heatmap, skip behavior, discovery curve, …). |
| `taxonomy/` | `artists_taxonomy.csv` + `tracks_taxonomy.csv` (see below). |

All times are converted to **America/Phoenix**. Plays shorter than 30s are ignored.
IP addresses from the export are dropped on load and never written anywhere.

## The taxonomy layer

`spotiviz/taxonomy_prep.py` builds "grounding packets" (top tracks, listening clock, skip
rate per artist), batches them 25 artists per worker, and an LLM subagent swarm classifies
each artist into a personal mood × social grid:

```
MOOD:   sad | excited | normal
SOCIAL: alone | social | both
```

- 945 core artists (95% of listening time) → classified by workers with strict
  anti-hallucination rules (`unknown` + low confidence instead of guessing)
- Long-tail artists → inferred by co-play label propagation (marked low confidence)
- 30 multi-cell artists → additional track-level pass (~440 tracks)

The subagent runs are assistant-driven (not reproducible by `run.sh` alone); committed
results in `output/taxonomy/` are what the explorer embeds. Delete them and the explorer
still works — just without the cell filters.

## Layout

```
run.sh              one-command entry point
run_all.py          pipeline orchestrator (argv: [data_dir] [out_dir])
make_sample.py      builds the sanitized demo dataset from a real export
spotiviz/
  data.py           load, dedupe, timezone, sessionize
  audio_charts.py   21 music charts (matplotlib)
  podcast_charts.py podcast charts
  dashboard.py      static plotly dashboard
  explorer.py       interactive explorer + drill page (self-contained, offline)
  report.py         markdown report
  taxonomy_prep.py  grounding packets + batching + co-play graph
  taxonomy_merge.py validation, merge, propagation
sample_data/        sanitized demo export + taxonomy (committed)
venv/               created by run.sh (ignored)
```

Everything is computed locally; nothing leaves the machine except the explicit subagent
classification step, which sends artist names + your play stats to the model — no IPs,
no timestamps beyond hour-of-day buckets.
