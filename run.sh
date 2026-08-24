#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

DATA="Spotify Extended Streaming History"
SAMPLE="sample_data/Spotify Extended Streaming History"

if [ -d "$DATA" ]; then
  ARGS="$DATA"
elif [ -d "$SAMPLE" ]; then
  echo "No personal data found — running on the bundled sample dataset."
  ARGS="$SAMPLE"
  if [ ! -f output/taxonomy/artists_taxonomy.csv ] && [ -f sample_data/taxonomy/artists_taxonomy.csv ]; then
    mkdir -p output/taxonomy
    cp sample_data/taxonomy/*.csv output/taxonomy/
  fi
else
  echo "ERROR: no data folder found."
  echo "Either drop your export as '$DATA' in this folder"
  echo "(spotify.com → account → privacy → extended streaming history),"
  echo "or use the bundled demo: sample_data/ (already present in the repo)."
  exit 1
fi

if [ ! -d venv ]; then
  echo "Creating venv…"
  python3 -m venv venv
fi

echo "Installing dependencies…"
./venv/bin/pip install -q --disable-pip-version-check -r requirements.txt

echo "Running analysis…"
./venv/bin/python run_all.py "$ARGS"

echo
echo "Done. Open:"
echo "  output/explorer.html   (interactive explorer; keep drill.html next to it)"
echo "  output/report.md       (written report)"
echo "  output/png/            (24 static charts)"
