import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
SAMPLE_ARTISTS = [
    "Wolfgang Amadeus Mozart", "Frédéric Chopin", "Johann Sebastian Bach",
    "Antonio Vivaldi", "Camille Saint-Saëns",
    "Kendrick Lamar", "Kanye West", "MF DOOM", "Chief Keef", "BONES",
    "Tyler, The Creator", "ДЕТИ RAVE",
    "Lana Del Rey", "SZA", "Arctic Monkeys", "Fleet Foxes", "Weyes Blood",
    "Suki Waterhouse", "Frank Ocean", "Elton John", "Nirvana",
    "Talking Heads", "Crystal Castles", "Fred again..",
    "Howard Shore", "Ennio Morricone", "C418",
]
SAMPLE_SHOWS = ["Lateral with Tom Scott", "Theory & Philosophy"]


def _resolve_names(data_names):
    lookup = {n.casefold(): n for n in data_names}
    resolved = {}
    for want in SAMPLE_ARTISTS:
        got = lookup.get(want.casefold())
        if not got:
            raise SystemExit(f"sample artist not found in data: {want}")
        resolved[got] = want
    return set(resolved)


def run(data_dir):
    src = Path(data_dir)
    dst = ROOT / "sample_data" / "Spotify Extended Streaming History"
    dst.mkdir(parents=True, exist_ok=True)

    all_names = set()
    for f in src.glob("Streaming_History_*.json"):
        for r in json.loads(f.read_text(encoding="utf-8")):
            if r.get("master_metadata_album_artist_name"):
                all_names.add(r["master_metadata_album_artist_name"])
    names = _resolve_names(all_names)

    total = 0
    for f in sorted(src.glob("Streaming_History_*.json")):
        rows = json.loads(f.read_text(encoding="utf-8"))
        keep = []
        for r in rows:
            if r.get("master_metadata_album_artist_name") in names or r.get("episode_show_name") in SAMPLE_SHOWS:
                r["ip_addr"] = "127.0.0.1"
                keep.append(r)
        if keep:
            (dst / f.name).write_text(json.dumps(keep, ensure_ascii=False), encoding="utf-8")
            total += len(keep)
            print(f"  {f.name}: kept {len(keep)}/{len(rows)}")

    tsrc = ROOT / "output" / "taxonomy"
    tdst = ROOT / "sample_data" / "taxonomy"
    tdst.mkdir(parents=True, exist_ok=True)
    for name, col in [("artists_taxonomy.csv", "artist"), ("tracks_taxonomy.csv", "artist")]:
        df = pd.read_csv(tsrc / name)
        df[df[col].isin(names)].to_csv(tdst / name, index=False)
        print(f"  {name}: {len(df[df[col].isin(names)])} rows")
    print(f"sample: {total} records, {len(names)} artists, {len(SAMPLE_SHOWS)} shows -> {dst}")
