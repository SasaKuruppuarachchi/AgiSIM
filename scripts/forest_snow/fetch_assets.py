#!/usr/bin/env python3
"""Fetch the CC0 data-only snow maps and HDRI used by FOREST_SNOW.

    python3 -I scripts/forest_snow/fetch_assets.py assets_src/forest_snow

Reuses FOREST_FI's streaming, MD5-verified downloader. Only the explicit image
maps below are selected; no .blend, scripts, archives or model files are fetched.
The original Poly Haven API responses and a complete manifest are kept beside
the assets. The API's ``dimensions`` value for snow_02 is 2000 x 2000 mm (2 m).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "forest_fi"))
from fetch_polyhaven import API, fetch, get_json


LICENSE_URL = "https://polyhaven.com/license"
LICENSE_ID = "CC0-1.0"
TEXTURE_MAPS = (
    ("Diffuse", "jpg"),
    ("nor_gl", "jpg"),
    ("Rough", "jpg"),
    ("AO", "jpg"),
    ("arm", "jpg"),
    ("Displacement", "png"),
)


def _checked_download(entry: dict, suffix: str) -> tuple[str, str, str]:
    """Treat API names/URLs as data; confine downloads to public image assets."""
    url = entry["url"]
    parsed = urlsplit(url)
    name = Path(parsed.path).name
    if (
        parsed.scheme != "https"
        or parsed.netloc != "dl.polyhaven.org"
        or not parsed.path.startswith("/file/ph-assets/")
        or parsed.query
        or parsed.fragment
        or not re.fullmatch(r"[A-Za-z0-9_-]+\." + re.escape(suffix), name)
    ):
        raise ValueError(f"Unexpected Poly Haven image URL: {url!r}")
    md5 = entry["md5"]
    if not re.fullmatch(r"[0-9a-f]{32}", md5):
        raise ValueError(f"Invalid MD5 for {name}")
    return url, name, md5


def fetch_assets(dest: Path, resolution: str = "2k", *, assets=None) -> dict:
    """Download selected maps, verify bytes, and return the saved manifest."""
    dest = Path(dest)
    api_dir = dest / "api"
    api_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for asset, expected_type in (assets if assets is not None else (("snow_02", 1), ("snowy_forest", 0))):
        info = get_json(f"{API}/info/{asset}")
        files = get_json(f"{API}/files/{asset}")
        if info.get("type") != expected_type:
            raise ValueError(f"Unexpected asset type for {asset}: {info.get('type')}")
        for kind, content in (("info", info), ("files", files)):
            (api_dir / f"{kind}_{asset}.json").write_text(
                json.dumps(content, indent=2) + "\n", encoding="utf-8"
            )
        maps = TEXTURE_MAPS if expected_type == 1 else (("hdri", "hdr"),)
        downloaded = []
        for channel, suffix in maps:
            entry = files[channel][resolution][suffix]
            url, name, md5 = _checked_download(entry, suffix)
            target = dest / "polyhaven" / asset / name
            result = fetch(url, str(target), md5)
            size = target.stat().st_size
            if size != entry["size"]:
                raise ValueError(f"Unexpected file size for {name}: {size}")
            downloaded.append({
                "channel": channel,
                "url": url,
                "file": str(target.relative_to(dest)),
                "md5": md5,
                "bytes": size,
            })
            print(asset, name, result, flush=True)
        manifest[asset] = {
            "page": f"https://polyhaven.com/a/{asset}",
            "license": LICENSE_ID,
            "license_url": LICENSE_URL,
            "authors": info["authors"],
            "type": "texture" if expected_type == 1 else "hdri",
            "resolution": resolution,
            "dimensions_mm": info.get("dimensions"),
            "files": downloaded,
        }
    (dest / "polyhaven_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dest", type=Path)
    parser.add_argument("--res", choices=("1k", "2k", "4k"), default="2k")
    args = parser.parse_args()
    manifest = fetch_assets(args.dest, args.res)
    print("done", len(manifest), "CC0 assets")


if __name__ == "__main__":
    main()
