#!/usr/bin/env python3
"""Download the Poly Haven 'pine_forest' assets used by build_forest_fi.py (CC0).

    python3 -I scripts/forest_fi/fetch_polyhaven.py DEST_DIR [--res 2k]

Models: glTF (+ .bin + textures). Textures: jpg maps. Every file is md5-verified against the
Poly Haven API listing, which is saved next to the files (api/). Downloads are data only:
nothing downloaded is executed.
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.request

API = "https://api.polyhaven.com"
UA = {"User-Agent": "AgiSIM-forest_fi-build/1.0"}
MODELS = ["pine_tree_01", "fir_tree_01", "pine_sapling_small", "fir_sapling", "fir_sapling_medium",
          "dead_tree_trunk", "dry_branches_medium_01", "tree_stump_01", "tree_stump_02", "pine_roots",
          "fern_02", "moss_01", "grass_medium_01", "rock_moss_set_01", "rock_moss_set_02"]
TEXTURES = ["forrest_ground_03", "forest_leaves_02", "leafy_grass"]
# HDRI for the dome light: overcast pine forest, so the horizon beyond the 100 m plot reads as forest
HDRIS = ["misty_pines"]
TEX_MAPS = ["Diffuse", "nor_gl", "Rough", "AO", "arm"]
# Alpha masks of the foliage materials (not part of the glTF download, which ships opaque jpgs).
# Used only as data: to test whether foliage is geometry or alpha cards, and to cut cards to shape.
ALPHA = {"pine_tree_01": "twig_alpha", "fir_tree_01": "twig_alpha", "pine_sapling_small": "twig_alpha",
         "fir_sapling": "twigs_alpha", "fir_sapling_medium": "twigs_alpha", "fern_02": "Alpha",
         "grass_medium_01": "Alpha", "moss_01": "Alpha"}


def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)


def fetch(url, dst, md5):
    if os.path.exists(dst):
        h = hashlib.md5(open(dst, "rb").read()).hexdigest()
        if h == md5:
            return "cached"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    h = hashlib.md5()
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            h.update(b)
            f.write(b)
    if h.hexdigest() != md5:
        raise SystemExit(f"md5 mismatch {url}")
    os.replace(tmp, dst)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dest")
    ap.add_argument("--res", default="2k")
    a = ap.parse_args()
    api_dir = os.path.join(a.dest, "api")
    os.makedirs(api_dir, exist_ok=True)
    manifest = {}
    for asset in MODELS + TEXTURES + HDRIS:
        files = get_json(f"{API}/files/{asset}")
        info = get_json(f"{API}/info/{asset}")
        json.dump(files, open(os.path.join(api_dir, f"files_{asset}.json"), "w"), indent=1)
        json.dump(info, open(os.path.join(api_dir, f"info_{asset}.json"), "w"), indent=1)
        out = os.path.join(a.dest, "polyhaven", asset)
        got = []
        if asset in MODELS:
            g = files["gltf"][a.res]["gltf"]
            items = [(g["url"], os.path.join(out, os.path.basename(g["url"])), g["md5"])]
            items += [(v["url"], os.path.join(out, k), v["md5"]) for k, v in g["include"].items()]
        elif asset in HDRIS:
            v = files["hdri"][a.res]["hdr"]
            items = [(v["url"], os.path.join(out, os.path.basename(v["url"])), v["md5"])]
        else:
            items = []
            for m in TEX_MAPS:
                if m in files and a.res in files[m]:
                    v = files[m][a.res]["jpg"] if "jpg" in files[m][a.res] else files[m][a.res]["png"]
                    items.append((v["url"], os.path.join(out, os.path.basename(v["url"])), v["md5"]))
        if asset in ALPHA:
            v = files[ALPHA[asset]]["1k"]
            v = v.get("png") or v.get("jpg")
            items.append((v["url"], os.path.join(out, "alpha", os.path.basename(v["url"])), v["md5"]))
        for url, dst, md5 in items:
            st = fetch(url, dst, md5)
            got.append({"url": url, "file": os.path.relpath(dst, a.dest), "md5": md5})
            print(asset, os.path.basename(dst), st, flush=True)
        manifest[asset] = {"page": f"https://polyhaven.com/a/{asset}", "license": "CC0 1.0",
                           "authors": info.get("authors"), "files": got}
    json.dump(manifest, open(os.path.join(a.dest, "polyhaven_manifest.json"), "w"), indent=1)
    print("done", len(manifest))


if __name__ == "__main__":
    sys.exit(main())
