#!/usr/bin/env python3
"""Compose a --record-video directory into a 1920x1080 H.264 MP4 (no Isaac imports).

    python3 agipix_video_compose.py VIDEO_DIR [--out flight.mp4] [--fps 30] [--title TEXT]

Layout: chase view 1280x720 top left; FL/FR (front pair) stacked at the right; RL and RR
at the bottom corners; the stitched 360 degree panorama and flight readout between them.
Frames are recorded at 10 Hz of simulation time, so --fps 30 plays at 3x speed.
Needs numpy, OpenCV and an ffmpeg with libx264 on PATH.
"""

import argparse
import csv
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

W, H = 1920, 1080
TILE = (576, 360)
BG = (18, 18, 20)
FONT = cv2.FONT_HERSHEY_DUPLEX


def text(img, s, org, scale=0.6, colour=(235, 235, 235), thickness=1, box=True):
    (tw, th), base = cv2.getTextSize(s, FONT, scale, thickness)
    x, y = org
    if box:
        overlay = img[y - th - 8:y + base + 4, x - 6:x + tw + 6]
        overlay[:] = (overlay * 0.45).astype(np.uint8)
    cv2.putText(img, s, (x, y), FONT, scale, colour, thickness, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video_dir", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--title", default="AgiPix autonomous exploration")
    ap.add_argument("--no-trim", action="store_true",
                    help="keep idle frames (default: start 3 s before the first motion, end 5 s after the last)")
    args = ap.parse_args()
    d = args.video_dir
    meta = json.loads((d / "meta.json").read_text())
    rows = {}
    with open(d / "frames.csv") as f:
        for r in csv.DictReader(f):
            if r["x"]:
                rows[int(r["stamp_ns"])] = dict(position=[float(r[k]) for k in "xyz"],
                                                velocity=[float(r[k]) for k in ("vx", "vy", "vz")])
    stamps = sorted(int(p.stem) for p in (d / "chase").glob("*.jpg"))
    if not stamps:
        raise SystemExit("no frames in " + str(d / "chase"))
    moving = [s for s in stamps if "velocity" in rows.get(s, {}) and np.linalg.norm(rows[s]["velocity"]) > 0.15]
    if moving and not args.no_trim:
        stamps = [s for s in stamps if moving[0] - 3e9 <= s <= moving[-1] + 5e9]
    rate = meta.get("rate_hz", 10.0)
    speedup = args.fps / rate
    out = args.out or d / "flight.mp4"
    mounts = meta["mounts"]
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf),
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    t0 = stamps[0]
    distance, last = 0.0, None
    slots = {"FL": (1280 + 32, 0), "FR": (1280 + 32, 360), "RL": (0, 720), "RR": (W - TILE[0], 720)}
    for i, stamp in enumerate(stamps):
        name = f"{stamp:019d}.jpg"
        frame = np.full((H, W, 3), BG, np.uint8)
        frame[:720, :1280] = cv2.imread(str(d / "chase" / name))
        for cam, (x, y) in slots.items():
            tile = cv2.imread(str(d / "cams" / cam / name))
            if tile is not None:
                frame[y:y + TILE[1], x:x + TILE[0]] = cv2.resize(tile, TILE, interpolation=cv2.INTER_AREA)
            m = mounts[cam]
            text(frame, f"{cam}  yaw {m['yaw_deg']:+.0f}  pitch {m['pitch_deg']:+.0f}", (x + 10, y + 26), 0.55)
        pano = cv2.imread(str(d / "pano" / name))
        if pano is not None:
            frame[720 + 84:720 + 84 + 192, 576:576 + 768] = cv2.resize(pano, (768, 192), interpolation=cv2.INTER_AREA)
        row = rows.get(stamp, {})
        pos, vel = row.get("position"), row.get("velocity")
        if pos is not None and last is not None:
            distance += float(np.linalg.norm(np.subtract(pos, last)))
        last = pos if pos is not None else last
        t = (stamp - t0) / 1e9
        text(frame, args.title, (20, 40), 0.85, thickness=1)
        text(frame, f"{meta.get('environment', '')}   simulation, {speedup:g}x speed", (20, 72), 0.6)
        readout = f"t {t:6.1f} s"
        if vel is not None:
            readout += f"   speed {np.linalg.norm(vel[:2]):4.1f} m/s"
        readout += f"   flown {distance:6.1f} m"
        text(frame, readout, (576 + 12, 720 + 40), 0.62, box=False)
        text(frame, "360 deg panorama from the four fisheyes (body frame, forward centred)",
             (576 + 12, 720 + 84 + 192 + 28), 0.45, colour=(170, 170, 170), box=False)
        text(frame, "chase camera", (20, 700), 0.5, colour=(200, 200, 200))
        ffmpeg.stdin.write(frame.tobytes())
        if i % 500 == 0:
            print(f"{i}/{len(stamps)}", flush=True)
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {out} ({len(stamps)} frames, {len(stamps) / args.fps:.1f} s at {args.fps:g} fps, {speedup:g}x)")


if __name__ == "__main__":
    main()
