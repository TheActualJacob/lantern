"""Quick quality report for a pulled Lantern Recorder session.

Reports per-frame depth coverage / range, camera motion between frames, and writes a
contact-sheet PNG (RGB on top, depth heatmap below) for eyeballing what was captured.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def load_depth_mm(path: Path) -> np.ndarray:
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(path)
    return d.astype(np.float32)  # millimeters, 0 = invalid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    sess = args.session
    out = args.out or (sess / "analysis_contact_sheet.png")

    cap = sess / "capture.json"
    if cap.exists():
        print("capture.json:", cap.read_text().strip())

    jsons = sorted(sess.glob("frame_*.json"))
    print(f"\n{len(jsons)} frames\n")
    print(f"{'frame':>6} {'valid%':>7} {'median_m':>9} {'min_m':>6} {'max_m':>6} {'cam_move_cm':>11}")

    prev_t = None
    tiles = []
    for jp in jsons:
        meta = json.loads(jp.read_text())
        idx = meta["frame_index"]
        dp = sess / f"depth_{idx:04d}.png"
        depth = load_depth_mm(dp) / 1000.0  # -> meters
        valid = depth[(depth > 0) & (depth < 5)]
        cov = 100.0 * valid.size / depth.size
        med = float(np.median(valid)) if valid.size else float("nan")
        dmin = float(valid.min()) if valid.size else float("nan")
        dmax = float(valid.max()) if valid.size else float("nan")

        t = np.array(meta["translation_xyz"], dtype=np.float64)
        move = "-" if prev_t is None else f"{100*np.linalg.norm(t-prev_t):.2f}"
        prev_t = t
        print(f"{idx:>6} {cov:>7.1f} {med:>9.3f} {dmin:>6.2f} {dmax:>6.2f} {move:>11}")

        # Build a tile: RGB (resized) over depth heatmap.
        rgb = cv2.imread(str(sess / f"frame_{idx:04d}.png"), cv2.IMREAD_COLOR)
        tw = 320
        th = int(rgb.shape[0] * tw / rgb.shape[1])
        rgb_small = cv2.resize(rgb, (tw, th))
        dn = depth.copy()
        dn[(dn <= 0) | (dn >= 5)] = np.nan
        lo, hi = np.nanpercentile(dn, [5, 95]) if np.isfinite(dn).any() else (0, 1)
        norm = np.clip((dn - lo) / max(hi - lo, 1e-6), 0, 1)
        norm = np.nan_to_num(norm, nan=0.0)
        heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        heat[np.isnan(dn)] = (0, 0, 0)
        heat = cv2.resize(heat, (tw, th), interpolation=cv2.INTER_NEAREST)
        cv2.putText(rgb_small, f"#{idx} {cov:.0f}% {med:.2f}m", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        tiles.append(np.vstack([rgb_small, heat]))

    if tiles:
        cols = min(5, len(tiles))
        rows = (len(tiles) + cols - 1) // cols
        h, w = tiles[0].shape[:2]
        sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
        for i, t in enumerate(tiles):
            r, c = divmod(i, cols)
            sheet[r*h:(r+1)*h, c*w:(c+1)*w] = t
        cv2.imwrite(str(out), sheet)
        print(f"\nWrote contact sheet -> {out}")


if __name__ == "__main__":
    main()
