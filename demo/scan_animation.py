"""
scan_animation.py — "scan growing" point-cloud animation from a DA3 dump (Person D, demo).

Standalone demo asset generator — imports no production/model code. Back-projects a
da3_outputs/<session> multi-view dump into a world cloud, then renders a frame sequence
where the colored points **accumulate view-by-view while the camera orbits** — i.e. it
looks like the object materializing as you scan it. Designed to be NARRATED over (slow,
no captions), for a slide demo video.

USAGE
  python demo/scan_animation.py da3_outputs/session_XXXX --out frames_dir [--frames 90] [--res 1280x720]

Output: frame_0000.png ... in <out>. Stitch to mp4 with ffmpeg (demo/make_demo_video.sh).
Needs: numpy, matplotlib (demo-only deps).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def backproject(d: Path, conf_pct=25):
    depth = np.load(d / "depth.npy"); conf = np.load(d / "conf.npy")
    masks = np.load(d / "object_masks.npy"); K = np.load(d / "intrinsics.npy")
    ext = np.load(d / "extrinsics_world2cam.npy"); rgb = np.load(d / "processed_images.npy")
    N, h, w = depth.shape
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    per_view = []
    for i in range(N):
        Ki, R, t = K[i], ext[i, :, :3], ext[i, :, 3]
        m = (masks[i] > 0) & (depth[i] > 0) & (conf[i] >= np.percentile(conf[i], conf_pct))
        z = depth[i][m]; u = uu[m]; v = vv[m]
        Xc = np.stack([(u - Ki[0, 2]) / Ki[0, 0] * z, (v - Ki[1, 2]) / Ki[1, 1] * z, z], 1)
        P = (Xc - t) @ R
        C = np.clip(rgb[i][m] / 255.0, 0, 1)
        per_view.append((P, C))
    return per_view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--out", default="demo/frames/scan")
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--res", default="1280x720")
    a = ap.parse_args()
    w, h = (int(x) for x in a.res.lower().split("x"))

    per_view = backproject(Path(a.dump))
    N = len(per_view)
    allP = np.concatenate([p for p, _ in per_view])
    cen = np.median(allP, 0)
    # global outlier trim for stable framing
    rad = np.percentile(np.linalg.norm(allP - cen, axis=1), 98)
    lim = rad * 1.1

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    # schedule: first ~70% of frames reveal views progressively; last 30% just orbit the full cloud
    reveal_frames = int(a.frames * 0.7)
    shown_P, shown_C = [], []
    vi = 0
    for f in range(a.frames):
        # add the next view(s) during the reveal phase
        target_views = min(N, int(np.ceil((f + 1) / max(reveal_frames, 1) * N)))
        while vi < target_views:
            shown_P.append(per_view[vi][0]); shown_C.append(per_view[vi][1]); vi += 1
        P = np.concatenate(shown_P); C = np.concatenate(shown_C)

        az = (f / a.frames) * 300.0 - 60.0     # slow orbit
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        fig.patch.set_facecolor("#0C0E14"); ax.set_facecolor("#0C0E14")
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=C, s=2, depthshade=True)
        ax.set_xlim(cen[0]-lim, cen[0]+lim); ax.set_ylim(cen[1]-lim, cen[1]+lim); ax.set_zlim(cen[2]-lim, cen[2]+lim)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=18, azim=az)
        ax.set_axis_off()
        fig.savefig(out / f"frame_{f:04d}.png", dpi=100, facecolor="#0C0E14")
        plt.close(fig)
        if f % 15 == 0:
            print(f"[scan-anim] frame {f}/{a.frames}  views={target_views}/{N}  pts={len(P)}")
    print(f"[scan-anim] done -> {out} ({a.frames} frames)")


if __name__ == "__main__":
    main()
