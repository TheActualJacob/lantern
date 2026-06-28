# DA3 multi-view raw outputs — session_20260629_000233 (the box)

Produced by one **multi-view** `DepthAnything3.inference(image=[all 12 frames])` pass
(DA3-SMALL). Unlike per-frame mono DA3, these depths share **one coordinate frame** and are
mutually consistent — the poses below are DA3's own estimate, not ARCore's.

## Files
Combined arrays (N=12 views, process-res h=280, w=504):

| file | shape | dtype | meaning |
|---|---|---|---|
| `depth.npy` | (12,280,504) | f32 | per-view depth, DA3 units, multi-view-consistent |
| `conf.npy` | (12,280,504) | f32 | per-view confidence (higher = trust more) |
| `extrinsics_world2cam.npy` | (12,3,4) | f32 | `[R|t]`, OpenCV **world→camera** |
| `intrinsics.npy` | (12,3,3) | f32 | pinhole `K` at process-res (504×280) |
| `object_masks.npy` | (12,280,504) | u8 | 1 = box, aligned to the depth grid |
| `processed_images.npy` | (12,280,504,3) | u8 | RGB exactly as fed to DA3 (letterboxed) |
| `per_view/*_NNNN.npy` | — | — | same data split one-file-per-view |
| `meta.json` | — | — | shapes + conventions |

## Back-project a view to a world point cloud
```python
import numpy as np
depth = np.load("depth.npy"); conf = np.load("conf.npy")
ext = np.load("extrinsics_world2cam.npy"); K = np.load("intrinsics.npy")
masks = np.load("object_masks.npy"); rgb = np.load("processed_images.npy")
N,h,w = depth.shape
uu,vv = np.meshgrid(np.arange(w), np.arange(h))
pts, cols = [], []
for i in range(N):
    Ki,R,t = K[i], ext[i,:,:3], ext[i,:,3]
    m = (masks[i]>0) & (depth[i]>0) & (conf[i] >= np.percentile(conf[i],20))
    z = depth[i][m]; u = uu[m]; v = vv[m]
    Xc = np.stack([(u-Ki[0,2])/Ki[0,0]*z, (v-Ki[1,2])/Ki[1,1]*z, z], 1)
    pts.append((Xc - t) @ R)          # X_world = R^T (X_cam - t)
    cols.append(rgb[i][m]/255.0)
P = np.concatenate(pts); C = np.concatenate(cols)   # box cloud + colors
```

Depth is up-to-scale (DA3 units); multiply by a single global factor if you want metric
(fit against ARCore depth, which lives in the source session under `pulled_sessions/`).
