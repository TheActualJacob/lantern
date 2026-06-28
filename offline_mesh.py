"""Offline reconstruction mirroring the on-device LiveReconstructor (DA3 + locked global scale +
ARCore *pose* TSDF fusion), run on a pulled session so we can inspect the mesh without a re-scan.

Pipeline per session:
  1. DA3 .pte (XNNPACK) -> relative disparity for each color frame.
  2. Object mask from DA3 relative depth (near-band + center-connected component) -- a stand-in
     for the on-device FastSAM mask (we can't run the QNN segmenter on the Mac).
  3. One *global* DA3 disparity->metric affine (s,t): per-frame robust fit vs ARCore depth, fed
     through the same warmup-median / jump-reject / EMA logic as DepthScaleTracker. ARCore depth is
     used ONLY to set this scale, never as fused geometry.
  4. Fuse masked DA3 metric depth into an Open3D TSDF using the ARCore camera pose (the good part),
     then marching cubes -> mesh.
"""
import glob, json, os
import numpy as np
from PIL import Image
import open3d as o3d
import scipy.ndimage as ndi
from executorch.runtime import Runtime

RES = 350
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
FLIP = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenGL(ARCore) camera <-> OpenCV camera

SESSION = os.environ.get("SESSION", "pulled_sessions/session_20260629_000233")
FUSE_W, FUSE_H = 320, 180          # fusion resolution (un-squashes DA3's square output to 16:9)
VOXEL = 0.005                      # 5 mm TSDF voxel
SDF_TRUNC = 0.02
DEPTH_TRUNC = 2.0


# ---- DA3 ----
def da3_runner():
    rt = Runtime.get()
    prog = rt.load_program("da3_small_xnnpack_r350.pte")
    method = prog.load_method("forward")
    import torch

    def run(path):
        im = Image.open(path).convert("RGB").resize((RES, RES), Image.BILINEAR)
        a = (np.asarray(im, np.float32) / 255.0 - MEAN) / STD
        a = a.transpose(2, 0, 1)[None, None]
        out = method.execute([torch.from_numpy(np.ascontiguousarray(a))])[0]
        depth = np.array(out).reshape(RES, RES).astype(np.float32)   # DA3 "depth" (relative)
        disp = np.where(depth > 1e-6, 1.0 / depth, 0.0).astype(np.float32)
        return disp
    return run


def resize(arr, w, h):
    return np.asarray(Image.fromarray(arr).resize((w, h), Image.NEAREST), np.float32)


# ---- object mask: real FastSAM masks (center-point prompt) precomputed into SESSION/masks/ ----
def load_mask(idx, w, h):
    p = f"{SESSION}/masks/mask_{idx:04d}.png"
    m = np.asarray(Image.open(p).convert("L").resize((w, h), Image.NEAREST))
    return m > 127


# ---- robust affine fit (port of AffineScaleSolver.solve: Huber IRLS on s*disp+t ~ 1/depth) ----
def fit_affine(disp160, arcore_m, conf, mask, focus_m):
    x, y = [], []
    h, w = arcore_m.shape
    near, far = focus_m * 0.6, focus_m * 1.4
    for i in range(h):
        for j in range(w):
            dm = arcore_m[i, j]
            if dm <= 0 or dm >= 65 or not np.isfinite(dm):
                continue
            if not mask[i, j]:
                continue
            if dm < near or dm > far:
                continue
            if conf[i, j] < 0.5:
                continue
            xv = disp160[i, j]
            if not np.isfinite(xv):
                continue
            x.append(xv); y.append(1.0 / dm)
    if len(x) < 16:
        return None
    x = np.array(x); y = np.array(y)
    A = np.vstack([x, np.ones_like(x)]).T
    s, t = np.linalg.lstsq(A, y, rcond=None)[0]
    for _ in range(5):  # Huber reweight, delta=0.1
        r = np.abs(s * x + t - y)
        wts = np.where(r <= 0.1, 1.0, 0.1 / np.maximum(r, 1e-9))
        W = np.sqrt(wts)
        s, t = np.linalg.lstsq(A * W[:, None], y * W, rcond=None)[0]
    if not (np.isfinite(s) and np.isfinite(t)):
        return None
    return float(s), float(t)


# ---- DepthScaleTracker port (warmup median, jump reject, EMA, lock) ----
class ScaleTracker:
    WARMUP, MAX_JUMP, ALPHA, LOCK_AFTER = 5, 1.5, 0.25, 20

    def __init__(self):
        self.ws, self.wt, self.s, self.t, self.acc, self.locked = [], [], 0.0, 0.0, 0, False

    def update(self, cand):
        if self.locked or cand is None:
            return
        cs, ct = cand
        if not (np.isfinite(cs) and np.isfinite(ct)) or cs <= 0:
            return
        if self.acc == 0:
            self.ws.append(cs); self.wt.append(ct)
            if len(self.ws) >= self.WARMUP:
                self.s = float(np.median(self.ws)); self.t = float(np.median(self.wt))
                self.acc = len(self.ws)
            return
        if (1 / self.MAX_JUMP) <= cs / self.s <= self.MAX_JUMP:
            self.s = (1 - self.ALPHA) * self.s + self.ALPHA * cs
            self.t = (1 - self.ALPHA) * self.t + self.ALPHA * ct
            self.acc += 1
            if self.acc >= self.LOCK_AFTER:
                self.locked = True


def load_pose(js):
    flat = np.array(js["pose_matrix_column_major"], np.float64)
    c2w_gl = flat.reshape(4, 4).T          # column-major -> 4x4 (camera->world, OpenGL)
    extrinsic = FLIP @ np.linalg.inv(c2w_gl)  # world -> OpenCV camera (Open3D convention)
    return extrinsic


def main():
    cols = sorted(glob.glob(f"{SESSION}/frame_*.png"))
    deps = sorted(glob.glob(f"{SESSION}/depth_*.png"))
    cons = sorted(glob.glob(f"{SESSION}/conf_*.png"))
    jsons = sorted(glob.glob(f"{SESSION}/frame_*.json"))
    run = da3_runner()

    frames = []
    tracker = ScaleTracker()
    for k, (c, dp, cf, jf) in enumerate(zip(cols, deps, cons, jsons), start=1):
        js = json.load(open(jf))
        intr = js["intrinsics"]
        disp = run(c)
        dw, dh = js["depth_width"], js["depth_height"]
        arcore_m = np.array(Image.open(dp), np.float32) / 1000.0     # mm -> m
        conf = np.array(Image.open(cf), np.float32) / 255.0
        disp160 = resize(disp, dw, dh)
        mask160 = load_mask(k, dw, dh)
        valid = arcore_m[(arcore_m > 0) & mask160]
        focus_m = float(np.median(valid)) if valid.size > 8 else None
        cand = fit_affine(disp160, arcore_m, conf, mask160, focus_m) if focus_m else None
        tracker.update(cand)
        frames.append(dict(idx=k, c=c, disp=disp, extr=load_pose(js), intr=intr,
                           cand=cand, focus=focus_m))
        print(f"{os.path.basename(c)} cand={cand} focus={focus_m}")

    print(f"\nGLOBAL SCALE: s={tracker.s:.4f} t={tracker.t:.4f} "
          f"ready={tracker.acc>0} locked={tracker.locked}\n")
    if tracker.acc == 0:
        print("No stable scale could be established."); return
    s, t = tracker.s, tracker.t

    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=VOXEL, sdf_trunc=SDF_TRUNC,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    sx, sy = FUSE_W / 1920.0, FUSE_H / 1080.0
    fused = 0
    for fr in frames:
        intr = fr["intr"]
        o3d_intr = o3d.camera.PinholeCameraIntrinsic(
            FUSE_W, FUSE_H, intr["fx"] * sx, intr["fy"] * sy, intr["cx"] * sx, intr["cy"] * sy)
        # DA3 metric depth at fusion res with the single global scale; masked to the object.
        disp_f = resize(fr["disp"], FUSE_W, FUSE_H)
        md = s * disp_f + t
        depth_m = np.where((md > 1e-4) & np.isfinite(md), 1.0 / md, 0.0).astype(np.float32)
        mask_f = load_mask(fr["idx"], FUSE_W, FUSE_H)
        depth_m[~mask_f] = 0.0
        depth_m[depth_m > DEPTH_TRUNC] = 0.0
        color = np.asarray(Image.open(fr["c"]).convert("RGB").resize((FUSE_W, FUSE_H)), np.uint8)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(color)),
            o3d.geometry.Image(np.ascontiguousarray(depth_m)),
            depth_scale=1.0, depth_trunc=DEPTH_TRUNC, convert_rgb_to_intensity=False)
        vol.integrate(rgbd, o3d_intr, fr["extr"])
        fused += 1

    mesh = vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(f"{SESSION}/offline_mesh.ply", mesh)
    o3d.io.write_triangle_mesh(f"{SESSION}/offline_mesh.obj", mesh)

    v = np.asarray(mesh.vertices)
    print(f"fused {fused} frames -> mesh: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")
    if len(v):
        ext = v.max(0) - v.min(0)
        print(f"bbox (cm): {ext[0]*100:.1f} x {ext[1]*100:.1f} x {ext[2]*100:.1f}")

    # also a fused point cloud (more forgiving to view than a coarse mesh)
    pcd = vol.extract_point_cloud()
    o3d.io.write_point_cloud(f"{SESSION}/offline_cloud.ply", pcd)
    print(f"cloud: {len(pcd.points)} points -> {SESSION}/offline_cloud.ply")


if __name__ == "__main__":
    main()
