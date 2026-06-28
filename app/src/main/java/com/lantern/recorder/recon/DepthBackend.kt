package com.lantern.recorder.recon

/**
 * Which monocular-depth runtime is currently feeding the live mesh. Surfaced to the UI readout
 * ([com.lantern.recorder.ui.CaptureUiState.liveMeshDepthBackend]) so the user can see whether
 * dense depth is coming from the Hexagon NPU (QNN), ExecuTorch CPU/NPU, or ARCore alone.
 */
enum class DepthBackendKind { ARCORE, EXECUTORCH, QNN }

/**
 * A swappable monocular-depth runtime. Produces a relative (affine-invariant) [DisparityMap]
 * from a color frame; everything downstream — [AffineScaleSolver] (scale vs ARCore),
 * [TsdfVolume] (fusion), [MarchingCubes] (mesh) — is backend-agnostic, so the depth source can
 * be swapped without touching the reconstruction pipeline.
 *
 * Implementations:
 *  - [ExecuTorchDepthModel]: Depth-Anything-3 `.pte` via the ExecuTorch runtime (XNNPACK CPU or
 *    QNN delegate). Input contract: NCHW, ImageNet-normalized RGB.
 *  - [QnnDlcDepthModel]: the AI Hub `depth_anything_v3.dlc`/context-binary run natively on the
 *    Qualcomm QNN/QAIRT runtime via JNI. Input contract: NHWC, `[0,1]` (no ImageNet norm).
 *
 * Each backend owns its own preprocessing because the two runtimes expect different input
 * layouts and normalization; the *output* is unified to relative disparity (`1 / depth`).
 */
interface DepthBackend {
    /** Identifies the runtime for the UI/logs. */
    val kind: DepthBackendKind

    /** Working square resolution the network expects (e.g. 518 for DA3-Small). */
    val res: Int

    /**
     * Run depth on a packed-ARGB color image; returns relative disparity at [res]x[res], or
     * null on failure so the caller can fall back to ARCore-only depth instead of crashing.
     * CPU/NPU heavy — call off the GL thread.
     */
    fun inferDisparity(argb: ImageUtils.Argb): DisparityMap?

    /** Release native resources (module/QNN context). Safe to call once. */
    fun close()
}
