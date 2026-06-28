// JNI bridge: runs the AI Hub Depth-Anything-3 `.dlc` / HTP context-binary natively on the
// Qualcomm QNN/QAIRT runtime (Hexagon NPU). Backs `recon.QnnDlcDepthModel` (Kotlin).
//
// Build: compiled into `libqnn_depth.so` only when the project is built with `QNN_SDK_ROOT`
// set (see app/build.gradle.kts + CMakeLists.txt). On a stock build this lib is absent and the
// Kotlin side ([QnnNative.available] == false) degrades to ExecuTorch/ARCore depth.
//
// Contract (must match the DLC metadata.json):
//   input  "image":           NHWC [1,518,518,3] float32, range [0,1]
//   output "depth_estimates":      [1,518,518,1] float32 monocular depth
//
// Approach: prefer a pre-compiled HTP **context binary** (`.bin`) loaded via the QNN System API
// + QnnContext_createFromBinary (fast init, no on-device prepare). The graph + tensor metadata
// are read back from the binary; we wire the single input/output tensors to client buffers and
// call QnnGraph_execute once per frame.
//
// This file is structured to compile against QAIRT >= 2.45. The QNN call sequence below mirrors
// the official qnn-sample-app (context-binary path); it requires a device + the matching
// Hexagon-v79 skel to actually execute and has not been run on this build host.

#include <jni.h>
#include <android/log.h>
#include <dlfcn.h>
#include <cstdarg>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

#define LOG_TAG "LANTERN_QNN"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

#ifndef LANTERN_QNN_ENABLED
// --------------------------------------------------------------------------------------------
// Stub build: no QNN SDK available at compile time. The lib still loads, but init always fails
// so Kotlin falls back. Lets the JNI plumbing be validated without the SDK (NDK-only builds).
// --------------------------------------------------------------------------------------------
extern "C" JNIEXPORT jlong JNICALL
Java_com_lantern_recorder_recon_QnnNative_nativeInit(JNIEnv*, jobject, jstring, jint) {
    LOGW("qnn_depth built without QNN SDK (LANTERN_QNN_ENABLED unset); init returns 0");
    return 0;
}
extern "C" JNIEXPORT jboolean JNICALL
Java_com_lantern_recorder_recon_QnnNative_nativeInfer(JNIEnv*, jobject, jlong, jfloatArray, jfloatArray) {
    return JNI_FALSE;
}
extern "C" JNIEXPORT void JNICALL
Java_com_lantern_recorder_recon_QnnNative_nativeClose(JNIEnv*, jobject, jlong) {}

#else
// --------------------------------------------------------------------------------------------
// Real build: QNN SDK headers on the include path (QNN_SDK_ROOT/include/QNN).
// --------------------------------------------------------------------------------------------
#include "QnnInterface.h"
#include "QnnBackend.h"
#include "QnnContext.h"
#include "QnnGraph.h"
#include "QnnTensor.h"
#include "QnnTypes.h"
#include "QnnDevice.h"
#include "System/QnnSystemInterface.h"
#include "System/QnnSystemContext.h"

namespace {

// Forwards QNN's internal backend logs to logcat so device/context failures show their real
// cause (e.g. cDSP open failure, arch mismatch) instead of just a numeric error handle.
void qnnLogCallback(const char* fmt, QnnLog_Level_t level, uint64_t /*ts*/, va_list args) {
    char line[1024];
    vsnprintf(line, sizeof(line), fmt, args);
    int prio = (level == QNN_LOG_LEVEL_ERROR) ? ANDROID_LOG_ERROR
             : (level == QNN_LOG_LEVEL_WARN)  ? ANDROID_LOG_WARN
                                              : ANDROID_LOG_INFO;
    __android_log_print(prio, "LANTERN_QNN", "[qnn] %s", line);
}

// Function-pointer getters exported by the QNN backend / system shared libs.
typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn)(const QnnInterface_t***, uint32_t*);
typedef Qnn_ErrorHandle_t (*QnnSystemInterfaceGetProvidersFn)(const QnnSystemInterface_t***, uint32_t*);

// Everything we need to keep alive for the lifetime of one loaded model.
struct QnnDepthContext {
    void* backendLibHandle = nullptr;   // libQnnHtp.so
    void* systemLibHandle = nullptr;    // libQnnSystem.so

    QNN_INTERFACE_VER_TYPE qnn{};       // core API function table
    QNN_SYSTEM_INTERFACE_VER_TYPE sys{}; // system API function table

    Qnn_LogHandle_t logger = nullptr;
    Qnn_BackendHandle_t backend = nullptr;
    Qnn_DeviceHandle_t device = nullptr;
    Qnn_ContextHandle_t context = nullptr;
    Qnn_GraphHandle_t graph = nullptr;

    // Kept alive for the model's lifetime: the queried tensor templates below point into its
    // memory (dims/ids/name), so it must outlive inference and is freed in nativeClose.
    QnnSystemContext_Handle_t sysCtxHandle = nullptr;

    // Tensor templates queried from the binary (shape/ids), filled with client buffers per call.
    Qnn_Tensor_t inputTensor{};
    Qnn_Tensor_t outputTensor{};

    int res = 0;
    size_t inputElems = 0;   // res*res*3
    size_t outputElems = 0;  // res*res
};

// The graph IO fields shared by GraphInfo V1/V2/V3 (same leading layout, different versions).
struct GraphIO {
    const char* name = nullptr;
    Qnn_Tensor_t* inputs = nullptr;
    uint32_t numInputs = 0;
    Qnn_Tensor_t* outputs = nullptr;
    uint32_t numOutputs = 0;
};

// Pull name + IO tensors out of a graph info record regardless of its struct version.
bool extractGraphIO(const QnnSystemContext_GraphInfo_t* g, GraphIO* io) {
    switch (g->version) {
        case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1:
            io->name = g->graphInfoV1.graphName;
            io->inputs = g->graphInfoV1.graphInputs;   io->numInputs = g->graphInfoV1.numGraphInputs;
            io->outputs = g->graphInfoV1.graphOutputs; io->numOutputs = g->graphInfoV1.numGraphOutputs;
            return true;
        case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2:
            io->name = g->graphInfoV2.graphName;
            io->inputs = g->graphInfoV2.graphInputs;   io->numInputs = g->graphInfoV2.numGraphInputs;
            io->outputs = g->graphInfoV2.graphOutputs; io->numOutputs = g->graphInfoV2.numGraphOutputs;
            return true;
        case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3:
            io->name = g->graphInfoV3.graphName;
            io->inputs = g->graphInfoV3.graphInputs;   io->numInputs = g->graphInfoV3.numGraphInputs;
            io->outputs = g->graphInfoV3.graphOutputs; io->numOutputs = g->graphInfoV3.numGraphOutputs;
            return true;
        default:
            LOGE("unsupported graph-info version %d", (int)g->version);
            return false;
    }
}

// Point a tensor (v1 or v2) at a CPU client buffer, version-safely.
void setRawClientBuf(Qnn_Tensor_t* t, void* data, uint32_t size) {
    if (t->version == QNN_TENSOR_VERSION_2) {
        t->v2.memType = QNN_TENSORMEMTYPE_RAW;
        t->v2.clientBuf = { data, size };
    } else {
        t->v1.memType = QNN_TENSORMEMTYPE_RAW;
        t->v1.clientBuf = { data, size };
    }
}

// Resolve QnnInterface_getProviders from a dlopen'd backend lib and pick the first provider.
bool loadCoreInterface(QnnDepthContext* c, const char* lib) {
    c->backendLibHandle = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (!c->backendLibHandle) { LOGE("dlopen %s failed: %s", lib, dlerror()); return false; }
    auto getProviders = reinterpret_cast<QnnInterfaceGetProvidersFn>(
        dlsym(c->backendLibHandle, "QnnInterface_getProviders"));
    if (!getProviders) { LOGE("QnnInterface_getProviders missing in %s", lib); return false; }

    const QnnInterface_t** providers = nullptr;
    uint32_t count = 0;
    if (getProviders(&providers, &count) != QNN_SUCCESS || count == 0 || !providers) {
        LOGE("QnnInterface_getProviders returned no providers"); return false;
    }
    c->qnn = providers[0]->QNN_INTERFACE_VER_NAME;
    return true;
}

bool loadSystemInterface(QnnDepthContext* c, const char* lib) {
    c->systemLibHandle = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (!c->systemLibHandle) { LOGE("dlopen %s failed: %s", lib, dlerror()); return false; }
    auto getProviders = reinterpret_cast<QnnSystemInterfaceGetProvidersFn>(
        dlsym(c->systemLibHandle, "QnnSystemInterface_getProviders"));
    if (!getProviders) { LOGE("QnnSystemInterface_getProviders missing in %s", lib); return false; }

    const QnnSystemInterface_t** providers = nullptr;
    uint32_t count = 0;
    if (getProviders(&providers, &count) != QNN_SUCCESS || count == 0 || !providers) {
        LOGE("QnnSystemInterface_getProviders returned no providers"); return false;
    }
    c->sys = providers[0]->QNN_SYSTEM_INTERFACE_VER_NAME;
    return true;
}

std::vector<uint8_t> readFile(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    std::streamsize n = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(static_cast<size_t>(n));
    if (!f.read(reinterpret_cast<char*>(buf.data()), n)) return {};
    return buf;
}

// Pull the first graph's name + IO tensor descriptors out of a context-binary blob.
bool inspectBinary(QnnDepthContext* c, const std::vector<uint8_t>& blob,
                   const QnnSystemContext_GraphInfo_t** graphInfoOut) {
    QnnSystemContext_Handle_t sysCtx = nullptr;
    if (c->sys.systemContextCreate(&sysCtx) != QNN_SUCCESS || !sysCtx) {
        LOGE("systemContextCreate failed"); return false;
    }
    const QnnSystemContext_BinaryInfo_t* binaryInfo = nullptr;
    Qnn_ContextBinarySize_t binaryInfoSize = 0;
    if (c->sys.systemContextGetBinaryInfo(
            sysCtx, const_cast<uint8_t*>(blob.data()), blob.size(),
            &binaryInfo, &binaryInfoSize) != QNN_SUCCESS || !binaryInfo) {
        LOGE("systemContextGetBinaryInfo failed");
        c->sys.systemContextFree(sysCtx);
        return false;
    }
    // V1/V2/V3 binary-info variants all expose graphs[]; DA3 export is a single graph.
    const QnnSystemContext_GraphInfo_t* graphs = nullptr;
    uint32_t numGraphs = 0;
    if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
        graphs = binaryInfo->contextBinaryInfoV1.graphs;
        numGraphs = binaryInfo->contextBinaryInfoV1.numGraphs;
    } else if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
        graphs = binaryInfo->contextBinaryInfoV2.graphs;
        numGraphs = binaryInfo->contextBinaryInfoV2.numGraphs;
    } else if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
        graphs = binaryInfo->contextBinaryInfoV3.graphs;
        numGraphs = binaryInfo->contextBinaryInfoV3.numGraphs;
    } else {
        LOGE("unsupported binary-info version %d", (int)binaryInfo->version);
        c->sys.systemContextFree(sysCtx);
        return false;
    }
    if (numGraphs == 0 || !graphs) { LOGE("binary has no graphs"); c->sys.systemContextFree(sysCtx); return false; }
    *graphInfoOut = &graphs[0];
    // graphInfo (and the tensor templates it points to) lives in sysCtx-owned memory; keep the
    // handle alive on the context and free it in nativeClose, after inference is done.
    c->sysCtxHandle = sysCtx;
    return true;
}

} // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_lantern_recorder_recon_QnnNative_nativeInit(JNIEnv* env, jobject, jstring jModelPath, jint jRes) {
    const char* modelPath = env->GetStringUTFChars(jModelPath, nullptr);
    LOGI("QNN init: %s (res=%d)", modelPath, jRes);

    auto* c = new QnnDepthContext();
    c->res = jRes;
    c->inputElems = static_cast<size_t>(jRes) * jRes * 3;
    c->outputElems = static_cast<size_t>(jRes) * jRes;

    bool ok = false;
    do {
        if (!loadCoreInterface(c, "libQnnHtp.so")) break;
        if (!loadSystemInterface(c, "libQnnSystem.so")) break;

        std::vector<uint8_t> blob = readFile(modelPath);
        if (blob.empty()) { LOGE("model file empty/unreadable: %s", modelPath); break; }

        // Backend + device bring-up (HTP). Create a logger first so QNN's own diagnostics
        // (the real reason for any device/context failure) reach logcat.
        Qnn_ErrorHandle_t e;
        if (c->qnn.logCreate(qnnLogCallback, QNN_LOG_LEVEL_VERBOSE, &c->logger) != QNN_SUCCESS) {
            LOGW("logCreate failed; QNN backend logs won't be available");
            c->logger = nullptr;
        }
        if ((e = c->qnn.backendCreate(c->logger, /*config*/ nullptr, &c->backend)) != QNN_SUCCESS) {
            LOGE("backendCreate failed: 0x%llx", (unsigned long long)e); break;
        }
        if ((e = c->qnn.deviceCreate(c->logger, /*config*/ nullptr, &c->device)) != QNN_SUCCESS) {
            LOGW("deviceCreate failed (0x%llx); continuing with default device", (unsigned long long)e);
            c->device = nullptr;
        }

        // Read graph metadata, then create the context directly from the binary.
        const QnnSystemContext_GraphInfo_t* graphInfo = nullptr;
        if (!inspectBinary(c, blob, &graphInfo)) break;

        if ((e = c->qnn.contextCreateFromBinary(
                c->backend, c->device, /*config*/ nullptr,
                blob.data(), blob.size(), &c->context, /*profile*/ nullptr)) != QNN_SUCCESS) {
            LOGE("contextCreateFromBinary failed: 0x%llx", (unsigned long long)e); break;
        }

        // Read the graph's name + IO tensor templates (version-safe across GraphInfo V1/V2/V3).
        GraphIO io;
        if (!extractGraphIO(graphInfo, &io)) break;
        if (c->qnn.graphRetrieve(c->context, io.name, &c->graph) != QNN_SUCCESS) {
            LOGE("graphRetrieve('%s') failed", io.name ? io.name : "?"); break;
        }
        // DA3 is single-in/single-out; copy the descriptors so we can attach client buffers.
        if (io.numInputs >= 1 && io.numOutputs >= 1) {
            c->inputTensor = io.inputs[0];
            c->outputTensor = io.outputs[0];
        } else {
            LOGE("graph missing IO tensors"); break;
        }
        ok = true;
    } while (false);

    env->ReleaseStringUTFChars(jModelPath, modelPath);
    if (!ok) {
        if (c->sysCtxHandle) c->sys.systemContextFree(c->sysCtxHandle);
        if (c->context) c->qnn.contextFree(c->context, nullptr);
        if (c->backend) c->qnn.backendFree(c->backend);
        if (c->backendLibHandle) dlclose(c->backendLibHandle);
        if (c->systemLibHandle) dlclose(c->systemLibHandle);
        delete c;
        return 0;
    }
    LOGI("QNN init OK");
    return reinterpret_cast<jlong>(c);
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_lantern_recorder_recon_QnnNative_nativeInfer(
        JNIEnv* env, jobject, jlong handle, jfloatArray jInput, jfloatArray jOutput) {
    auto* c = reinterpret_cast<QnnDepthContext*>(handle);
    if (!c || !c->graph) return JNI_FALSE;

    const jsize inN = env->GetArrayLength(jInput);
    const jsize outN = env->GetArrayLength(jOutput);
    if ((size_t)inN < c->inputElems || (size_t)outN < c->outputElems) {
        LOGE("infer buffer size mismatch (in=%d out=%d)", inN, outN);
        return JNI_FALSE;
    }

    jfloat* input = env->GetFloatArrayElements(jInput, nullptr);
    jfloat* output = env->GetFloatArrayElements(jOutput, nullptr);

    // Point the (copied) tensor descriptors at our client buffers (CPU memory, RAW layout).
    Qnn_Tensor_t in = c->inputTensor;
    Qnn_Tensor_t out = c->outputTensor;
    setRawClientBuf(&in, input, static_cast<uint32_t>(c->inputElems * sizeof(float)));
    setRawClientBuf(&out, output, static_cast<uint32_t>(c->outputElems * sizeof(float)));

    Qnn_ErrorHandle_t err = c->qnn.graphExecute(
        c->graph, &in, 1, &out, 1, /*profile*/ nullptr, /*signal*/ nullptr);

    bool ok = (err == QNN_SUCCESS);
    if (!ok) LOGE("graphExecute failed: %lld", (long long)err);

    env->ReleaseFloatArrayElements(jInput, input, JNI_ABORT);          // input not modified
    env->ReleaseFloatArrayElements(jOutput, output, ok ? 0 : JNI_ABORT); // commit on success
    return ok ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_lantern_recorder_recon_QnnNative_nativeClose(JNIEnv*, jobject, jlong handle) {
    auto* c = reinterpret_cast<QnnDepthContext*>(handle);
    if (!c) return;
    if (c->sysCtxHandle) c->sys.systemContextFree(c->sysCtxHandle);
    if (c->context) c->qnn.contextFree(c->context, nullptr);
    if (c->device) c->qnn.deviceFree(c->device);
    if (c->backend) c->qnn.backendFree(c->backend);
    if (c->backendLibHandle) dlclose(c->backendLibHandle);
    if (c->systemLibHandle) dlclose(c->systemLibHandle);
    delete c;
    LOGI("QNN context closed");
}

#endif // LANTERN_QNN_ENABLED
