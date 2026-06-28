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

// Function-pointer getters exported by the QNN backend / system shared libs.
typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn)(const QnnInterface_t***, uint32_t*);
typedef Qnn_ErrorHandle_t (*QnnSystemInterfaceGetProvidersFn)(const QnnSystemInterface_t***, uint32_t*);

// Everything we need to keep alive for the lifetime of one loaded model.
struct QnnDepthContext {
    void* backendLibHandle = nullptr;   // libQnnHtp.so
    void* systemLibHandle = nullptr;    // libQnnSystem.so

    QNN_INTERFACE_VER_TYPE qnn{};       // core API function table
    QNN_SYSTEM_INTERFACE_VER_TYPE sys{}; // system API function table

    Qnn_BackendHandle_t backend = nullptr;
    Qnn_DeviceHandle_t device = nullptr;
    Qnn_ContextHandle_t context = nullptr;
    Qnn_GraphHandle_t graph = nullptr;

    // Tensor templates queried from the binary (shape/ids), filled with client buffers per call.
    Qnn_Tensor_t inputTensor{};
    Qnn_Tensor_t outputTensor{};

    int res = 0;
    size_t inputElems = 0;   // res*res*3
    size_t outputElems = 0;  // res*res
};

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
    } else {
        LOGE("unsupported binary-info version %d", (int)binaryInfo->version);
        c->sys.systemContextFree(sysCtx);
        return false;
    }
    if (numGraphs == 0 || !graphs) { LOGE("binary has no graphs"); c->sys.systemContextFree(sysCtx); return false; }
    *graphInfoOut = &graphs[0];
    // NOTE: graphInfo points into sysCtx-owned memory; caller must copy what it needs before
    // freeing. We free in initFromBinary after copying tensor templates.
    return true; // sysCtx intentionally leaked to caller scope; freed by caller.
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

        // Backend + device bring-up (HTP).
        if (c->qnn.backendCreate(/*logger*/ nullptr, /*config*/ nullptr, &c->backend) != QNN_SUCCESS) {
            LOGE("backendCreate failed"); break;
        }
        if (c->qnn.deviceCreate(/*logger*/ nullptr, /*config*/ nullptr, &c->device) != QNN_SUCCESS) {
            LOGW("deviceCreate failed; continuing with default device");
            c->device = nullptr;
        }

        // Read graph metadata, then create the context directly from the binary.
        const QnnSystemContext_GraphInfo_t* graphInfo = nullptr;
        if (!inspectBinary(c, blob, &graphInfo)) break;

        if (c->qnn.contextCreateFromBinary(
                c->backend, c->device, /*config*/ nullptr,
                blob.data(), blob.size(), &c->context, /*profile*/ nullptr) != QNN_SUCCESS) {
            LOGE("contextCreateFromBinary failed"); break;
        }

        // Retrieve the graph by the name embedded in the binary and copy its IO tensor templates.
        const char* graphName = (graphInfo->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1)
            ? graphInfo->graphInfoV1.graphName : graphInfo->graphInfoV1.graphName;
        if (c->qnn.graphRetrieve(c->context, graphName, &c->graph) != QNN_SUCCESS) {
            LOGE("graphRetrieve('%s') failed", graphName ? graphName : "?"); break;
        }
        // DA3 is single-in/single-out; copy the descriptors so we can attach client buffers.
        if (graphInfo->graphInfoV1.numGraphInputs >= 1 && graphInfo->graphInfoV1.numGraphOutputs >= 1) {
            c->inputTensor = graphInfo->graphInfoV1.graphInputs[0];
            c->outputTensor = graphInfo->graphInfoV1.graphOutputs[0];
        } else {
            LOGE("graph missing IO tensors"); break;
        }
        ok = true;
    } while (false);

    env->ReleaseStringUTFChars(jModelPath, modelPath);
    if (!ok) {
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
    in.v1.memType = QNN_TENSORMEMTYPE_RAW;
    in.v1.clientBuf = { input, static_cast<uint32_t>(c->inputElems * sizeof(float)) };
    out.v1.memType = QNN_TENSORMEMTYPE_RAW;
    out.v1.clientBuf = { output, static_cast<uint32_t>(c->outputElems * sizeof(float)) };

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
    if (c->context) c->qnn.contextFree(c->context, nullptr);
    if (c->device) c->qnn.deviceFree(c->device);
    if (c->backend) c->qnn.backendFree(c->backend);
    if (c->backendLibHandle) dlclose(c->backendLibHandle);
    if (c->systemLibHandle) dlclose(c->systemLibHandle);
    delete c;
    LOGI("QNN context closed");
}

#endif // LANTERN_QNN_ENABLED
