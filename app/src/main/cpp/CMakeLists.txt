# Native QNN depth bridge (libqnn_depth.so). Built only when app/build.gradle.kts enables the
# externalNativeBuild (i.e. when QNN_SDK_ROOT is provided). See recon/QnnDlcDepthModel.kt.
cmake_minimum_required(VERSION 3.22.1)
project(qnn_depth LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(qnn_depth SHARED qnn_depth.cpp)

# QNN_SDK_ROOT is passed from Gradle (-DQNN_SDK_ROOT=...). When present we compile the real QNN
# implementation against its headers; otherwise the file builds as a stub that always fails init.
if(DEFINED QNN_SDK_ROOT AND EXISTS "${QNN_SDK_ROOT}/include/QNN")
    message(STATUS "Building qnn_depth WITH QNN SDK at ${QNN_SDK_ROOT}")
    target_compile_definitions(qnn_depth PRIVATE LANTERN_QNN_ENABLED=1)
    target_include_directories(qnn_depth PRIVATE "${QNN_SDK_ROOT}/include/QNN")
else()
    message(WARNING "QNN_SDK_ROOT not set / headers missing; building qnn_depth as a STUB")
endif()

# We dlopen libQnnHtp.so / libQnnSystem.so at runtime (bundled in jniLibs), so no link-time
# dependency on them — only the Android system libs.
find_library(log-lib log)
target_link_libraries(qnn_depth ${log-lib} android dl)
