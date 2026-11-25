# Builds libllmbridge.so: the NAPI module that hosts the MNN LLM runtime.
#
# MNN itself is prebuilt for OHOS arm64 (see README) and shipped as
# libs/arm64-v8a/libMNN.so, which DevEco packages into the HAP automatically.
# Linking against the prebuilt keeps the app build fast while the engine stays
# a plain SHARED library.

cmake_minimum_required(VERSION 3.5.0)
project(llmbridge)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED OHOS_ARCH)
    set(OHOS_ARCH arm64-v8a)
endif()

set(MNN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/mnn")
set(MNN_LIB  "${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/${OHOS_ARCH}/libMNN.so")

if(NOT EXISTS "${MNN_LIB}")
    message(FATAL_ERROR "libMNN.so not found at ${MNN_LIB}. Run scripts/build_mnn_ohos.sh first.")
endif()

add_library(llmbridge SHARED
    napi_init.cpp
    llm_engine.cpp)

target_include_directories(llmbridge PRIVATE
    ${MNN_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR})

# Prebuilt MNN runtime, resolved at runtime from the HAP's lib/ directory.
add_library(mnn SHARED IMPORTED)
set_target_properties(mnn PROPERTIES IMPORTED_LOCATION "${MNN_LIB}")

target_link_libraries(llmbridge PUBLIC
    libace_napi.z.so
    libhilog_ndk.z.so
    mnn)

# HarmonyOS 5+ ships 16 KB page size kernels on some devices.
target_link_options(llmbridge PRIVATE "-Wl,-z,max-page-size=16384")
