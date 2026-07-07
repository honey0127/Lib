cmake_minimum_required(VERSION 3.22.1)

project("ondevice-rag-core")

# C++17 표준 강제
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- llama.cpp (Phase 3) — 태그 b9893 고정, 정적 링크 ----
# FetchContent 가 구성 단계에 소스를 내려받는다(최초 1회, ABI별 빌드 디렉터리에 캐시).
# 모델 가중치(GGUF)는 저장소/AAR 에 포함하지 않는다 — README 의 모델 준비 안내 참고.
include(FetchContent)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)   # 크로스컴파일에서 -march=native 금지
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)   # NDK OpenMP 링크 이슈 회피
FetchContent_Declare(
        llama
        URL https://github.com/ggml-org/llama.cpp/archive/refs/tags/b9893.tar.gz
)
FetchContent_MakeAvailable(llama)

# 네이티브 라이브러리 정의 및 소스 파일 등록
# (tests/ 하위는 호스트 전용 테스트라 타깃에 넣지 않는다)
add_library(
        rag-core
        SHARED
        native-lib.cpp
        rag_jni.cpp
        vector_index.cpp
        hnsw_index.cpp
        index_io.cpp
        llm_engine.cpp
        llm_jni.cpp
)

# 안드로이드 로그 라이브러리 등 링크
find_library(
        log-lib
        log
)

target_link_libraries(
        rag-core
        llama
        ${log-lib}
)
