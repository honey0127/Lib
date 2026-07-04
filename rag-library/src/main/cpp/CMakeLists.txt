cmake_minimum_required(VERSION 3.22.1)

project("ondevice-rag-core")

# C++17 표준 강제
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 네이티브 라이브러리 정의 및 소스 파일 등록
add_library(
        rag-core
        SHARED
        native-lib.cpp
)

# 안드로이드 로그 라이브러리 등 링크
find_library(
        log-lib
        log
)

target_link_libraries(
        rag-core
        ${log-lib}
)