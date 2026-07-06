#pragma once

#include <cstdio>
#include <memory>

namespace rag {

class VectorIndex;

// ---- 인덱스 파일 직렬화 (Phase 2) ----
// 단일 바이너리 파일, little-endian 고정(안드로이드 arm64/x86_64·리눅스 호스트 공통):
//   u32 magic 'RAG1' | u32 kind(1=BruteForce,2=Hnsw) | i32 dim | i64 count | <본문>
// 본문 레이아웃은 각 구현의 writeBody/readBody 참고. 벡터 블록이 고정 오프셋에
// 연속 배치되므로 추후 mmap 제로카피 로드로 확장 가능한 포맷이다.
//
// 실패 시: saveIndex 는 false 반환 + 불완전 파일 삭제, loadIndex 는 nullptr 반환
// (마법값/범위 검증으로 손상·타 포맷 파일을 거부한다).

bool saveIndex(const VectorIndex& idx, const char* path);
std::unique_ptr<VectorIndex> loadIndex(const char* path);

// 내부 공용 IO 헬퍼 (writeBody/readBody 구현용)
namespace io {

inline bool writeBytes(std::FILE* f, const void* p, size_t n) {
    if (n == 0) {
        return true;
    }
    return std::fwrite(p, 1, n, f) == n;
}

inline bool readBytes(std::FILE* f, void* p, size_t n) {
    if (n == 0) {
        return true;
    }
    return std::fread(p, 1, n, f) == n;
}

template <typename T>
inline bool writePod(std::FILE* f, const T& v) {
    return writeBytes(f, &v, sizeof(T));
}

template <typename T>
inline bool readPod(std::FILE* f, T* v) {
    return readBytes(f, v, sizeof(T));
}

}  // namespace io
}  // namespace rag
