#include <jni.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "hnsw_index.h"
#include "index_io.h"
#include "vector_index.h"

// com.example.rag_library.NativeVectorIndex 의 @JvmStatic external 함수 구현.
// 심볼 규칙: 패키지명 rag_library 안의 '_' 는 '_1' 로 이스케이프된다.
//   → Java_com_example_rag_1library_NativeVectorIndex_<메서드>
// @JvmStatic(정적)이므로 두 번째 인자는 jclass. (CLAUDE.md JNI 규칙 참고)

namespace {

rag::VectorIndex* fromHandle(jlong handle) {
    return reinterpret_cast<rag::VectorIndex*>(handle);
}

void throwIllegalArgument(JNIEnv* env, const char* msg) {
    if (jclass cls = env->FindClass("java/lang/IllegalArgumentException")) {
        env->ThrowNew(cls, msg);
    }
}

void throwIllegalState(JNIEnv* env, const char* msg) {
    if (jclass cls = env->FindClass("java/lang/IllegalStateException")) {
        env->ThrowNew(cls, msg);
    }
}

void throwIoException(JNIEnv* env, const char* msg) {
    if (jclass cls = env->FindClass("java/io/IOException")) {
        env->ThrowNew(cls, msg);
    }
}

// jstring → std::string (Modified UTF-8; 파일 경로 용도로 충분)
bool copyPath(JNIEnv* env, jstring path, std::string& out) {
    const char* p = env->GetStringUTFChars(path, nullptr);
    if (p == nullptr) {
        return false;  // OOM — 예외 pending
    }
    out.assign(p);
    env->ReleaseStringUTFChars(path, p);
    return true;
}

// FloatArray → C++ 버퍼 복사. GetPrimitiveArrayCritical 은 (대부분) 복사 없이 배열을
// 잠깐 고정(pin)하므로, Critical 구간 안에서는 JNI 호출·할당·블로킹 없이 memcpy 만
// 하고 즉시 해제한다. dst 는 호출 전에 미리 할당해 둘 것.
bool copyFloatArray(JNIEnv* env, jfloatArray src, std::vector<float>& dst) {
    void* p = env->GetPrimitiveArrayCritical(src, nullptr);
    if (p == nullptr) {
        return false;  // OOM 등 — JVM 예외가 이미 걸려 있다
    }
    std::memcpy(dst.data(), p, dst.size() * sizeof(float));
    env->ReleasePrimitiveArrayCritical(src, p, JNI_ABORT);  // 읽기 전용 → write-back 생략
    return true;
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_rag_1library_NativeVectorIndex_nativeCreate(JNIEnv* env, jclass, jint dim) {
    if (dim <= 0) {
        throwIllegalArgument(env, "dim must be > 0");
        return 0;
    }
    return reinterpret_cast<jlong>(new rag::BruteForceIndex(dim));
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_rag_1library_NativeVectorIndex_nativeCreateHnsw(
        JNIEnv* env, jclass, jint dim, jint m, jint efConstruction, jint efSearch) {
    if (dim <= 0) {
        throwIllegalArgument(env, "dim must be > 0");
        return 0;
    }
    if (m < 2 || efConstruction < m || efSearch < 1) {
        throwIllegalArgument(env, "invalid HNSW params (m>=2, efConstruction>=m, efSearch>=1)");
        return 0;
    }
    return reinterpret_cast<jlong>(new rag::HnswIndex(dim, m, efConstruction, efSearch));
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_rag_1library_NativeVectorIndex_nativeDestroy(JNIEnv*, jclass, jlong handle) {
    delete fromHandle(handle);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_rag_1library_NativeVectorIndex_nativeAdd(
        JNIEnv* env, jclass, jlong handle, jint id, jfloatArray vector) {
    rag::VectorIndex* idx = fromHandle(handle);
    if (idx == nullptr) {
        throwIllegalState(env, "index handle is closed");
        return;
    }
    if (env->GetArrayLength(vector) != idx->dim()) {
        throwIllegalArgument(env, "vector length != dim");
        return;
    }
    std::vector<float> tmp(static_cast<size_t>(idx->dim()));
    if (!copyFloatArray(env, vector, tmp)) {
        return;
    }
    idx->add(id, tmp.data());
}

// 반환: 삭제된 개수. 백엔드가 삭제 미지원(HNSW)이면 -1.
extern "C" JNIEXPORT jint JNICALL
Java_com_example_rag_1library_NativeVectorIndex_nativeRemove(
        JNIEnv* env, jclass, jlong handle, jintArray removeIds) {
    rag::VectorIndex* idx = fromHandle(handle);
    if (idx == nullptr) {
        throwIllegalState(env, "index handle is closed");
        return 0;
    }
    if (!idx->supportsRemove()) {
        return -1;
    }
    const jsize count = env->GetArrayLength(removeIds);
    if (count == 0) {
        return 0;
    }
    std::vector<int32_t> ids(static_cast<size_t>(count));
    env->GetIntArrayRegion(removeIds, 0, count, ids.data());
    if (env->ExceptionCheck()) {
        return 0;
    }
    return static_cast<jint>(idx->removeByIds(ids.data(), ids.size()));
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_rag_1library_NativeVectorIndex_nativeClear(JNIEnv* env, jclass, jlong handle) {
    rag::VectorIndex* idx = fromHandle(handle);
    if (idx == nullptr) {
        throwIllegalState(env, "index handle is closed");
        return;
    }
    idx->clear();
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_rag_1library_NativeVectorIndex_nativeSave(
        JNIEnv* env, jclass, jlong handle, jstring path) {
    rag::VectorIndex* idx = fromHandle(handle);
    if (idx == nullptr) {
        throwIllegalState(env, "index handle is closed");
        return;
    }
    std::string p;
    if (!copyPath(env, path, p)) {
        return;
    }
    if (!rag::saveIndex(*idx, p.c_str())) {
        throwIoException(env, "failed to save index file");
    }
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_rag_1library_NativeVectorIndex_nativeLoad(JNIEnv* env, jclass, jstring path) {
    std::string p;
    if (!copyPath(env, path, p)) {
        return 0;
    }
    std::unique_ptr<rag::VectorIndex> idx = rag::loadIndex(p.c_str());
    if (!idx) {
        throwIoException(env, "failed to load index file (missing or corrupt)");
        return 0;
    }
    return reinterpret_cast<jlong>(idx.release());
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_rag_1library_NativeVectorIndex_nativeDim(JNIEnv* env, jclass, jlong handle) {
    rag::VectorIndex* idx = fromHandle(handle);
    if (idx == nullptr) {
        throwIllegalState(env, "index handle is closed");
        return 0;
    }
    return idx->dim();
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_rag_1library_NativeVectorIndex_nativeSize(JNIEnv* env, jclass, jlong handle) {
    rag::VectorIndex* idx = fromHandle(handle);
    if (idx == nullptr) {
        throwIllegalState(env, "index handle is closed");
        return 0;
    }
    return static_cast<jint>(idx->size());
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_rag_1library_NativeVectorIndex_nativeSearch(
        JNIEnv* env, jclass, jlong handle, jfloatArray query, jint k,
        jbyteArray allowMask, jintArray outIds, jfloatArray outScores) {
    rag::VectorIndex* idx = fromHandle(handle);
    if (idx == nullptr) {
        throwIllegalState(env, "index handle is closed");
        return 0;
    }
    if (env->GetArrayLength(query) != idx->dim()) {
        throwIllegalArgument(env, "query length != dim");
        return 0;
    }
    if (k <= 0) {
        throwIllegalArgument(env, "k must be > 0");
        return 0;
    }
    if (env->GetArrayLength(outIds) < k || env->GetArrayLength(outScores) < k) {
        throwIllegalArgument(env, "out arrays must hold at least k elements");
        return 0;
    }

    std::vector<float> q(static_cast<size_t>(idx->dim()));
    if (!copyFloatArray(env, query, q)) {
        return 0;
    }

    // 허용 마스크(id 인덱스, 0=제외)는 배열 1회 복사 — 후보별 자바 업콜 없음
    std::vector<uint8_t> mask;
    if (allowMask != nullptr) {
        const jsize maskLen = env->GetArrayLength(allowMask);
        mask.resize(static_cast<size_t>(maskLen));
        if (maskLen > 0) {
            env->GetByteArrayRegion(allowMask, 0, maskLen,
                                    reinterpret_cast<jbyte*>(mask.data()));
            if (env->ExceptionCheck()) {
                return 0;
            }
        }
    }

    const std::vector<rag::SearchHit> hits =
            (allowMask != nullptr)
                    ? idx->topK(q.data(), k, mask.data(), static_cast<int32_t>(mask.size()))
                    : idx->topK(q.data(), k);
    const jint n = static_cast<jint>(hits.size());
    if (n == 0) {
        return 0;
    }

    std::vector<jint> ids(static_cast<size_t>(n));
    std::vector<jfloat> scores(static_cast<size_t>(n));
    for (jint i = 0; i < n; ++i) {
        ids[static_cast<size_t>(i)] = hits[static_cast<size_t>(i)].id;
        scores[static_cast<size_t>(i)] = hits[static_cast<size_t>(i)].score;
    }
    env->SetIntArrayRegion(outIds, 0, n, ids.data());
    env->SetFloatArrayRegion(outScores, 0, n, scores.data());
    return n;
}
