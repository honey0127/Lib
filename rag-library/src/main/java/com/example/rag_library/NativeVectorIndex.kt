package com.example.rag_library

import java.io.Closeable
import java.io.File
import java.io.IOException

/**
 * C++ 벡터 인덱스(librag-core.so)의 얇은 JNI 래퍼.
 *
 * 백엔드는 생성 팩토리로 선택한다:
 * - [bruteForce]: 정확(exact) 전수 비교 — 수천 청크까지 기본값
 * - [hnsw]: HNSW 근사 최근접 이웃 — 대규모 코퍼스용
 *
 * - 스레드 안전하지 않다. 동기화는 상위 레이어([RagEngine])가 담당한다.
 * - 네이티브 힙 메모리를 쥐고 있으므로 사용 후 반드시 [close] (또는 `use {}`) 할 것.
 */
internal class NativeVectorIndex private constructor(
    val dim: Int,
    private var handle: Long,
) : Closeable {

    internal data class Hit(val id: Int, val score: Float)

    fun add(id: Int, vector: FloatArray) {
        require(vector.size == dim) { "vector.size(${vector.size}) != dim($dim)" }
        nativeAdd(checkOpen(), id, vector)
    }

    /** 코사인 유사도 상위 k개를 점수 내림차순으로 반환한다(인덱스가 k보다 작으면 그만큼만). */
    fun search(query: FloatArray, k: Int): List<Hit> {
        require(query.size == dim) { "query.size(${query.size}) != dim($dim)" }
        require(k > 0) { "k must be > 0" }
        val ids = IntArray(k)
        val scores = FloatArray(k)
        val n = nativeSearch(checkOpen(), query, k, ids, scores)
        return List(n) { Hit(ids[it], scores[it]) }
    }

    fun size(): Int = nativeSize(checkOpen())

    /** 모든 벡터를 비운다(백엔드/파라미터는 유지). */
    fun clear() = nativeClear(checkOpen())

    /** 인덱스를 [file]에 저장한다(백엔드·그래프 포함). 실패 시 [IOException]. */
    fun save(file: File) {
        nativeSave(checkOpen(), file.absolutePath)
    }

    override fun close() {
        if (handle != 0L) {
            nativeDestroy(handle)
            handle = 0L
        }
    }

    private fun checkOpen(): Long {
        check(handle != 0L) { "NativeVectorIndex is closed" }
        return handle
    }

    companion object {
        init {
            System.loadLibrary("rag-core")
        }

        /** 정확(브루트포스) 인덱스. 수천 청크 규모까지는 이걸 쓴다(기본값). */
        fun bruteForce(dim: Int): NativeVectorIndex {
            require(dim > 0) { "dim must be > 0" }
            return NativeVectorIndex(dim, nativeCreate(dim))
        }

        /**
         * HNSW 근사 인덱스. 수만 청크 이상에서 브루트포스 대신 사용.
         * @param m 레벨당 연결 수  @param efConstruction 삽입 beam 폭
         * @param efSearch 검색 beam 폭 — 클수록 재현율↑/느려짐.
         *   (무작위 최악 데이터 기준 재현율은 cpp/tests/bench.cpp 로 측정; 실제 임베딩은 더 높다)
         */
        fun hnsw(
            dim: Int,
            m: Int = 16,
            efConstruction: Int = 200,
            efSearch: Int = 128,
        ): NativeVectorIndex {
            require(dim > 0) { "dim must be > 0" }
            require(m >= 2) { "m must be >= 2" }
            require(efConstruction >= m) { "efConstruction must be >= m" }
            require(efSearch >= 1) { "efSearch must be >= 1" }
            return NativeVectorIndex(dim, nativeCreateHnsw(dim, m, efConstruction, efSearch))
        }

        /**
         * [save]로 저장한 인덱스를 복원한다. 파일 헤더의 kind 에 따라
         * 브루트포스/HNSW 백엔드가 자동 선택된다. 손상·비정상 파일이면 [IOException].
         */
        fun load(file: File): NativeVectorIndex {
            val handle = nativeLoad(file.absolutePath)
            return NativeVectorIndex(nativeDim(handle), handle)
        }

        // 대응 C++ 심볼: Java_com_example_rag_1library_NativeVectorIndex_<이름>
        // @JvmStatic 이므로 네이티브 시그니처는 (JNIEnv*, jclass, ...) — CLAUDE.md JNI 규칙 참고
        @JvmStatic private external fun nativeCreate(dim: Int): Long
        @JvmStatic private external fun nativeCreateHnsw(
            dim: Int,
            m: Int,
            efConstruction: Int,
            efSearch: Int,
        ): Long
        @JvmStatic private external fun nativeDestroy(handle: Long)
        @JvmStatic private external fun nativeClear(handle: Long)
        @JvmStatic private external fun nativeSave(handle: Long, path: String)
        @JvmStatic private external fun nativeLoad(path: String): Long
        @JvmStatic private external fun nativeDim(handle: Long): Int
        @JvmStatic private external fun nativeAdd(handle: Long, id: Int, vector: FloatArray)
        @JvmStatic private external fun nativeSize(handle: Long): Int
        @JvmStatic private external fun nativeSearch(
            handle: Long,
            query: FloatArray,
            k: Int,
            outIds: IntArray,
            outScores: FloatArray,
        ): Int
    }
}
