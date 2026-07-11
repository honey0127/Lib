package com.example.rag_library

import java.io.Closeable
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File

/** [RagEngine] 이 사용할 벡터 인덱스 백엔드. */
enum class IndexKind {
    /**
     * 정확한 브루트포스 — 수천 청크까지 충분히 빠르고 결과가 항상 정확하다(기본).
     * 모바일 개인 문서 규모에선 이것이 옳은 선택이다: [RagEngine.search]의 docId 필터가
     * 정확도 손실 0으로 동작하고, [RagEngine.removeDocument](swap-remove)도 무손실 공짜다
     * — 필터드 서치/삭제가 알려진 난제인 ANN 계열과의 결정적 차이.
     */
    BRUTE_FORCE,

    /**
     * HNSW 근사 최근접 이웃 — 수만 청크 이상 대규모 코퍼스용(재현율 ≈ 1, 훨씬 빠름).
     * 제약: 삭제 미지원, docId 필터는 포스트 필터(좁은 필터에서 k 미만 반환 가능).
     */
    HNSW,
}

/**
 * 온디바이스 RAG 파사드 — 문서 추가/삭제와 유사도 검색.
 *
 * ```
 * RagEngine().use { rag ->
 *     rag.addDocument("guide", "김치는 발효 음식이다. 배추와 고춧가루로 만든다.")
 *     val top = rag.search("발효 음식", topK = 3)
 *     rag.search("발효", topK = 3, allowDocIds = setOf("guide"))  // 문서 필터
 *     rag.removeDocument("guide")                                  // 삭제 (BRUTE_FORCE)
 * }
 * ```
 *
 * 재시작 후 재인덱싱을 피하려면 [save] / [load] 스냅샷을 사용한다 (Phase 2):
 * ```
 * rag.save(File(dir, "snapshot"))                  // snapshot.idx + snapshot.meta 생성
 * val rag2 = RagEngine.load(File(dir, "snapshot")) // 같은 임베더로 즉시 복원
 * ```
 *
 * - 완전 오프라인: 청킹 → 임베딩 → 벡터 검색이 전부 기기 안에서 끝난다.
 * - 네이티브(C++) 인덱스를 쥐고 있으므로 사용 후 반드시 [close] (또는 `use {}`) 할 것.
 * - 모든 공개 메서드는 스레드 안전(@Synchronized).
 */
class RagEngine private constructor(
    private val embedder: Embedder,
    private val chunker: TextChunker,
    private val index: NativeVectorIndex,
    private val chunks: LinkedHashMap<Int, Chunk>, // id → 청크 (삭제 후 id 는 불연속일 수 있음)
) : Closeable {

    constructor(
        embedder: Embedder = HashingEmbedder(),
        chunker: TextChunker = TextChunker(),
        indexKind: IndexKind = IndexKind.BRUTE_FORCE,
    ) : this(
        embedder = embedder,
        chunker = chunker,
        index = when (indexKind) {
            IndexKind.BRUTE_FORCE -> NativeVectorIndex.bruteForce(embedder.dim)
            IndexKind.HNSW -> NativeVectorIndex.hnsw(embedder.dim)
        },
        chunks = LinkedHashMap(),
    )

    private var closed = false
    private var nextId = (chunks.keys.maxOrNull() ?: -1) + 1

    /** [text]를 청크로 분할·임베딩해 인덱스에 추가하고, 추가된 청크 수를 반환한다. */
    @Synchronized
    fun addDocument(docId: String, text: String): Int {
        checkOpen()
        val newChunks = chunker.chunk(docId, text)
        for (c in newChunks) {
            val id = nextId++
            index.add(id, embedder.embed(c.text))
            chunks[id] = c
        }
        return newChunks.size
    }

    /**
     * [query]와 가장 유사한 청크 [topK]개를 유사도 내림차순으로 반환한다.
     *
     * [allowDocIds]가 주어지면 해당 문서의 청크만 검색한다 — 허용 마스크가 배열 1회로
     * 네이티브에 전달되어 C++ 스캔 루프 안에서 걸러진다(청크별 콜백 없음).
     * BRUTE_FORCE 에선 정확 필터, HNSW 에선 포스트 필터([IndexKind] 참고).
     */
    @Synchronized
    fun search(query: String, topK: Int = 5, allowDocIds: Set<String>? = null): List<SearchResult> {
        checkOpen()
        require(topK > 0) { "topK must be > 0" }
        if (chunks.isEmpty() || query.isBlank()) return emptyList()

        val mask: ByteArray? = allowDocIds?.let { allowed ->
            val m = ByteArray(nextId)
            for ((id, chunk) in chunks) {
                if (chunk.docId in allowed) m[id] = 1
            }
            m
        }
        return index.search(embedder.embedQuery(query), topK, mask)
            .map { SearchResult(chunks.getValue(it.id), it.score) }
    }

    /**
     * [docId]의 모든 청크를 삭제하고 삭제된 청크 수를 반환한다(없으면 0).
     * @throws UnsupportedOperationException HNSW 백엔드 — 삭제는 BRUTE_FORCE 전용
     *   (그래프 노드 삭제는 알려진 난제. HNSW 는 [clear] 후 재구축할 것)
     */
    @Synchronized
    fun removeDocument(docId: String): Int {
        checkOpen()
        val ids = chunks.filterValues { it.docId == docId }.keys
        if (ids.isEmpty()) return 0
        val removed = index.remove(ids.toIntArray()) // HNSW 면 여기서 예외 (chunks 는 그대로)
        ids.forEach { chunks.remove(it) }
        return removed
    }

    /** 지금까지 인덱싱된 청크 수. */
    @Synchronized
    fun chunkCount(): Int = chunks.size

    /** 모든 문서/청크를 비운다. 인덱스 백엔드와 파라미터는 유지된다. */
    @Synchronized
    fun clear() {
        checkOpen()
        index.clear()
        chunks.clear()
        nextId = 0
    }

    /**
     * 현재 상태를 `<baseFile>.idx`(네이티브 인덱스) + `<baseFile>.meta`(청크 텍스트)로 저장한다.
     * 임베더 자체는 저장되지 않는다 — [load] 시 저장 때와 같은 임베더를 넘겨야 한다.
     * 실패 시 [java.io.IOException].
     */
    @Synchronized
    fun save(baseFile: File) {
        checkOpen()
        index.save(indexFileOf(baseFile))
        DataOutputStream(metaFileOf(baseFile).outputStream().buffered()).use { out ->
            ChunkStore.save(out, embedder.dim, chunks)
        }
    }

    @Synchronized
    override fun close() {
        if (!closed) {
            closed = true
            index.close()
        }
    }

    private fun checkOpen() = check(!closed) { "RagEngine is closed" }

    companion object {
        /**
         * [save] 스냅샷을 복원한다. 인덱스 백엔드(브루트포스/HNSW)는 파일에서 자동 복원된다.
         *
         * @param embedder 저장 시점과 같은 차원의 임베더 (다르면 [IllegalArgumentException])
         * @throws java.io.IOException 파일 누락/손상
         */
        fun load(
            baseFile: File,
            embedder: Embedder = HashingEmbedder(),
            chunker: TextChunker = TextChunker(),
        ): RagEngine {
            val (dim, chunks) = DataInputStream(metaFileOf(baseFile).inputStream().buffered())
                .use { ChunkStore.load(it) }
            require(dim == embedder.dim) {
                "저장된 임베더 차원($dim) != 전달된 임베더 차원(${embedder.dim}) — 저장 때와 같은 임베더로 로드할 것"
            }
            val index = NativeVectorIndex.load(indexFileOf(baseFile))
            try {
                check(index.dim == dim) { "인덱스 차원(${index.dim}) != 메타 차원($dim)" }
                check(index.size() == chunks.size) {
                    "인덱스 벡터 수(${index.size()}) != 청크 수(${chunks.size}) — 스냅샷 불일치"
                }
            } catch (t: Throwable) {
                index.close()
                throw t
            }
            return RagEngine(embedder, chunker, index, chunks)
        }

        private fun indexFileOf(base: File) = File(base.parentFile, base.name + ".idx")
        private fun metaFileOf(base: File) = File(base.parentFile, base.name + ".meta")
    }
}
