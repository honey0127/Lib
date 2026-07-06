package com.example.rag_library

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.io.IOException
import java.util.Random
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Phase 2 영속화 검증 — 저장 → 복원 후 동일한 검색 결과가 나와야 한다.
 * (직렬화 포맷/손상 파일 케이스는 cpp/tests/index_io_test.cpp 호스트 테스트가 커버)
 */
@RunWith(AndroidJUnit4::class)
class PersistenceTest {

    private val cacheDir: File =
        InstrumentationRegistry.getInstrumentation().targetContext.cacheDir

    @Test
    fun nativeIndex_saveLoad_sameResults() {
        val file = File(cacheDir, "hnsw_snapshot.bin")
        try {
            NativeVectorIndex.hnsw(8).use { idx ->
                val rnd = Random(3)
                val vecs = List(40) { FloatArray(8) { rnd.nextFloat() - 0.5f } }
                vecs.forEachIndexed { i, v -> idx.add(i, v) }
                idx.save(file)

                NativeVectorIndex.load(file).use { loaded ->
                    assertEquals(idx.dim, loaded.dim)
                    assertEquals(idx.size(), loaded.size())
                    val q = FloatArray(8) { rnd.nextFloat() - 0.5f }
                    assertEquals(idx.search(q, 5), loaded.search(q, 5))
                }
            }
        } finally {
            file.delete()
        }
    }

    @Test
    fun ragEngine_saveLoad_roundTrip() {
        val base = File(cacheDir, "rag_snapshot")
        try {
            RagEngine().use { rag ->
                rag.addDocument("food", "김치는 발효 음식이다. 배추로 만든다.")
                rag.addDocument("space", "화성은 태양계의 네 번째 행성이다.")
                rag.save(base)
            }
            RagEngine.load(base).use { rag ->
                assertEquals(2, rag.chunkCount())
                assertEquals("food", rag.search("김치", topK = 1).single().chunk.docId)
            }
        } finally {
            File(cacheDir, "rag_snapshot.idx").delete()
            File(cacheDir, "rag_snapshot.meta").delete()
        }
    }

    @Test(expected = IllegalArgumentException::class)
    fun ragEngine_load_embedderDimMismatch_throws() {
        val base = File(cacheDir, "rag_dim_mismatch")
        try {
            RagEngine().use { rag -> // 기본 HashingEmbedder(256)
                rag.addDocument("d", "문서 하나.")
                rag.save(base)
            }
            RagEngine.load(base, embedder = HashingEmbedder(dim = 128)).close()
        } finally {
            File(cacheDir, "rag_dim_mismatch.idx").delete()
            File(cacheDir, "rag_dim_mismatch.meta").delete()
        }
    }

    @Test(expected = IOException::class)
    fun nativeIndex_load_corruptFile_throwsIOException() {
        val file = File(cacheDir, "corrupt.bin")
        try {
            file.writeBytes("garbage-not-an-index".toByteArray())
            NativeVectorIndex.load(file).close()
        } finally {
            file.delete()
        }
    }
}
