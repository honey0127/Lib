package com.example.rag_library

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * 청킹 → 임베딩 → JNI → C++ 인덱스 → 결과 매핑까지 전 구간(E2E)을 실기기에서 검증한다.
 */
@RunWith(AndroidJUnit4::class)
class RagEngineTest {

    @Test
    fun endToEnd_addAndSearch() {
        RagEngine().use { rag ->
            assertTrue(rag.addDocument("food", "김치는 발효 음식이다. 배추와 고춧가루로 만든다.") > 0)
            assertTrue(rag.addDocument("space", "화성은 태양계의 네 번째 행성이다. 로켓은 연료를 태워 우주로 간다.") > 0)

            val results = rag.search("김치는 어떤 음식이야?", topK = 2)

            assertTrue(results.isNotEmpty())
            assertEquals("food", results[0].chunk.docId)
            assertTrue(results[0].score >= results.last().score) // 내림차순
            assertTrue(results.all { it.score in -1.0001f..1.0001f })
        }
    }

    @Test
    fun emptyEngine_returnsEmpty() {
        RagEngine().use { rag ->
            assertTrue(rag.search("아무거나").isEmpty())
        }
    }

    @Test
    fun topK_boundsResultCount() {
        RagEngine().use { rag ->
            rag.addDocument("d", "하나. 둘. 셋.")
            assertTrue(rag.search("하나", topK = 1).size <= 1)
        }
    }

    @Test
    fun clear_emptiesIndex_andAllowsReuse() {
        RagEngine().use { rag ->
            rag.addDocument("food", "김치는 발효 음식이다.")
            assertTrue(rag.chunkCount() > 0)

            rag.clear()
            assertEquals(0, rag.chunkCount())
            assertTrue(rag.search("김치").isEmpty())

            rag.addDocument("space", "화성은 붉은 행성이다.")
            assertEquals("space", rag.search("화성", topK = 1).single().chunk.docId)
        }
    }

    @Test
    fun endToEnd_withHnswIndex() {
        RagEngine(indexKind = IndexKind.HNSW).use { rag ->
            rag.addDocument("food", "김치는 발효 음식이다. 배추와 고춧가루로 만든다.")
            rag.addDocument("space", "화성은 태양계의 네 번째 행성이다.")
            val results = rag.search("김치", topK = 1)
            assertEquals("food", results.single().chunk.docId)
        }
    }
}
