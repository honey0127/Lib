package com.example.rag_library

import androidx.test.ext.junit.runners.AndroidJUnit4
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * ONNX e5 임베더 통합 테스트 — 모델 파일이 기기에 있을 때만 실행된다(없으면 skip).
 *
 * 준비(개발 PC):
 *   python scripts/prepare_model.py --out models
 *   adb shell mkdir -p /data/local/tmp/rag-models
 *   adb push models/model.onnx models/vocab.tsv /data/local/tmp/rag-models/
 */
@RunWith(AndroidJUnit4::class)
class OnnxEmbedderTest {

    private val modelFile = File("/data/local/tmp/rag-models/model.onnx")
    private val vocabFile = File("/data/local/tmp/rag-models/vocab.tsv")

    private fun cos(a: FloatArray, b: FloatArray): Float {
        var d = 0f
        for (i in a.indices) d += a[i] * b[i]
        return d
    }

    @Test
    fun semanticSimilarity_ranksRelatedPassageHigher() {
        assumeTrue(
            "모델 파일 없음 — scripts/prepare_model.py + adb push 후 실행 (건너뜀)",
            modelFile.exists() && vocabFile.exists(),
        )
        OnnxEmbedder.create(modelFile.absolutePath, vocabFile.absolutePath).use { embedder ->
            val query = embedder.embedQuery("김치는 어떤 음식이야?")
            assertEquals(embedder.dim, query.size)

            val food = embedder.embed("김치는 배추를 발효시켜 만드는 한국 음식이다.")
            val space = embedder.embed("화성은 태양계의 네 번째 행성이다.")

            // 단위 노름 확인
            assertEquals(1f, cos(query, query), 1e-3f)
            // 시맨틱 랭킹: 해싱 임베더와 달리 의미가 다른 문장은 낮아야 한다
            assertTrue("cos(food)=${cos(query, food)} <= cos(space)=${cos(query, space)}",
                cos(query, food) > cos(query, space))
        }
    }

    @Test
    fun ragEngine_endToEnd_withOnnxEmbedder() {
        assumeTrue(
            "모델 파일 없음 — scripts/prepare_model.py + adb push 후 실행 (건너뜀)",
            modelFile.exists() && vocabFile.exists(),
        )
        val embedder = OnnxEmbedder.create(modelFile.absolutePath, vocabFile.absolutePath)
        RagEngine(embedder).use { rag ->
            rag.addDocument("food", "김치는 배추를 발효시켜 만드는 한국의 전통 음식이다.")
            rag.addDocument("space", "화성 탐사선은 로켓 연료를 사용해 발사된다.")
            val results = rag.search("발효 음식", topK = 1)
            assertEquals("food", results.single().chunk.docId)
        }
        embedder.close()
    }
}
