package com.example.rag_library

import androidx.test.ext.junit.runners.AndroidJUnit4
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * LLM(GGUF) 통합 테스트 — 모델 파일이 기기에 있을 때만 실행(없으면 skip, CI 는 초록 유지).
 *
 * 준비(개발 PC):
 *   huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF \
 *       qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir models
 *   adb push models/qwen2.5-1.5b-instruct-q4_k_m.gguf /data/local/tmp/rag-models/llm.gguf
 */
@RunWith(AndroidJUnit4::class)
class LlmEngineTest {

    private val modelFile = File("/data/local/tmp/rag-models/llm.gguf")

    @Test
    fun generate_streamsAndReturnsSameText() {
        assumeTrue("LLM 모델 없음 — README Phase 3 안내 참고 (건너뜀)", modelFile.exists())
        LlmEngine.create(modelFile.absolutePath, contextSize = 1024, threads = 2).use { llm ->
            assertTrue(llm.countTokens("안녕하세요") > 0)

            val streamed = StringBuilder()
            val full = llm.generate(
                QwenPromptBuilder.build("1 더하기 1은 몇이야? 숫자만 답해.", emptyList()),
                maxTokens = 24,
            ) { piece ->
                streamed.append(piece)
                true
            }
            assertTrue("생성 결과가 비어 있음", full.isNotBlank())
            assertEquals(full, streamed.toString())
        }
    }

    @Test
    fun ragChat_answersFromIndexedDocuments() {
        assumeTrue("LLM 모델 없음 (건너뜀)", modelFile.exists())
        RagEngine().use { rag ->
            rag.addDocument("food", "김치는 배추를 소금에 절여 고춧가루 양념으로 발효시킨 한국 전통 음식이다.")
            rag.addDocument("space", "화성은 태양계의 네 번째 행성이며 붉은 표면을 가졌다.")

            LlmEngine.create(modelFile.absolutePath, contextSize = 1024, threads = 2).use { llm ->
                val chat = RagChat(rag, llm, topK = 2, maxAnswerTokens = 64)
                val answer = chat.ask("김치는 어떤 음식이야?")

                assertTrue("답변이 비어 있음", answer.text.isNotBlank())
                assertTrue(answer.sources.isNotEmpty())
                assertEquals("food", answer.sources.first().chunk.docId)
            }
        }
    }
}
