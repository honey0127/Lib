package com.example.lib

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.example.rag_library.Embedder
import com.example.rag_library.HashingEmbedder
import com.example.rag_library.LlmEngine
import com.example.rag_library.OnnxEmbedder
import com.example.rag_library.RagChat
import com.example.rag_library.RagEngine
import com.example.rag_library.SearchResult
import java.io.Closeable
import java.io.File
import java.util.Locale
import java.util.concurrent.Executors

/**
 * 온디바이스 RAG 데모 (Phase 4) — 완전 오프라인 문서 검색 + 답변 생성.
 *
 * 모델 파일이 [MODEL_DIR] 에 있으면 자동 사용한다 (없으면 무모델 폴백):
 * - model.onnx + vocab.tsv → OnnxEmbedder(e5, 시맨틱 검색)  ← 없으면 HashingEmbedder
 * - llm.gguf               → LlmEngine(답변 생성)           ← 없으면 검색 결과만 표시
 * 배치 방법은 README "모델 준비" 참고 (adb push).
 */
class MainActivity : AppCompatActivity() {

    // 엔진 접근은 전부 이 단일 스레드에서 — UI 스레드 블로킹/동시 접근 방지
    private val engineExecutor = Executors.newSingleThreadExecutor()

    private var embedder: Embedder? = null
    private var rag: RagEngine? = null
    private var llm: LlmEngine? = null
    private var chat: RagChat? = null
    private var docCounter = 0

    private lateinit var statusText: TextView
    private lateinit var docInput: EditText
    private lateinit var addDocButton: Button
    private lateinit var addSamplesButton: Button
    private lateinit var chunkCountText: TextView
    private lateinit var questionInput: EditText
    private lateinit var askButton: Button
    private lateinit var answerText: TextView
    private lateinit var sourcesText: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        statusText = findViewById(R.id.statusText)
        docInput = findViewById(R.id.docInput)
        addDocButton = findViewById(R.id.addDocButton)
        addSamplesButton = findViewById(R.id.addSamplesButton)
        chunkCountText = findViewById(R.id.chunkCountText)
        questionInput = findViewById(R.id.questionInput)
        askButton = findViewById(R.id.askButton)
        answerText = findViewById(R.id.answerText)
        sourcesText = findViewById(R.id.sourcesText)

        setButtonsEnabled(false)
        engineExecutor.execute { initEngines() }

        addDocButton.setOnClickListener {
            val text = docInput.text.toString()
            if (text.isNotBlank()) {
                docInput.setText("")
                addDocuments(listOf("문서${++docCounter}" to text))
            }
        }
        addSamplesButton.setOnClickListener { addDocuments(SAMPLE_DOCS) }
        askButton.setOnClickListener { onAsk() }
    }

    override fun onDestroy() {
        super.onDestroy()
        engineExecutor.execute {
            chat = null
            llm?.close()
            rag?.close()
            (embedder as? Closeable)?.close()
        }
        engineExecutor.shutdown()
    }

    /** 모델 파일 유무에 따라 임베더/LLM 을 선택해 초기화한다 (백그라운드). */
    private fun initEngines() {
        val dir = File(MODEL_DIR)
        val onnxFile = File(dir, "model.onnx")
        val vocabFile = File(dir, "vocab.tsv")
        val ggufFile = File(dir, "llm.gguf")

        val onnxEmbedder = if (onnxFile.exists() && vocabFile.exists()) {
            runCatching { OnnxEmbedder.create(onnxFile.absolutePath, vocabFile.absolutePath) }
                .getOrNull()
        } else {
            null
        }
        val activeEmbedder = onnxEmbedder ?: HashingEmbedder()
        val engine = RagEngine(activeEmbedder)

        runOnUiThread { statusText.text = "임베더 준비 완료 — LLM 로딩 중…" }
        val llmEngine = if (ggufFile.exists()) {
            runCatching { LlmEngine.create(ggufFile.absolutePath) }.getOrNull()
        } else {
            null
        }

        embedder = activeEmbedder
        rag = engine
        llm = llmEngine
        chat = llmEngine?.let { RagChat(engine, it) }

        val embedderLabel = if (onnxEmbedder != null) "e5(ONNX·시맨틱)" else "해싱(무모델)"
        val llmLabel = if (llmEngine != null) "Qwen GGUF ✓" else "없음 → 검색만"
        runOnUiThread {
            statusText.text = "임베더: $embedderLabel  |  LLM: $llmLabel"
            setButtonsEnabled(true)
        }
    }

    private fun addDocuments(docs: List<Pair<String, String>>) {
        setButtonsEnabled(false)
        engineExecutor.execute {
            val engine = rag ?: return@execute
            docs.forEach { (id, text) -> engine.addDocument(id, text) }
            val count = engine.chunkCount()
            runOnUiThread {
                chunkCountText.text = "인덱싱된 청크: $count"
                setButtonsEnabled(true)
            }
        }
    }

    private fun onAsk() {
        val question = questionInput.text.toString().trim()
        if (question.isEmpty()) return
        setButtonsEnabled(false)
        answerText.text = ""
        sourcesText.text = ""

        engineExecutor.execute {
            val engine = rag ?: return@execute
            val ragChat = chat
            if (ragChat != null) {
                // 검색 + LLM 답변 (완전한 문자 단위 스트리밍)
                val answer = ragChat.ask(question) { piece ->
                    runOnUiThread { answerText.append(piece) }
                    true
                }
                runOnUiThread {
                    sourcesText.text = formatSources(answer.sources)
                    setButtonsEnabled(true)
                }
            } else {
                // LLM 모델이 없으면 검색 결과만
                val results = engine.search(question, topK = 3)
                runOnUiThread {
                    answerText.text = "(LLM 모델 없음 — 검색 결과만 표시. README 모델 준비 참고)"
                    sourcesText.text = formatSources(results)
                    setButtonsEnabled(true)
                }
            }
        }
    }

    private fun formatSources(sources: List<SearchResult>): String {
        if (sources.isEmpty()) return "(검색 결과 없음 — 먼저 문서를 넣어주세요)"
        return sources.mapIndexed { i, r ->
            val preview = r.chunk.text.replace('\n', ' ').take(80)
            String.format(Locale.KOREA, "%d. [%s] 유사도 %.3f\n%s", i + 1, r.chunk.docId, r.score, preview)
        }.joinToString("\n\n")
    }

    private fun setButtonsEnabled(enabled: Boolean) {
        addDocButton.isEnabled = enabled
        addSamplesButton.isEnabled = enabled
        askButton.isEnabled = enabled
    }

    companion object {
        private const val MODEL_DIR = "/data/local/tmp/rag-models"

        private val SAMPLE_DOCS = listOf(
            "김치" to "김치는 배추를 소금에 절인 뒤 고춧가루, 마늘, 젓갈 양념으로 버무려 발효시킨 한국 전통 음식이다. " +
                "유산균이 풍부해 장 건강에 도움을 준다. 지역과 계절에 따라 종류가 매우 다양하다.",
            "한글" to "한글은 1443년 세종대왕이 창제한 문자다. 자음과 모음을 조합해 소리를 표기하는 표음 문자이며, " +
                "창제 원리를 기록한 훈민정음 해례본은 유네스코 세계기록유산으로 등재되어 있다.",
            "화성" to "화성은 태양계의 네 번째 행성으로, 산화철 때문에 표면이 붉게 보여 붉은 행성이라고 불린다. " +
                "여러 탐사 로버가 물의 흔적을 조사해 왔으며 유인 탐사 계획도 논의되고 있다.",
            "온디바이스AI" to "온디바이스 AI는 클라우드 서버 없이 스마트폰 같은 기기 안에서 인공지능 모델을 실행하는 기술이다. " +
                "데이터가 기기를 떠나지 않아 개인정보 보호에 유리하고, 인터넷 없이 오프라인에서도 동작한다.",
        )
    }
}
