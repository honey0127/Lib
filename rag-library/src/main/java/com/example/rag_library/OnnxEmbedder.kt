package com.example.rag_library

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.io.Closeable
import java.io.File
import java.nio.LongBuffer

/**
 * ONNX Runtime + multilingual-e5-small 시맨틱 임베더 (Phase 0.5).
 *
 * 모델/vocab 파일은 저장소에 포함하지 않는다 — `scripts/prepare_model.py` 로 준비해
 * 기기에 배치한 뒤 경로를 넘긴다:
 * ```
 * val embedder = OnnxEmbedder.create("/data/local/tmp/rag-models/model.onnx",
 *                                    "/data/local/tmp/rag-models/vocab.tsv")
 * RagEngine(embedder).use { ... }
 * ```
 *
 * e5 계열 규약: 문서는 `"passage: "`, 질의는 `"query: "` 프리픽스를 붙여야
 * 검색 품질이 제대로 나온다 — [embed]/[embedQuery]가 각각 자동으로 붙인다.
 *
 * 스레드 안전: OrtSession.run 은 스레드 안전하지만, 이 클래스는 [RagEngine]의
 * @Synchronized 아래에서 쓰는 것을 전제로 한다. 사용 후 [close] 필수.
 */
class OnnxEmbedder private constructor(
    private val env: OrtEnvironment,
    private val session: OrtSession,
    private val tokenizer: Tokenizer,
    override val dim: Int,
    private val maxSeqLen: Int,
) : Embedder, Closeable {

    override fun embed(text: String): FloatArray = run("$PASSAGE_PREFIX$text")

    override fun embedQuery(text: String): FloatArray = run("$QUERY_PREFIX$text")

    private fun run(text: String): FloatArray {
        val ids = tokenizer.encode(text, maxSeqLen)
        val seqLen = ids.size.toLong()
        val inputIds = LongArray(ids.size) { ids[it].toLong() }
        val attentionMask = LongArray(ids.size) { 1L } // 배치 1 → 패딩 없음

        OnnxTensor.createTensor(env, LongBuffer.wrap(inputIds), longArrayOf(1, seqLen)).use { idsTensor ->
            OnnxTensor.createTensor(env, LongBuffer.wrap(attentionMask), longArrayOf(1, seqLen)).use { maskTensor ->
                session.run(
                    mapOf(INPUT_IDS to idsTensor, ATTENTION_MASK to maskTensor),
                ).use { result ->
                    @Suppress("UNCHECKED_CAST")
                    val batch = result[0].value as Array<Array<FloatArray>> // [1][seq][dim]
                    return Pooling.meanPoolNormalize(batch[0])
                }
            }
        }
    }

    /** OrtEnvironment 는 프로세스 전역 싱글턴이라 닫지 않는다. */
    override fun close() {
        session.close()
    }

    companion object {
        private const val INPUT_IDS = "input_ids"
        private const val ATTENTION_MASK = "attention_mask"
        private const val PASSAGE_PREFIX = "passage: "
        private const val QUERY_PREFIX = "query: "
        private const val FALLBACK_DIM = 384 // multilingual-e5-small

        /**
         * @param modelPath  optimum 으로 내보낸 e5 ONNX 모델(.onnx)
         * @param vocabPath  prepare_model.py 가 생성한 vocab TSV
         * @param maxSeqLen  최대 토큰 수 (e5 = 512)
         */
        fun create(modelPath: String, vocabPath: String, maxSeqLen: Int = 512): OnnxEmbedder {
            require(File(modelPath).exists()) { "모델 파일 없음: $modelPath" }
            val env = OrtEnvironment.getEnvironment()
            val session = env.createSession(modelPath, OrtSession.SessionOptions())
            try {
                // last_hidden_state 의 마지막 차원에서 임베딩 크기 추론 (동적이면 fallback)
                val outInfo = session.outputInfo.values.first().info as? TensorInfo
                val lastDim = outInfo?.shape?.lastOrNull() ?: -1L
                val dim = if (lastDim > 0) lastDim.toInt() else FALLBACK_DIM
                return OnnxEmbedder(
                    env = env,
                    session = session,
                    tokenizer = UnigramTokenizer.load(File(vocabPath)),
                    dim = dim,
                    maxSeqLen = maxSeqLen,
                )
            } catch (t: Throwable) {
                session.close()
                throw t
            }
        }
    }
}
