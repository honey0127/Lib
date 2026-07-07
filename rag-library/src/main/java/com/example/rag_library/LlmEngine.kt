package com.example.rag_library

import java.io.Closeable
import java.io.File
import java.io.IOException

/**
 * 온디바이스 LLM(llama.cpp/GGUF) 래퍼 (Phase 3).
 *
 * 모델 가중치는 저장소/AAR 에 포함되지 않는다 — 권장 모델(README 참고):
 * `Qwen/Qwen2.5-1.5B-Instruct-GGUF` 의 `q4_k_m` (약 1.1GB, Apache-2.0).
 *
 * - [generate]는 블로킹 호출이다 — 반드시 백그라운드 스레드에서 부를 것.
 * - 스레드 안전하지 않다. 한 번에 하나의 generate 만 실행한다.
 * - 모델 가중치를 네이티브 힙에 크게 쥔다 — 사용 후 반드시 [close] (또는 `use {}`).
 */
class LlmEngine private constructor(private var handle: Long) : Closeable {

    /**
     * [prompt]에 이어 최대 [maxTokens] 토큰을 생성해 전체 텍스트를 반환한다.
     * [onToken]은 완전한 UTF-8 문자 경계의 조각을 스트리밍으로 받고, false 를 반환하면 중단한다.
     * EOG(답변 종료)·컨텍스트 한계에서도 중단된다. 프롬프트가 컨텍스트를 넘으면 빈 문자열.
     */
    fun generate(
        prompt: String,
        maxTokens: Int = 256,
        onToken: ((String) -> Boolean)? = null,
    ): String {
        require(maxTokens > 0) { "maxTokens must be > 0" }
        val callback = onToken?.let { user ->
            NativePieceCallback { bytes -> user(String(bytes, Charsets.UTF_8)) }
        }
        val out = NativeLlm.nativeGenerate(checkOpen(), prompt.toByteArray(), maxTokens, callback)
        return String(out, Charsets.UTF_8)
    }

    /** [text]의 모델 토큰 수 — 프롬프트 예산 계산용. */
    fun countTokens(text: String): Int =
        NativeLlm.nativeCountTokens(checkOpen(), text.toByteArray())

    override fun close() {
        if (handle != 0L) {
            NativeLlm.nativeDestroy(handle)
            handle = 0L
        }
    }

    private fun checkOpen(): Long {
        check(handle != 0L) { "LlmEngine is closed" }
        return handle
    }

    companion object {
        /**
         * GGUF 모델을 로드한다. 1~2GB 모델은 수 초 걸릴 수 있다(백그라운드 권장).
         *
         * @param modelPath GGUF 파일 경로
         * @param contextSize 컨텍스트 토큰 한도(프롬프트+답변 합계)
         * @param threads 생성 스레드 수(기기 빅코어 수 정도가 적당)
         * @throws IOException 모델 로드 실패
         */
        fun create(
            modelPath: String,
            contextSize: Int = 2048,
            threads: Int = 4,
            temperature: Float = 0.7f,
            topK: Int = 40,
            topP: Float = 0.9f,
        ): LlmEngine {
            require(File(modelPath).exists()) { "모델 파일 없음: $modelPath" }
            require(contextSize >= 256) { "contextSize must be >= 256" }
            require(threads >= 1) { "threads must be >= 1" }
            val handle =
                NativeLlm.nativeCreate(modelPath, contextSize, threads, temperature, topK, topP)
            return LlmEngine(handle)
        }
    }
}
