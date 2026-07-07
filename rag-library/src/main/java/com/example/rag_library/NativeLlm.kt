package com.example.rag_library

/** JNI 스트리밍 콜백 — 완전한 UTF-8 문자 경계의 조각 단위로 호출된다. false 반환 시 생성 중단. */
internal fun interface NativePieceCallback {
    fun onPiece(pieceUtf8: ByteArray): Boolean
}

/**
 * librag-core.so 의 LLM(llama.cpp/GGUF) 브리지. 공개 래퍼는 [LlmEngine].
 *
 * 텍스트는 ByteArray(표준 UTF-8)로 주고받는다 — jstring(Modified UTF-8) 경유 시
 * 이모지 등 보충 평면 문자가 변형되는 문제를 피하기 위함.
 */
internal object NativeLlm {

    init {
        System.loadLibrary("rag-core")
    }

    // 대응 C++ 심볼: Java_com_example_rag_1library_NativeLlm_<이름> (@JvmStatic → jclass)
    @JvmStatic external fun nativeCreate(
        path: String,
        nCtx: Int,
        nThreads: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
    ): Long

    @JvmStatic external fun nativeDestroy(handle: Long)

    @JvmStatic external fun nativeCountTokens(handle: Long, textUtf8: ByteArray): Int

    @JvmStatic external fun nativeGenerate(
        handle: Long,
        promptUtf8: ByteArray,
        maxTokens: Int,
        callback: NativePieceCallback?,
    ): ByteArray
}
