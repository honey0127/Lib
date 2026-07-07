package com.example.rag_library

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class QwenPromptBuilderTest {

    @Test
    fun build_containsChatMlStructure_andContextInOrder() {
        val chunks = listOf(
            Chunk("a", 0, "첫 번째 문서 내용."),
            Chunk("b", 0, "두 번째 문서 내용."),
        )
        val prompt = QwenPromptBuilder.build("질문입니다?", chunks)

        assertTrue(prompt.startsWith("<|im_start|>system\n"))
        assertTrue(prompt.endsWith("<|im_start|>assistant\n"))
        assertTrue(prompt.contains("<|im_start|>user\n"))
        // 청크가 번호 순서대로 들어간다
        val i1 = prompt.indexOf("1. 첫 번째 문서 내용.")
        val i2 = prompt.indexOf("2. 두 번째 문서 내용.")
        assertTrue(i1 in 0 until i2)
        // 질문 포함 + user 턴이 닫힌다
        assertTrue(prompt.contains("질문: 질문입니다?<|im_end|>"))
        // im_start 3개(system/user/assistant)
        assertEquals(3, Regex("<\\|im_start\\|>").findAll(prompt).count())
    }

    @Test
    fun build_withoutChunks_omitsDocumentSection() {
        val prompt = QwenPromptBuilder.build("질문", emptyList())
        assertFalse(prompt.contains("[문서]"))
        assertTrue(prompt.contains("질문: 질문<|im_end|>"))
    }
}
