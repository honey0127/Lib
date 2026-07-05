package com.example.rag_library

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class TextChunkerTest {

    @Test
    fun blankText_returnsEmpty() {
        assertEquals(emptyList<Chunk>(), TextChunker().chunk("d", "   \n  "))
    }

    @Test
    fun shortText_singleChunk() {
        val chunks = TextChunker().chunk("d", "안녕하세요.")
        assertEquals(1, chunks.size)
        assertEquals(Chunk("d", 0, "안녕하세요."), chunks[0])
    }

    @Test
    fun packsSentences_andCarriesOverlap() {
        val chunker = TextChunker(chunkSize = 20, overlap = 5)
        val text = "가나다라마바. 사아자차카타. 하거너더러머. 버서어저처커."
        val chunks = chunker.chunk("d", text)

        assertEquals(3, chunks.size)
        // 청크 순번은 0부터
        chunks.forEachIndexed { i, c -> assertEquals(i, c.index) }
        // 다음 청크는 이전 청크의 꼬리(overlap)로 시작한다
        assertTrue(chunks[1].text.startsWith(chunks[0].text.takeLast(5)))
        assertTrue(chunks[2].text.startsWith(chunks[1].text.takeLast(5)))
        // 소프트 한도: chunkSize + overlap 초과 금지
        assertTrue(chunks.all { it.text.length <= 25 })
    }

    @Test
    fun longSentenceWithoutDelimiter_isHardSplit() {
        val chunker = TextChunker(chunkSize = 30, overlap = 5)
        val chunks = chunker.chunk("d", "가".repeat(100))

        assertEquals(4, chunks.size)
        assertTrue(chunks.all { it.text.length <= 35 })
        // 겹침을 제외한 총 길이 = 원문 길이 (유실·중복 없음)
        val total = chunks.sumOf { it.text.length } - 5 * (chunks.size - 1)
        assertEquals(100, total)
    }

    @Test
    fun invalidOverlap_throws() {
        assertThrows(IllegalArgumentException::class.java) {
            TextChunker(chunkSize = 10, overlap = 10)
        }
    }
}
